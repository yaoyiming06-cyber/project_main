import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml
from PIL import Image

from cxr_project.config import load_config
from cxr_project.distillation import TeacherCache, write_teacher_cache
from cxr_project.distillation_teachers import build_teacher_cache
from cxr_project.evaluation_tools import export_fp16
from cxr_project.model import ConvNeXtStudyClassifier, file_sha256
from cxr_project.train import train
from scripts.predict_test import inference_weights, predict_dataset


ROOT = Path(__file__).resolve().parents[1]


def tiny(config, **kwargs):
    return ConvNeXtStudyClassifier(
        backbone=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()),
        feature_dim=3, pretrained=False, hard_label_indices=[], experts_enabled=False,
        gradient_checkpointing=False, image_chunk_size=2, use_view_embedding=False,
    )


class DistillationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for split in ('train', 'valid'):
            with (self.root / f'{split}.csv').open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['image_path', 'Subject_id', 'Study_id', *[f'label_{i}' for i in range(10)]])
                for i in range(4):
                    study = f'{split}_{i}'
                    for view in range(1 + i % 2):
                        name = f'{study}_{view}.png'
                        Image.new('RGB', (16, 16), (30 + i * 50,) * 3).save(self.root / name)
                        writer.writerow([name, f'patient_{study}', study, *([i % 2] * 10)])
        self.config = yaml.safe_load((ROOT / 'configs/head_baseline_l5.yaml').read_text())
        self.config.update(project_root=str(self.root), pretrained=False, input_size=16,
                           epochs=2, warmup_epochs=0, num_workers=0, batch_size_studies=2,
                           accum_steps=1, amp=False, gradient_checkpointing=False,
                           image_root='.', train_manifest='train.csv', valid_manifest='valid.csv',
                           test_manifest='DO_NOT_READ.csv', output_dir='run', experts_enabled=False,
                           hard_label_indices=[], pos_weight_mode='unit', augmentation='G0',
                           distillation_alpha=.5, distillation_cache='teacher.npz')
        self.config_path = self.root / 'config.yaml'
        self.save_config()
        self.recipe = {'format': 'cxr_distillation_teachers_v1',
                       'train_manifest_sha256': file_sha256(self.root / 'train.csv'),
                       'valid_manifest_sha256': file_sha256(self.root / 'valid.csv'), 'models': []}
        self.expected = []
        for i in range(2):
            cfg = {**self.config, 'distillation_alpha': 0., 'distillation_cache': None}
            path = self.root / f'teacher_{i}.yaml'
            path.write_text(yaml.safe_dump(cfg))
            model = tiny(cfg)
            with torch.no_grad():
                model.main_head[-1].weight.zero_()
                model.main_head[-1].bias.fill_(float(i * 2))
            raw = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
            checkpoint = self.root / f'teacher_{i}.pth'
            torch.save(dict(model=raw, ema=model.state_dict(), ema_active=True,
                            selected_weights='ema', config=cfg), checkpoint)
            self.recipe['models'].append(dict(name=f'teacher_{i}', config=path.name,
                checkpoint=checkpoint.name, checkpoint_sha256=file_sha256(checkpoint),
                config_sha256=file_sha256(path), weights='ema', fusion_weight=i + 1))
            self.expected.append(float(torch.tensor(float(i * 2)).sigmoid()))
        self.recipe_path = self.root / 'teachers.json'
        self.recipe_path.write_text(json.dumps(self.recipe))
        self.output = self.root / 'teacher.npz'

    def tearDown(self):
        self.temp.cleanup()

    def save_config(self):
        self.config_path.write_text(yaml.safe_dump(self.config))

    def generate(self):
        return build_teacher_cache(self.config_path, self.recipe_path, self.output, device_name='cpu')

    def test_teacher_cache_uses_train_only_ema_probabilities_and_reuses_progress(self):
        with patch('cxr_project.distillation_teachers.build_model', side_effect=tiny) as builder:
            metadata = self.generate()
            self.assertEqual(builder.call_count, 2)
        cache = TeacherCache(self.output, train_manifest=self.root / 'train.csv',
                             valid_manifest=self.root / 'valid.csv', study_ids=[f'train_{i}' for i in range(4)])
        expected = (self.expected[0] + 2 * self.expected[1]) / 3
        torch.testing.assert_close(cache.lookup(['train_3', 'train_0'], 'cpu'), torch.full((2, 10), expected))
        self.assertEqual([x['weight_source'] for x in metadata['teachers']], ['ema', 'ema'])
        self.output.unlink()
        self.output.with_suffix('.json').unlink()
        with patch('cxr_project.distillation_teachers.build_model', side_effect=AssertionError('must reuse progress')):
            self.generate()

    def test_preflight_refuses_bad_checkpoint_hash_and_view_enabled_teacher(self):
        self.recipe['models'][1]['checkpoint_sha256'] = '0' * 64
        self.recipe_path.write_text(json.dumps(self.recipe))
        with patch('cxr_project.distillation_teachers.build_model', side_effect=AssertionError('preflight first')):
            with self.assertRaisesRegex(ValueError, 'hash'):
                self.generate()
        self.recipe['models'][1]['checkpoint_sha256'] = file_sha256(self.root / 'teacher_1.pth')
        cfg = self.root / 'teacher_0.yaml'
        data = yaml.safe_load(cfg.read_text())
        data['use_view_embedding'] = True
        cfg.write_text(yaml.safe_dump(data))
        self.recipe['models'][0]['config_sha256'] = file_sha256(cfg)
        self.recipe_path.write_text(json.dumps(self.recipe))
        with self.assertRaisesRegex(ValueError, 'image-only'):
            self.generate()
        self.assertFalse(self.output.exists())

    def test_training_resume_and_fp16_inference_are_independent_of_teachers(self):
        import cxr_project.train as module
        from cxr_project.dataset import StudyDataset, build_transform
        with patch('cxr_project.distillation_teachers.build_model', side_effect=tiny):
            self.generate()
        with patch('cxr_project.train.build_model', side_effect=tiny):
            train(self.config_path, output_dir=self.root / 'continuous', device_name='cpu')
        original = module._train_epoch
        def interrupted(*args, **kwargs):
            if args[7] == 1:
                raise KeyboardInterrupt('simulate interruption')
            return original(*args, **kwargs)
        with patch('cxr_project.train.build_model', side_effect=tiny), patch('cxr_project.train._train_epoch', side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                train(self.config_path, output_dir=self.root / 'interrupted', device_name='cpu')
        resume = self.root / 'interrupted/checkpoint-last.pth'
        with patch('cxr_project.train.build_model', side_effect=tiny):
            train(self.config_path, output_dir=resume.parent, resume=resume, device_name='cpu')
        a = torch.load(self.root / 'continuous/checkpoint-last.pth', weights_only=False)
        b = torch.load(resume, weights_only=False)
        for key, value in a['model'].items():
            torch.testing.assert_close(value, b['model'][key], rtol=0, atol=0)
        self.assertIn('distillation', a['data'])
        with (self.root / 'continuous/distillation_metrics.csv').open() as handle:
            metrics = list(csv.DictReader(handle))
        self.assertEqual(len(metrics), 2)
        self.assertGreater(float(metrics[0]['soft_loss']), 0)
        exported = self.root / 'student_fp16.pth'
        export_fp16(self.root / 'continuous/checkpoint-best-ema.pth', exported, weights='ema')
        self.output.unlink()
        self.output.with_suffix('.json').unlink()
        for p in self.root.glob('teacher_*.pth'):
            p.unlink()
        checkpoint = torch.load(exported, weights_only=False)
        self.assertNotIn('optimizer', checkpoint)
        state, source = inference_weights(checkpoint, 'ema')
        model = tiny(self.config)
        model.load_state_dict(state)
        cfg = load_config(self.config_path)
        dataset = StudyDataset(self.root / 'valid.csv', self.root, build_transform(False, 16, cfg),
                               use_view_embedding=False)
        ids, probabilities, _ = predict_dataset(model, dataset, cfg, torch.device('cpu'))
        self.assertEqual(len(ids), 4)
        self.assertTrue(torch.isfinite(probabilities).all())

    def test_resume_rejects_a_different_valid_cache(self):
        with patch('cxr_project.distillation_teachers.build_model', side_effect=tiny):
            self.generate()
        self.config['epochs'] = 1
        self.save_config()
        with patch('cxr_project.train.build_model', side_effect=tiny):
            train(self.config_path, device_name='cpu')
        teachers = json.loads(self.output.with_suffix('.json').read_text())['teachers']
        self.output.unlink()
        self.output.with_suffix('.json').unlink()
        write_teacher_cache(self.output, [f'train_{i}' for i in range(4)], np.full((4, 10), .1),
                            train_manifest=self.root / 'train.csv', valid_manifest=self.root / 'valid.csv',
                            teachers=teachers)
        with self.assertRaisesRegex(ValueError, 'distillation'):
            train(self.config_path, resume=self.root / 'run/checkpoint-last.pth', device_name='cpu')


if __name__ == '__main__':
    unittest.main()
