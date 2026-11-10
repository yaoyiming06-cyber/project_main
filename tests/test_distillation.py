import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from cxr_project.config import load_config
from cxr_project.distillation import (
    TeacherCache, distillation_loss, validate_splits, write_teacher_cache,
)


ROOT = Path(__file__).resolve().parents[1]


def make_manifest(path, ids):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['image_path', 'Subject_id', 'Study_id', *[f'label_{i}' for i in range(10)]])
        for i, study in enumerate(ids):
            writer.writerow([f'{study}.png', f'patient_{study}', study, *([i % 2] * 10)])


def teacher_records():
    return [dict(name='teacher', checkpoint_sha256='a' * 64, config_sha256='b' * 64,
                 weight_source='ema', fusion_weight=1.0, input_size=32)]


class DistillationCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.train = self.root / 'train.csv'
        self.valid = self.root / 'valid.csv'
        self.ids = ['study_a', 'study_b']
        make_manifest(self.train, self.ids)
        make_manifest(self.valid, ['study_valid'])
        self.path = self.root / 'teacher.npz'

    def tearDown(self):
        self.temp.cleanup()

    def write(self, ids=None, probabilities=None):
        return write_teacher_cache(
            self.path, self.ids if ids is None else ids,
            np.array([[.1] * 10, [.8] * 10], dtype=np.float32) if probabilities is None else probabilities,
            train_manifest=self.train, valid_manifest=self.valid, teachers=teacher_records(),
        )

    def load(self, ids=None):
        return TeacherCache(self.path, train_manifest=self.train, valid_manifest=self.valid,
                            study_ids=self.ids if ids is None else ids)

    def test_lookup_aligns_shuffled_and_repeated_studies(self):
        self.write()
        cache = self.load()
        result = cache.lookup(['study_b', 'study_a', 'study_b'], torch.device('cpu'))
        torch.testing.assert_close(result[:, 0], torch.tensor([.8, .1, .8]))
        self.assertEqual(cache.provenance['cache_sha256'], hashlib.sha256(self.path.read_bytes()).hexdigest())
        with self.assertRaisesRegex(ValueError, 'Study'):
            cache.lookup(['unknown'], torch.device('cpu'))

    def test_cache_rejects_missing_duplicate_extra_and_invalid_probabilities(self):
        for ids, probabilities in [
            (['study_a'], np.zeros((1, 10))),
            (['study_a', 'study_a'], np.zeros((2, 10))),
            (['study_a', 'extra'], np.zeros((2, 10))),
            (self.ids, np.full((2, 10), np.nan)),
            (self.ids, np.full((2, 10), 1.1)),
            (self.ids, np.zeros((2, 9))),
        ]:
            with self.subTest(ids=ids, shape=probabilities.shape):
                with self.assertRaises(ValueError):
                    self.write(ids, probabilities)
        self.assertFalse(self.path.exists())

    def test_rejects_payload_manifest_label_order_tampering_and_overwrite(self):
        self.write()
        original = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.write()
        self.path.write_bytes(original + b'changed')
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.load()
        self.path.write_bytes(original)
        original_manifest = self.train.read_text()
        self.train.write_text(original_manifest + '\n')
        with self.assertRaisesRegex(ValueError, 'manifest'):
            self.load()
        self.train.write_text(original_manifest)
        sidecar = self.path.with_suffix('.json')
        metadata = json.loads(sidecar.read_text())
        metadata['labels'] = list(reversed(metadata['labels']))
        sidecar.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'label'):
            self.load()

    def test_split_guard_rejects_study_and_patient_overlap(self):
        make_manifest(self.valid, ['study_a'])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_splits(self.train, self.valid)
        make_manifest(self.valid, ['study_valid'])
        text = self.valid.read_text().replace('patient_study_valid', 'patient_study_a')
        self.valid.write_text(text)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_splits(self.train, self.valid)


class DistillationLossTests(unittest.TestCase):
    def test_independent_sigmoid_soft_targets_temperature_and_masked_gradients(self):
        logits = torch.tensor([[2., -1., 3.]], requires_grad=True)
        probabilities = torch.tensor([[.9, .8, .2]], requires_grad=True)
        targets = torch.tensor([[1., 0., -1.]])
        loss = distillation_loss(logits, probabilities, targets, temperature=2.)
        softened = torch.sigmoid(torch.logit(probabilities.detach()) / 2.)
        expected = F.binary_cross_entropy_with_logits(logits[:, :2] / 2., softened[:, :2]) * 4
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(logits.grad[0, 2].item(), 0.)
        self.assertIsNone(probabilities.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_unknown_targets_yield_differentiable_zero_and_endpoints_are_finite(self):
        logits = torch.zeros((1, 2), requires_grad=True)
        loss = distillation_loss(logits, torch.tensor([[0., 1.]]), torch.full((1, 2), -1.))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))
        self.assertTrue(torch.isfinite(distillation_loss(logits, torch.tensor([[0., 1.]]), torch.ones(1, 2))))


class DistillationConfigTests(unittest.TestCase):
    def config(self, **updates):
        config = yaml.safe_load((ROOT / 'configs/head_baseline_l5.yaml').read_text())
        config.update(updates)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'config.yaml'
            path.write_text(yaml.safe_dump(config))
            return load_config(path)

    def test_disabled_defaults_need_no_cache_and_active_config_is_explicit(self):
        config = self.config()
        self.assertEqual(config['distillation_alpha'], 0.)
        self.assertIsNone(config['distillation_cache'])
        self.assertEqual(self.config(distillation_cache='teacher.npz', distillation_alpha=.5,
                                     distillation_temperature=2.)['distillation_alpha'], .5)

    def test_rejects_invalid_and_incompatible_distillation(self):
        for updates in [
            dict(distillation_alpha=.5),
            dict(distillation_cache='teacher.npz'),
            dict(distillation_alpha=1.1, distillation_cache='teacher.npz'),
            dict(distillation_alpha=True, distillation_cache='teacher.npz'),
            dict(distillation_alpha=.5, distillation_cache='teacher.npz', distillation_temperature=0),
            dict(distillation_alpha=.5, distillation_cache='teacher.npz', mixup_alpha=.2),
            dict(distillation_alpha=.5, distillation_cache='teacher.npz', use_view_embedding=True),
            dict(distillation_alpha=.5, distillation_cache='teacher.npz', train_label_indices=[0]),
            dict(distillation_alpha=.5, distillation_cache='teacher.npz', final_training=True, valid_manifest=None),
        ]:
            with self.subTest(updates=updates):
                with self.assertRaisesRegex(ValueError, 'distillation'):
                    self.config(**updates)


if __name__ == '__main__':
    unittest.main()
