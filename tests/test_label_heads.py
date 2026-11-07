"""Behavioral contracts for the pure-image label head experiments."""
import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cxr_project.config import load_config
from cxr_project.grid import build_configs
from cxr_project.model import ConvNeXtStudyClassifier, model_options


class TinySpatialBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)

    def forward_features(self, images):
        return self.conv(images)

    def forward_head(self, features, pre_logits=False):
        return features.mean((-2, -1))


def tiny_model(**options):
    return ConvNeXtStudyClassifier(
        backbone=TinySpatialBackbone(), feature_dim=8, pretrained=False,
        experts_enabled=False, use_view_embedding=False, image_chunk_size=2,
        query_dim=8, query_heads=2, query_layers=1, query_dropout=0.,
        head_dropout=0., **options,
    )


class LabelHeadTests(unittest.TestCase):
    def test_label_attention_can_select_different_images_per_label(self):
        from cxr_project.pooling import LabelAttentionPooling
        pool = LabelAttentionPooling(2, num_labels=2, hidden_dim=2)
        # Opposite label scores must lead to opposite image preferences.
        with torch.no_grad():
            pool.projection.weight.copy_(torch.eye(2)); pool.projection.bias.zero_()
            pool.score.weight.copy_(torch.tensor([[4., -4.], [-4., 4.]]))
            pool.score.bias.zero_()
        images = torch.tensor([[3., 0.], [0., 3.]])
        pooled = pool(images)
        self.assertGreater(pooled[0, 0].item(), pooled[0, 1].item())
        self.assertGreater(pooled[1, 1].item(), pooled[1, 0].item())
        torch.testing.assert_close(pool(images.flip(0)), pooled)
        torch.testing.assert_close(pool(images[:1]), images[:1].expand(2, -1))

    def test_all_new_structures_preserve_study_order_views_and_gradients(self):
        for head, pooling in [('linear', 'label_attention'),
                              ('label_query', 'attention'),
                              ('label_query', 'label_attention')]:
            with self.subTest(head=head, pooling=pooling):
                torch.manual_seed(42)
                model = tiny_model(head_type=head, study_pooling=pooling,
                                   gradient_checkpointing=True)
                images = torch.randn(3, 3, 4, 4, requires_grad=True)
                logits, experts = model(images, [0, 1, 3], torch.tensor([0, 1, 2]))
                self.assertEqual(logits.shape, (2, 10))
                self.assertEqual(experts.shape, (2, 0))
                logits.square().sum().backward()
                self.assertTrue(torch.isfinite(images.grad).all())
                self.assertGreater(model.backbone.conv.weight.grad.abs().sum().item(), 0)
                if head == 'label_query':
                    self.assertGreater(model.label_query.queries.grad.abs().sum().item(), 0)
                if pooling == 'label_attention':
                    self.assertGreater(model.label_pooling.score.weight.grad.abs().sum().item(), 0)
                model.eval()
                with torch.no_grad():
                    a = model(images, [0, 1, 3], torch.tensor([0, 1, 2]))[0]
                    b = model(images[[0, 2, 1]], [0, 1, 3], torch.tensor([5, 4, 3]))[0]
                    alone = model(images[:1], [0, 1], torch.tensor([3]))[0]
                torch.testing.assert_close(a, b)
                torch.testing.assert_close(a[:1], alone)

    def test_query_uses_spatial_distribution_beyond_global_mean(self):
        from cxr_project.heads import SpatialLabelQuery
        torch.manual_seed(7)
        head = SpatialLabelQuery(8, query_dim=8, num_heads=2, num_layers=1,
                                 dropout=0., num_labels=10).eval()
        a = torch.zeros(1, 8, 2, 2)
        b = torch.randn(1, 8, 2, 2)
        b -= b.mean((-2, -1), keepdim=True)
        torch.testing.assert_close(a.mean((-2, -1)), b.mean((-2, -1)), atol=1e-6, rtol=0)
        self.assertFalse(torch.allclose(head(a), head(b), atol=1e-5))
        with self.assertRaisesRegex(ValueError, 'spatial'):
            head(torch.zeros(1, 8))

    def test_new_checkpoint_round_trip_preserves_predictions(self):
        model = tiny_model(head_type='label_query', study_pooling='label_attention').eval()
        images = torch.randn(3, 3, 4, 4)
        with torch.no_grad():
            before = model(images, [0, 1, 3], torch.zeros(3, dtype=torch.long))[0]
        restored = tiny_model(head_type='label_query', study_pooling='label_attention').eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            after = restored(images, [0, 1, 3], torch.zeros(3, dtype=torch.long))[0]
        torch.testing.assert_close(before, after, atol=0, rtol=0)

    def test_default_legacy_model_has_only_historical_parameter_names(self):
        model = ConvNeXtStudyClassifier(backbone=TinySpatialBackbone(), feature_dim=8,
                                       pretrained=False)
        self.assertEqual(model_options({})['head_type'], 'linear')
        self.assertFalse(any(k.startswith(('label_query.', 'label_pooling.')) for k in model.state_dict()))
        self.assertEqual(tuple(model.main_head[2].weight.shape), (10, 8))
        clone = ConvNeXtStudyClassifier(backbone=TinySpatialBackbone(), feature_dim=8,
                                       pretrained=False, head_type='linear')
        clone.load_state_dict(model.state_dict(), strict=True)

    def test_query_rejects_vector_only_backbone_and_cached_vectors(self):
        model = tiny_model(head_type='label_query')
        with self.assertRaisesRegex(ValueError, 'spatial|query'):
            model.forward_from_features(torch.randn(3, 8), [0, 1, 3], torch.zeros(3, dtype=torch.long))

    def test_real_timm_convnext_spatial_maps_with_frozen_and_trainable_backbone(self):
        from timm.models.convnext import ConvNeXt
        from cxr_project.model import configure_backbone_training
        # Real timm feature/head contracts, with small widths for fast regression.
        for head, pooling in [('linear', 'label_attention'),
                              ('label_query', 'attention'),
                              ('label_query', 'label_attention')]:
            for frozen in (False, True):
                with self.subTest(head=head, pooling=pooling, frozen=frozen):
                    backbone = ConvNeXt(depths=(1, 1, 1, 1), dims=(8, 16, 32, 64),
                                        num_classes=0)
                    model = ConvNeXtStudyClassifier(
                        backbone=backbone, feature_dim=64, pretrained=False,
                        experts_enabled=False, use_view_embedding=False,
                        head_type=head, study_pooling=pooling, query_dim=16,
                        query_heads=4, query_dropout=0., head_dropout=0.,
                        image_chunk_size=2, gradient_checkpointing=True,
                    )
                    configure_backbone_training(model, freeze_backbone=frozen)
                    images = torch.randn(3, 3, 64, 64)
                    logits, experts = model(images, [0, 1, 3], torch.zeros(3, dtype=torch.long))
                    self.assertEqual(logits.shape, (2, 10))
                    self.assertEqual(experts.shape, (2, 0))
                    self.assertTrue(torch.isfinite(logits).all().item())
                    logits.square().mean().backward()
                    stem_grad = model.backbone.stem[0].weight.grad
                    if frozen:
                        self.assertIsNone(stem_grad)
                    else:
                        self.assertTrue(torch.isfinite(stem_grad).all().item())
                        self.assertGreater(stem_grad.abs().sum().item(), 0.)
                    if model.label_query is not None:
                        self.assertGreater(model.label_query.queries.grad.abs().sum().item(), 0.)
                    if model.label_pooling is not None:
                        self.assertGreater(model.label_pooling.score.weight.grad.abs().sum().item(), 0.)


class HeadConfigTests(unittest.TestCase):
    def fixture(self):
        c = yaml.safe_load((ROOT / 'configs/g1_seed42.yaml').read_text())
        c.update(head_type='label_query', study_pooling='attention', query_dim=256,
                 query_heads=4, query_layers=1, query_dropout=.2, experts_enabled=False,
                 hard_label_indices=[], expert_loss_weight=0., use_view_embedding=False,
                 view_filter=None, mixup_alpha=0.)
        return c

    def load(self, config):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'config.yaml'
            p.write_text(yaml.safe_dump(config))
            return load_config(p)

    def test_valid_query_and_label_attention_configs(self):
        c = self.fixture()
        self.assertEqual(self.load(c)['head_type'], 'label_query')
        c.update(head_type='linear', study_pooling='label_attention')
        self.assertEqual(self.load(c)['study_pooling'], 'label_attention')

    def test_windows_mac_baselines_only_change_runtime_factors(self):
        windows = load_config(ROOT / 'configs/head_baseline_l5.yaml')
        mac = load_config(ROOT / 'configs/head_baseline_l5_mac.yaml')
        runtime = {'_config_path', 'batch_size_studies', 'accum_steps', 'num_workers',
                   'pin_memory', 'amp', 'image_chunk_size'}
        self.assertEqual({k: v for k, v in windows.items() if k not in runtime},
                         {k: v for k, v in mac.items() if k not in runtime})
        self.assertEqual(mac['batch_size_studies'] * mac['accum_steps'], 16)

    def test_unsupported_combinations_fail_before_training(self):
        for override, message in [({'use_view_embedding': True}, 'view'),
                                  ({'view_filter': ['AP']}, 'view'),
                                  ({'experts_enabled': True}, 'expert'),
                                  ({'mixup_alpha': .2}, 'mixup'),
                                  ({'query_dim': 255}, 'query'),
                                  ({'query_layers': 0}, 'query'),
                                  ({'query_heads': True}, 'query'),
                                  ({'query_dropout': 1.}, 'query'),
                                  ({'model': 'vit_base_patch16_224.augreg_in21k'}, 'ConvNeXt')]:
            with self.subTest(override=override):
                c = self.fixture(); c.update(override)
                with self.assertRaisesRegex(ValueError, message): self.load(c)

    def test_head_recipes_reset_protocol_and_preserve_effective_batch(self):
        base = self.fixture()
        base.update(early_stopping_patience=5, validation_weights='auto',
                    batch_size_studies=2, accum_steps=8)
        recipes = ['HEAD_C0', 'HEAD_C1', 'HEAD_H1', 'HEAD_H2', 'HEAD_H3']
        generated = build_configs(base, recipes, [42, 2027, 3407], ROOT,
                                  Path('/tmp/head-test-runs'), stage='heads')
        self.assertEqual(len(generated), 15)
        for c in generated.values():
            self.assertEqual(c['batch_size_studies'] * c['accum_steps'], 16)
            self.assertEqual(c['epochs'], 15)
            self.assertEqual(c['early_stopping_patience'], 0)
            self.assertEqual(c['validation_weights'], 'ema')
            self.assertIsNone(c['clip_grad'])
            self.load(c)
        self.assertTrue(generated['heads_HEAD_C0_seed42']['experts_enabled'])
        self.assertFalse(generated['heads_HEAD_C1_seed42']['experts_enabled'])

    def test_inference_and_resume_reject_changed_structure_but_accept_old_defaults(self):
        spec = importlib.util.spec_from_file_location('label_head_predict', ROOT / 'scripts/predict_test.py')
        predict = importlib.util.module_from_spec(spec); spec.loader.exec_module(predict)
        from cxr_project.train import _validate_resume
        old = {'model': 'convnext_base.fb_in22k', 'input_size': 256}
        new = {**old, 'head_type': 'linear', 'query_dim': 256, 'query_layers': 1,
               'query_heads': 4, 'query_dropout': .2}
        predict.validate_checkpoint_config(new, {'config': old})
        _validate_resume({'config': old}, new, {})
        for key, value in [('head_type', 'label_query'), ('query_layers', 2)]:
            changed = {**new, key: value}
            with self.assertRaisesRegex(ValueError, key):
                predict.validate_checkpoint_config(changed, {'config': new})
            with self.assertRaisesRegex(ValueError, key):
                _validate_resume({'config': old}, changed, {})


class HeadPipelineTests(unittest.TestCase):
    def test_new_heads_train_ema_resume_and_predict_without_view_column(self):
        """Real training loop on synthetic images; mock only the large backbone."""
        from PIL import Image
        from cxr_project.train import train
        spec = importlib.util.spec_from_file_location('head_pipeline_predict', ROOT / 'scripts/predict_test.py')
        predict = importlib.util.module_from_spec(spec); spec.loader.exec_module(predict)

        def builder(config, **kwargs):
            keys = model_options(config)
            keys.update(query_dim=8, query_heads=2)
            return ConvNeXtStudyClassifier(backbone=TinySpatialBackbone(), feature_dim=8,
                                           pretrained=False, **keys)

        for head, pooling in [('linear', 'label_attention'), ('label_query', 'attention'),
                              ('label_query', 'label_attention')]:
            with self.subTest(head=head, pooling=pooling), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                manifest = root / 'data.csv'
                with manifest.open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['image_path', 'Subject_id', 'Study_id',
                                     *[f'label_{i}' for i in range(10)]])
                    for i in range(4):
                        for j in range(1 + i % 2):
                            name = f'image_{i}_{j}.png'
                            Image.new('RGB', (8, 8), (20 + i * 45, 30 + j * 50, 80)).save(root / name)
                            writer.writerow([name, f'p{i}', f's{i}', *([i % 2] * 10)])
                cfg = yaml.safe_load((ROOT / 'configs/head_baseline_l5.yaml').read_text())
                cfg.update(project_root=str(root), pretrained=False, head_type=head,
                           study_pooling=pooling, experts_enabled=False, hard_label_indices=[],
                           expert_loss_weight=0., query_dim=8, query_heads=2, query_dropout=0.,
                           head_dropout=0., input_size=32, epochs=2, warmup_epochs=0,
                           batch_size_studies=2, accum_steps=1, num_workers=0, amp=False,
                           pin_memory=False, image_root='.', train_manifest='data.csv',
                           valid_manifest='data.csv', output_dir='run', augmentation='G0')
                path = root / 'config.yaml'; path.write_text(yaml.safe_dump(cfg))
                import cxr_project.train as module
                original_epoch = module._train_epoch

                def interrupt_second_epoch(*args, **kwargs):
                    if args[7] == 1: raise KeyboardInterrupt('synthetic interruption')
                    return original_epoch(*args, **kwargs)

                with patch('cxr_project.train.build_model', side_effect=builder):
                    with patch('cxr_project.train._train_epoch', side_effect=interrupt_second_epoch):
                        with self.assertRaises(KeyboardInterrupt): train(path, device_name='cpu')
                    train(path, resume=root / 'run/checkpoint-last.pth', device_name='cpu')
                best = root / 'run/checkpoint-best-ema.pth'
                checkpoint = torch.load(best, map_location='cpu', weights_only=False)
                self.assertEqual(checkpoint['selected_weights'], 'ema')
                self.assertTrue((root / 'run/valid_best_ema_predictions.csv').is_file())
                output = root / 'prediction.csv'
                argv = ['predict_test.py', '--config', str(path), '--checkpoint', str(best),
                        '--data-root', str(root), '--manifest', str(manifest), '--labeled',
                        '--output', str(output), '--weights', 'ema', '--device', 'cpu']
                with patch.object(sys, 'argv', argv), patch.object(predict, 'build_model', side_effect=builder):
                    predict.main()
                with output.open() as f: rows = list(csv.DictReader(f))
                with (root / 'run/valid_best_ema_predictions.csv').open() as f:
                    reference = list(csv.DictReader(f))
                self.assertEqual([r['Study_id'] for r in rows], [r['Study_id'] for r in reference])
                for a, b in zip(rows, reference):
                    for i in range(10):
                        self.assertAlmostEqual(float(a[f'probability_{i}']), float(b[f'probability_{i}']), places=6)

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS is unavailable')
    def test_all_new_heads_backward_and_ema_on_mps(self):
        from cxr_project.train import ModelEMA
        for head, pooling in [('linear', 'label_attention'), ('label_query', 'attention'),
                              ('label_query', 'label_attention')]:
            with self.subTest(head=head, pooling=pooling):
                model = tiny_model(head_type=head, study_pooling=pooling,
                                   gradient_checkpointing=True).to('mps')
                ema = ModelEMA(model, .9)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
                images = torch.randn(3, 3, 4, 4, device='mps')
                views = torch.zeros(3, dtype=torch.long, device='mps')
                logits, _ = model(images, [0, 1, 3], views)
                logits.square().mean().backward(); optimizer.step(); ema.update(model)
                model.eval()
                with torch.inference_mode(), ema.use(model):
                    output = model(images, [0, 1, 3], views)[0]
                self.assertEqual(output.shape, (2, 10))
                self.assertTrue(torch.isfinite(output).all().item())


if __name__ == '__main__':
    unittest.main()
