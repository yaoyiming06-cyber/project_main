import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from torch import nn
from cxr_project.model import ConvNeXtStudyClassifier

class GridModelTests(unittest.TestCase):
    def model(self, **kwargs):
        return ConvNeXtStudyClassifier(backbone=nn.Linear(4, 4), feature_dim=4, pretrained=False, **kwargs)

    def test_no_experts_has_empty_logits_and_main_gradient(self):
        model = self.model(experts_enabled=False)
        features = torch.randn(3, 4, requires_grad=True)
        main, expert = model.forward_from_features(features, [0, 1, 3], torch.zeros(3, dtype=torch.long))
        self.assertEqual(expert.shape, (2, 0))
        self.assertEqual(model.hard_label_indices, ())
        main.sum().backward()
        self.assertIsNotNone(features.grad)

    def test_mean_pooling_ignores_views_and_order(self):
        model = self.model(study_pooling='mean', use_view_embedding=False).eval()
        x = torch.randn(3, 4)
        a = model.forward_from_features(x, [0, 3], torch.tensor([0, 1, 2]))
        b = model.forward_from_features(x.flip(0), [0, 3], torch.tensor([3, 2, 1]))
        torch.testing.assert_close(a[0], b[0])
        self.assertFalse(any('attention' in key or 'view_embedding' in key for key in model.state_dict()))

    def test_dropout_and_default_checkpoint_layout(self):
        baseline = self.model()
        explicit = self.model(head_dropout=.4, expert_dropout=.3)
        explicit.load_state_dict(baseline.state_dict(), strict=True)
        self.assertEqual(explicit.main_head[1].p, .4)
        self.assertEqual(explicit.expert_heads[0][3].p, .3)

    def test_partial_unfreezing_keeps_early_stages_frozen(self):
        from cxr_project.model import configure_backbone_training
        backbone = nn.Module()
        backbone.stem = nn.Linear(4, 4)
        backbone.stages = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
        backbone.norm_pre = nn.LayerNorm(4)
        model = ConvNeXtStudyClassifier(backbone=backbone, pretrained=False, feature_dim=4)
        configure_backbone_training(model, unfreeze_last_stages=2)
        self.assertFalse(backbone.stem.weight.requires_grad)
        self.assertFalse(backbone.stages[1].weight.requires_grad)
        self.assertTrue(backbone.stages[2].weight.requires_grad)
        self.assertTrue(backbone.norm_pre.weight.requires_grad)
        self.assertTrue(model.main_head[2].weight.requires_grad)

    def test_adapter_starts_as_identity_delta_and_trains(self):
        from cxr_project.adapters import AdaptFormerMLP
        base = nn.Linear(4, 4)
        wrapped = AdaptFormerMLP(base, 4, bottleneck=2, dropout=0)
        x = torch.randn(2, 3, 4)
        torch.testing.assert_close(wrapped(x), base(x))
        wrapped(x).sum().backward()
        self.assertGreater(wrapped.adapter.up.weight.grad.abs().sum().item(), 0)

    def test_unknown_model_rejected_before_loading(self):
        from cxr_project.model import build_model
        with self.assertRaisesRegex(ValueError, 'Unsupported model'):
            build_model({'model': 'medical_model', 'pretrained': False})

    def test_attention_without_view_embedding_is_view_invariant(self):
        model = self.model(use_view_embedding=False).eval()
        x = torch.randn(3, 4)
        a = model.forward_from_features(x, [0, 3], torch.tensor([0, 1, 2]))
        b = model.forward_from_features(x, [0, 3], torch.tensor([3, 4, 1]))
        torch.testing.assert_close(a[0], b[0])
        self.assertEqual(model.attention[0].normalized_shape, (4,))

    def test_tiny_vit_adapter_only_backbone_gradients(self):
        from timm.models.vision_transformer import VisionTransformer
        from cxr_project.adapters import install_adaptformer
        from cxr_project.model import configure_backbone_training
        backbone = VisionTransformer(img_size=32, patch_size=16, embed_dim=24,
                                     depth=2, num_heads=3, num_classes=0)
        install_adaptformer(backbone, bottleneck=4, dropout=0)
        model = ConvNeXtStudyClassifier(backbone=backbone, pretrained=False, feature_dim=24,
                                       gradient_checkpointing=True)
        configure_backbone_training(model, adaptformer=True)
        main, experts = model(torch.randn(2, 3, 32, 32), [0, 1, 2], torch.tensor([0, 1]))
        (main.sum() + experts.sum()).backward()
        for name, parameter in backbone.named_parameters():
            self.assertEqual(parameter.requires_grad, '.adapter.' in name)
            if '.adapter.' not in name:
                self.assertIsNone(parameter.grad)
        self.assertGreater(backbone.blocks[0].mlp.adapter.up.weight.grad.abs().sum().item(), 0)

    def test_full_freeze_leaves_classifier_trainable(self):
        from cxr_project.model import configure_backbone_training
        model = self.model()
        configure_backbone_training(model, freeze_backbone=True)
        self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.main_head.parameters()))

if __name__ == '__main__': unittest.main()
