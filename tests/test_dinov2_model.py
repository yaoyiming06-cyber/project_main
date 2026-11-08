import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from cxr_project.config import load_config
from cxr_project.general_weights import DINOV2_MODEL, DINOV2_URL
from cxr_project.model import SUPPORTED_MODELS, build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DINOv2ModelTests(unittest.TestCase):
    def setUp(self):
        from cxr_project import dinov2_weights

        self.pinned_weights_patch = patch.multiple(
            dinov2_weights,
            WEIGHTS_BYTES=0,
            WEIGHTS_SHA256="0" * 64,
        )
        self.pinned_weights_patch.start()

    def tearDown(self):
        self.pinned_weights_patch.stop()

    def test_dinov2_is_an_explicitly_supported_general_backbone(self):
        self.assertIn("LVD-142M", SUPPORTED_MODELS[DINOV2_MODEL])

    def test_invalid_checkpoint_provenance_fails_before_timm_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "dinov2_vitb14_pretrain.pth"
            checkpoint.write_bytes(b"bad local file")
            config = {
                "model": DINOV2_MODEL,
                "pretrained": True,
                "pretrained_checkpoint": str(checkpoint),
                "pretrained_sha256": hashlib.sha256(b"bad local file").hexdigest(),
                "input_size": 252,
                "use_view_embedding": False,
                "view_filter": None,
            }
            with patch("timm.create_model") as create_model:
                with self.assertRaisesRegex(ValueError, "provenance"):
                    build_model(config)
            create_model.assert_not_called()

    def test_valid_checkpoint_is_passed_as_local_timm_file(self):
        class TinyBackbone(torch.nn.Module):
            num_features = 768

            def forward(self, images):
                return images.new_zeros((images.shape[0], self.num_features))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "dinov2_vitb14_pretrain.pth"
            payload = b"verified synthetic checkpoint"
            checkpoint.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            sidecar = {
                "model": DINOV2_MODEL,
                "url": DINOV2_URL,
                "bytes": len(payload),
                "sha256": digest,
                "path": str(checkpoint.resolve()),
            }
            checkpoint.with_suffix(checkpoint.suffix + ".source.json").write_text(
                json.dumps(sidecar), encoding="utf-8",
            )
            from cxr_project import dinov2_weights
            dinov2_weights.WEIGHTS_BYTES = len(payload)
            dinov2_weights.WEIGHTS_SHA256 = digest
            config = {
                "model": DINOV2_MODEL,
                "pretrained": True,
                "pretrained_checkpoint": str(checkpoint),
                "pretrained_sha256": digest,
                "input_size": 252,
                "use_view_embedding": False,
                "view_filter": None,
                "experts_enabled": False,
                "hard_label_indices": [],
                "gradient_checkpointing": False,
            }
            with patch("timm.create_model", return_value=TinyBackbone()) as create_model:
                model = build_model(config)
            kwargs = create_model.call_args.kwargs
            self.assertTrue(kwargs["pretrained"])
            self.assertTrue(kwargs["dynamic_img_size"])
            self.assertEqual(kwargs["pretrained_cfg_overlay"]["file"], str(checkpoint))
            self.assertEqual(model.feature_dim, 768)

    def test_untrained_dinov2_classifier_accepts_all_patch_aligned_sizes(self):
        model = build_model({
            "model": DINOV2_MODEL,
            "pretrained": False,
            "input_size": 224,
            "drop_path": 0.0,
            "experts_enabled": False,
            "hard_label_indices": [],
            "expert_loss_weight": 0.0,
            "study_pooling": "mean",
            "use_view_embedding": False,
            "gradient_checkpointing": False,
            "image_chunk_size": 1,
            "head_dropout": 0.0,
        }).eval()
        self.assertEqual(model.feature_dim, 768)
        with torch.inference_mode():
            for size in (224, 252, 280):
                logits, expert_logits = model(
                    torch.randn(1, 3, size, size),
                    torch.tensor([0, 1]),
                    torch.zeros(1, dtype=torch.long),
                )
                self.assertEqual(tuple(logits.shape), (1, 10))
                self.assertEqual(tuple(expert_logits.shape), (1, 0))
                self.assertTrue(torch.isfinite(logits).all())

    def test_config_requires_patch_aligned_dinov2_resolution(self):
        baseline = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "head_baseline_l5.yaml").read_text(encoding="utf-8")
        )
        baseline["model"] = DINOV2_MODEL
        baseline["input_size"] = 256
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "invalid.yaml"
            config_path.write_text(yaml.safe_dump(baseline), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "divisible by 14"):
                load_config(config_path)

    def test_config_accepts_the_three_grid_resolutions(self):
        baseline = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "head_baseline_l5.yaml").read_text(encoding="utf-8")
        )
        baseline["model"] = DINOV2_MODEL
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "valid.yaml"
            for size in (224, 252, 280):
                baseline["input_size"] = size
                config_path.write_text(yaml.safe_dump(baseline), encoding="utf-8")
                self.assertEqual(load_config(config_path)["input_size"], size)

    def test_pretrained_normalization_uses_timm_dinov2_metadata(self):
        from cxr_project.dataset import build_transform

        transform = build_transform(False, 252, {
            "model": DINOV2_MODEL,
            "normalization": "pretrained",
        })
        normalization = transform.transforms[-1]
        self.assertEqual(tuple(normalization.mean), (0.485, 0.456, 0.406))
        self.assertEqual(tuple(normalization.std), (0.229, 0.224, 0.225))


if __name__ == "__main__":
    unittest.main()
