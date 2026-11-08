import tempfile
import unittest
from pathlib import Path


class DINOv2ExportTests(unittest.TestCase):
    def test_fp16_export_preserves_ema_config_and_normalization(self):
        import torch
        from cxr_project.evaluation_tools import export_fp16

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "checkpoint-best-ema.pth"
            output = root / "checkpoint-best-ema-fp16.pth"
            torch.save({
                "model": {"weight": torch.tensor([1.0])},
                "ema": {"weight": torch.tensor([2.0])},
                "ema_active": True,
                "selected_weights": "ema",
                "config": {
                    "model": "vit_base_patch14_dinov2",
                    "normalization": "pretrained",
                },
                "normalization_stats": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                "optimizer": {"state": {}},
            }, source)

            export_fp16(source, output, weights="ema")
            exported = torch.load(output, weights_only=False)

        self.assertEqual(exported["model"]["weight"].item(), 2.0)
        self.assertEqual(exported["model"]["weight"].dtype, torch.float16)
        self.assertEqual(exported["config"]["model"], "vit_base_patch14_dinov2")
        self.assertEqual(exported["normalization"]["mean"], [0.485, 0.456, 0.406])
        self.assertNotIn("optimizer", exported)


if __name__ == "__main__":
    unittest.main()
