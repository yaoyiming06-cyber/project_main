import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PretrainedWeightsConfigTest(unittest.TestCase):
    def test_training_config_uses_pure_imagenet_22k_backbone(self):
        config = (PROJECT_ROOT / "configs" / "g1_seed42.yaml").read_text(encoding="utf-8")
        self.assertIn("model: convnext_base.fb_in22k\n", config)
        self.assertIn(
            "pretrained_checkpoint: weights/convnext_base.fb_in22k.safetensors\n",
            config,
        )
        requirements = (PROJECT_ROOT / "requirements-windows.txt").read_text(encoding="utf-8")
        self.assertIn("timm", requirements)
        self.assertIn("safetensors", requirements)

    def test_model_loader_checks_the_pinned_checkpoint_digest(self):
        model = (PROJECT_ROOT / "src" / "cxr_project" / "model.py").read_text(encoding="utf-8")
        self.assertIn("convnext_base.fb_in22k", model)
        self.assertIn("0f09df05894c7fd1e78569daaae9e839a7b3bc5666587a08ba1d8e8cea63a8a4", model)
        self.assertIn("safetensors.torch", model)
        self.assertIn("strict=True", model)
        self.assertIn("network.reset_classifier(0)", model)

    def test_layer_decay_levels_follow_timm_stem_and_four_stages(self):
        from cxr_project.train import _optimizer_layer_id
        self.assertEqual(_optimizer_layer_id("backbone.stem.0.weight"), 0)
        self.assertEqual(_optimizer_layer_id("backbone.stages.3.blocks.0.weight"), 4)
        self.assertEqual(_optimizer_layer_id("backbone.layers.2.blocks.0.weight"), 3)
        self.assertEqual(_optimizer_layer_id("backbone.blocks.11.norm.weight", 13), 12)
        self.assertEqual(_optimizer_layer_id("main_head.2.weight", 13), 13)


if __name__ == "__main__":
    unittest.main()
