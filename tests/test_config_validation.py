import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml

from cxr_project.config import load_config


class FiniteConfigValidationTests(unittest.TestCase):
    def test_reports_missing_label_policy_fields_as_config_error(self):
        config_path = PROJECT_ROOT / "configs" / "g1_seed42.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        del config["label_strategy"]
        del config["blank_policy"]
        del config["uncertain_policy"]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config = Path(temp_dir) / "missing_label_policy.yaml"
            temp_config.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing config keys"):
                load_config(temp_config)

    def test_rejects_nan_learning_rate(self):
        config_path = PROJECT_ROOT / "configs" / "g1_seed42.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["lr"] = float("nan")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config = Path(temp_dir) / "nan_lr.yaml"
            temp_config.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lr.*finite"):
                load_config(temp_config)

    def test_rejects_infinite_manual_positive_weight(self):
        config_path = PROJECT_ROOT / "configs" / "g1_seed42.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["pos_weight_mode"] = "manual"
        config["manual_pos_weights"] = [1.0] * 9 + [float("inf")]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config = Path(temp_dir) / "inf_pos_weight.yaml"
            temp_config.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manual_pos_weights.*finite"):
                load_config(temp_config)

    def test_rejects_non_finite_augmentation_value(self):
        config_path = PROJECT_ROOT / "configs" / "g1_seed42.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["augmentation_translate"] = float("inf")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config = Path(temp_dir) / "inf_translate.yaml"
            temp_config.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "augmentation_translate.*finite"):
                load_config(temp_config)


if __name__ == "__main__":
    unittest.main()
