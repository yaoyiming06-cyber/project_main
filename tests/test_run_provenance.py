import hashlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_project.provenance import (
    dataset_normalization_source,
    git_working_tree_state,
    validate_dataset_normalization_cache,
)


class DatasetNormalizationCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_root = self.root / "images"
        self.image_root.mkdir()
        self.manifest = self.root / "train.csv"
        self.manifest.write_text("image_path,label\na.jpg,0\n", encoding="utf-8")
        self.source = dataset_normalization_source(
            self.manifest, self.image_root, input_size=224,
        )
        self.cache = {
            "name": "dataset",
            "mean": [0.4, 0.5, 0.6],
            "std": [0.2, 0.3, 0.4],
            "input_size": 224,
            "images": 1,
            "pixels_per_image": 224 * 224,
            "source": self.source,
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cache_is_valid_when_manifest_and_image_root_match(self):
        validated = validate_dataset_normalization_cache(
            self.cache, self.source, self.root / "dataset_normalization.json",
        )
        self.assertEqual(validated, self.cache)

    def test_cache_is_rejected_when_training_manifest_changes(self):
        self.manifest.write_text("image_path,label\nb.jpg,1\n", encoding="utf-8")
        current_source = dataset_normalization_source(
            self.manifest, self.image_root, input_size=224,
        )
        self.assertNotEqual(self.source["train_manifest_sha256"], current_source["train_manifest_sha256"])
        with self.assertRaisesRegex(ValueError, "different training data"):
            validate_dataset_normalization_cache(
                self.cache, current_source, self.root / "dataset_normalization.json",
            )

    def test_cache_is_rejected_when_image_root_changes(self):
        other_root = self.root / "other_images"
        other_root.mkdir()
        current_source = dataset_normalization_source(
            self.manifest, other_root, input_size=224,
        )
        with self.assertRaisesRegex(ValueError, "different training data"):
            validate_dataset_normalization_cache(
                self.cache, current_source, self.root / "dataset_normalization.json",
            )

    def test_legacy_cache_without_source_is_rejected(self):
        legacy_cache = {key: value for key, value in self.cache.items() if key != "source"}
        with self.assertRaisesRegex(ValueError, "missing provenance"):
            validate_dataset_normalization_cache(
                legacy_cache, self.source, self.root / "dataset_normalization.json",
            )

    def test_cache_rejects_non_finite_normalization_values(self):
        invalid_cache = {**self.cache, "std": [0.2, float("nan"), 0.4]}
        with self.assertRaisesRegex(ValueError, "must be finite"):
            validate_dataset_normalization_cache(
                invalid_cache, self.source, self.root / "dataset_normalization.json",
            )


@unittest.skipUnless(shutil.which("git"), "Git is required for repository fingerprint tests")
class GitWorkingTreeStateTests(unittest.TestCase):
    def test_records_clean_and_dirty_state_and_tracked_diff_digest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            tracked = root / "tracked.py"
            tracked.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "tracked.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-m", "initial"], check=True, capture_output=True)

            clean_state = git_working_tree_state(root)
            self.assertFalse(clean_state["working_tree_dirty"])
            self.assertEqual(clean_state["tracked_diff_sha256"], hashlib.sha256(b"").hexdigest())
            self.assertEqual(clean_state["untracked_paths"], [])

            tracked.write_text("value = 2\n", encoding="utf-8")
            (root / "new.py").write_text("value = 3\n", encoding="utf-8")
            dirty_state = git_working_tree_state(root)
            self.assertTrue(dirty_state["working_tree_dirty"])
            self.assertNotEqual(dirty_state["tracked_diff_sha256"], clean_state["tracked_diff_sha256"])
            self.assertEqual(dirty_state["untracked_paths"], ["new.py"])


if __name__ == "__main__":
    unittest.main()
