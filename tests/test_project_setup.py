import csv
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_project.config import load_config
from cxr_project.dataset import StudyDataset, build_transform, collate_studies
from cxr_project.manifest import read_study_manifest


class ProjectSetupTests(unittest.TestCase):
    def test_g1_config_keeps_seed_and_training_schedule(self):
        config = load_config(PROJECT_ROOT / "configs" / "g1_seed42.yaml")
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["epochs"], 10)
        self.assertEqual(config["warmup_epochs"], 5)
        self.assertEqual(config["batch_size_studies"], 2)
        self.assertEqual(config["accum_steps"], 8)
        self.assertEqual(config["lr"], 1e-3)
        self.assertEqual(config["min_lr"], 1e-6)
        self.assertEqual(config["weight_decay"], 0.05)
        self.assertEqual(config["hard_label_indices"], [0, 3, 7])

    def test_config_selects_pure_imagenet_22k_convnext_base_weights(self):
        config = load_config(PROJECT_ROOT / "configs" / "g1_seed42.yaml")
        self.assertEqual(config["model"], "convnext_base.fb_in22k")
        self.assertEqual(
            config["pretrained_checkpoint"],
            "weights/convnext_base.fb_in22k.safetensors",
        )
        self.assertIn("timm", (PROJECT_ROOT / "requirements-windows.txt").read_text())

    def test_bundled_training_manifest_excludes_both_corrupt_images(self):
        manifest = PROJECT_ROOT / "data" / "manifests" / "train_manifest.csv"
        excluded_file = PROJECT_ROOT / "data" / "manifests" / "excluded_train_paths.txt"
        excluded = set(excluded_file.read_text(encoding="utf-8").splitlines())
        self.assertEqual(excluded, {
            "train/subject_01922/study_02467/pic_004044.jpg",
            "train/subject_05239/study_13172/pic_021690.jpg",
        })
        studies = read_study_manifest(manifest, labeled=True)
        used_paths = {path for study in studies for path in study.image_paths}
        self.assertEqual(len(used_paths), 30342)
        self.assertTrue(excluded)
        self.assertTrue(used_paths.isdisjoint(excluded))

    def test_study_dataset_loads_all_views_and_collates_offsets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            labels = {f"label_{index}": "0" for index in range(10)}
            rows = []
            for study_number, views in ((1, ("AP", "LATERAL")), (2, ("PA",))):
                subject = f"subject{study_number}"
                study = f"study{study_number}"
                for image_number, view in enumerate(views):
                    relative = f"train/{subject}/{study}/{image_number}.png"
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (10, 8), color=(image_number, 20, 30)).save(path)
                    rows.append({
                        "image_path": relative,
                        "Subject_id": subject,
                        "Study_id": study,
                        "Dicom_id": str(image_number),
                        "ViewPosition": view,
                        **labels,
                    })
            manifest = root / "train.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            dataset = StudyDataset(
                manifest, root, transform=build_transform(False, input_size=8), labeled=True,
            )
            batch = collate_studies([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["images"].shape), (3, 3, 8, 8))
            self.assertEqual(batch["offsets"].tolist(), [0, 2, 3])
            self.assertEqual(batch["targets"].shape, (2, 10))


if __name__ == "__main__":
    unittest.main()
