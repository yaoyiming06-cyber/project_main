import csv
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_project.manifest import compute_pos_weights, read_study_manifest

try:
    import torch
except ImportError:
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "image_path", "Subject_id", "Study_id", "Dicom_id", "ViewPosition",
        *[f"label_{index}" for index in range(10)],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class ManifestContractsTest(unittest.TestCase):
    def test_groups_all_images_by_study_and_keeps_view_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "train.csv"
            positive = {f"label_{index}": "0" for index in range(10)}
            positive["label_1"] = "1"
            rows = [
                {"image_path": "train/s1/study1/a.jpg", "Subject_id": "s1",
                 "Study_id": "study1", "Dicom_id": "a", "ViewPosition": "AP", **positive},
                {"image_path": "train/s1/study1/b.jpg", "Subject_id": "s1",
                 "Study_id": "study1", "Dicom_id": "b", "ViewPosition": "LATERAL", **positive},
            ]
            write_manifest(manifest, rows)

            studies = read_study_manifest(manifest, labeled=True)

            self.assertEqual(len(studies), 1)
            self.assertEqual(studies[0].study_id, "study1")
            self.assertEqual(studies[0].image_paths, ("train/s1/study1/a.jpg", "train/s1/study1/b.jpg"))
            self.assertEqual(studies[0].view_positions, ("AP", "LATERAL"))
            self.assertEqual(studies[0].labels, tuple(int(value) for value in [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))

    def test_rejects_inconsistent_labels_within_one_study(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "train.csv"
            first = {f"label_{index}": "0" for index in range(10)}
            second = dict(first)
            second["label_7"] = "1"
            rows = [
                {"image_path": "a.jpg", "Subject_id": "s1", "Study_id": "study1",
                 "Dicom_id": "a", "ViewPosition": "AP", **first},
                {"image_path": "b.jpg", "Subject_id": "s1", "Study_id": "study1",
                 "Dicom_id": "b", "ViewPosition": "PA", **second},
            ]
            write_manifest(manifest, rows)

            with self.assertRaisesRegex(ValueError, "inconsistent labels"):
                read_study_manifest(manifest, labeled=True)

    def test_positive_weights_count_unique_studies_and_cap_rare_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "train.csv"
            rows = []
            for study_index, label_one_positive in enumerate([True, False, False]):
                labels = {f"label_{index}": "0" for index in range(10)}
                labels["label_0"] = "1" if label_one_positive else "0"
                for image_index in range(2):
                    rows.append({
                        "image_path": f"train/s{study_index}/st{study_index}/{image_index}.jpg",
                        "Subject_id": f"s{study_index}", "Study_id": f"st{study_index}",
                        "Dicom_id": str(image_index), "ViewPosition": "AP", **labels,
                    })
            write_manifest(manifest, rows)

            studies = read_study_manifest(manifest, labeled=True)
            weights = compute_pos_weights(studies, cap=20.0)

            self.assertEqual(weights[0], 2.0)
            self.assertTrue(all(weight == 20.0 for weight in weights[1:]))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is installed on Windows for loss contract tests")
class LossContractsTest(unittest.TestCase):
    def test_uncertain_targets_are_ignored_for_main_and_expert_losses(self):
        from cxr_project.losses import compute_training_loss

        main_logits = torch.zeros((2, 10), requires_grad=True)
        expert_logits = torch.zeros((2, 3), requires_grad=True)
        targets = torch.zeros((2, 10))
        targets[:, 0] = -1
        pos_weights = torch.ones(10)

        loss, combined_logits = compute_training_loss(
            main_logits,
            expert_logits,
            targets,
            pos_weights,
            hard_label_indices=(0, 3, 7),
            expert_loss_weight=0.5,
        )
        loss.backward()

        torch.testing.assert_close(combined_logits[:, 1], main_logits.detach()[:, 1])
        torch.testing.assert_close(combined_logits[:, 0], torch.zeros(2))
        torch.testing.assert_close(main_logits.grad[:, 0], torch.zeros(2))
        torch.testing.assert_close(expert_logits.grad[:, 0], torch.zeros(2))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is installed on Windows for model contract tests")
class ModelContractsTest(unittest.TestCase):
    def test_model_emits_ten_main_and_three_specialist_logits_per_study(self):
        from torch import nn
        from cxr_project.model import ConvNeXtStudyClassifier

        model = ConvNeXtStudyClassifier(backbone=nn.Identity(), feature_dim=8, pretrained=False)
        image_features = torch.randn(5, 8)
        offsets = torch.tensor([0, 2, 5], dtype=torch.long)
        view_ids = torch.tensor([0, 2, 1, 4, 3], dtype=torch.long)

        main_logits, hard_logits = model.forward_from_features(image_features, offsets, view_ids)

        self.assertEqual(tuple(main_logits.shape), (2, 10))
        self.assertEqual(tuple(hard_logits.shape), (2, 3))
        self.assertTrue(torch.isfinite(main_logits).all())
        self.assertTrue(torch.isfinite(hard_logits).all())
