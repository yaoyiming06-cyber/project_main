import csv
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import cost


class ComparisonTests(unittest.TestCase):
    def write_submission(self, rows):
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "submission.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Study_id", "Predict_class", "Probability"])
            writer.writerows(rows)
        self.addCleanup(temporary.cleanup)
        return path

    def compare(self, reference, candidate):
        rows = []
        for index, row in enumerate(reference):
            rows.extend(
                [f"study_{index}", label, value]
                for label, value in enumerate(row, 1)
            )
        path = self.write_submission(rows)
        return cost._compare_submission(
            path,
            [f"study_{index}" for index in range(len(candidate))],
            np.asarray(candidate, dtype=np.float64),
        )

    def test_probability_error_within_tolerance_and_order_is_allowed(self):
        reference = [[0.2 + index * 0.01 for index in range(10)]]
        candidate = [[value + 0.0002 for value in reference[0]]]
        report = self.compare(reference, candidate)
        self.assertTrue(report["matched"])
        self.assertLessEqual(report["max_absolute_error"], 0.001)

    def test_threshold_crossing_is_rejected_before_rounding(self):
        reference = [[0.4] * 10]
        candidate = [[0.4] * 10]
        reference[0][0] = 0.50004
        candidate[0][0] = 0.49996
        report = self.compare(reference, candidate)
        self.assertFalse(report["matched"])
        self.assertEqual(report["classification_mismatch_count"], 1)

    def test_per_class_study_ranking_reversal_is_rejected(self):
        reference = [[0.2] * 10, [0.20002] * 10]
        candidate = [[0.20002] * 10, [0.2] * 10]
        report = self.compare(reference, candidate)
        self.assertFalse(report["matched"])
        self.assertGreater(report["study_ranking_mismatch_count"], 0)

    def test_a_changed_tie_is_rejected(self):
        reference = [[0.2] * 10, [0.2] * 10]
        candidate = [[0.2] * 10, [0.20001] * 10]
        report = self.compare(reference, candidate)
        self.assertFalse(report["matched"])
        self.assertGreater(report["study_ranking_mismatch_count"], 0)

    def test_probability_out_of_range_is_not_hidden_by_rounding(self):
        with self.assertRaises(ValueError):
            self.compare([[1.0] * 10], [[1.000001] * 10])

    def test_probability_error_above_tolerance_is_rejected(self):
        report = self.compare([[0.2] * 10], [[0.2011] * 10])
        self.assertFalse(report["matched"])
        self.assertGreater(report["probability_tolerance_mismatch_count"], 0)

    def test_duplicate_class_row_is_rejected(self):
        path = self.write_submission(
            [["study_1", 1, 0.2], ["study_1", 1, 0.3]]
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            cost._read_submission(path)

    def test_missing_class_row_is_rejected(self):
        path = self.write_submission([["study_1", 1, 0.2]])
        with self.assertRaisesRegex(ValueError, "ten|10"):
            cost._read_submission(path)

    def test_invalid_class_index_is_rejected(self):
        path = self.write_submission([["study_1", 11, 0.2]])
        with self.assertRaisesRegex(ValueError, "invalid"):
            cost._read_submission(path)


class ImageTreeTests(unittest.TestCase):
    def make_tree(self, root, subject, study, filename="image.jpg"):
        path = Path(root) / subject / study / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image fixture")
        return path

    def test_scanner_extracts_only_ids_from_paths_and_sets_unknown_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pictures"
            self.make_tree(root, "subject_7", "study_11", "b.jpg")
            self.make_tree(root, "subject_7", "study_11", "a.jpg")
            rows = cost._scan_image_tree(root)

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["Subject_id"] for row in rows}, {"subject_7"})
        self.assertEqual({row["Study_id"] for row in rows}, {"study_11"})
        self.assertEqual({row["ViewPosition"] for row in rows}, {"UNKNOWN"})
        self.assertEqual(
            [row["image_path"] for row in rows],
            ["subject_7/study_11/a.jpg", "subject_7/study_11/b.jpg"],
        )

    def test_warmup_and_measured_studies_must_be_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            measured = root / "pictures_100"
            warmup = root / "pictures_5"
            self.make_tree(measured, "subject_7", "study_11")
            self.make_tree(warmup, "subject_7", "study_11")
            with self.assertRaisesRegex(ValueError, "overlap|disjoint"):
                cost._validate_warmup_separation(
                    cost._scan_image_tree(measured), cost._scan_image_tree(warmup)
                )

    def test_warmup_selection_uses_exact_image_count_and_spreads_across_studies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pictures_5"
            for study_index in range(5):
                for image_index in range(2):
                    self.make_tree(
                        root,
                        "subject_1",
                        f"study_{study_index}",
                        f"image_{image_index}.jpg",
                    )
            selected = cost._select_warmup_rows(cost._scan_image_tree(root), 5)

        self.assertEqual(len(selected), 5)
        self.assertEqual(len({row["Study_id"] for row in selected}), 5)


class TimingTests(unittest.TestCase):
    def test_timer_includes_final_csv_serialization(self):
        class FakePipeline:
            device = __import__("torch").device("cpu")

            def predict(self, prepared, output_path):
                time.sleep(0.02)
                output_path.write_text("Study_id,Predict_class,Probability\n", encoding="utf-8")
                return ["study_1"], np.zeros((1, 10), dtype=np.float64)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "submission.csv"
            _, _, elapsed = cost._timed_prediction_run(FakePipeline(), object(), output)
            self.assertTrue(output.is_file())
            self.assertGreaterEqual(elapsed, 0.018)


if __name__ == "__main__":
    unittest.main()
