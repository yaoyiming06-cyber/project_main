import csv
import contextlib
import io
import json
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
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

    def test_threshold_difference_hidden_by_official_rounding_is_allowed(self):
        reference = [[0.4] * 10]
        candidate = [[0.4] * 10]
        reference[0][0] = 0.50004
        candidate[0][0] = 0.49996
        report = self.compare(reference, candidate)
        self.assertTrue(report["matched"])
        self.assertEqual(report["classification_mismatch_count"], 0)
        self.assertEqual(report["probability_decimal_places"], 4)

    def test_threshold_crossing_after_rounding_is_rejected(self):
        report = self.compare([[0.49994] * 10], [[0.50004] * 10])
        self.assertFalse(report["matched"])
        self.assertEqual(report["classification_mismatch_count"], 10)
        self.assertEqual(report["sample_mismatch_count"], 1)

    def test_per_class_study_ranking_reversal_is_rejected(self):
        reference = [[0.2] * 10, [0.2002] * 10]
        candidate = [[0.2002] * 10, [0.2] * 10]
        report = self.compare(reference, candidate)
        self.assertFalse(report["matched"])
        self.assertGreater(report["study_ranking_mismatch_count"], 0)
        self.assertEqual(report["sample_mismatch_count"], 2)
        self.assertEqual(report["mismatch_count"], 2)
        self.assertEqual(report["mismatched_study_ids"], ["study_0", "study_1"])

    def test_a_changed_tie_is_rejected(self):
        reference = [[0.2] * 10, [0.2] * 10]
        candidate = [[0.2] * 10, [0.2001] * 10]
        report = self.compare(reference, candidate)
        self.assertFalse(report["matched"])
        self.assertGreater(report["study_ranking_mismatch_count"], 0)

    def test_subdecimal_rank_reversals_and_ties_are_ignored(self):
        report = self.compare([[0.2] * 10, [0.20002] * 10],
                              [[0.20002] * 10, [0.2] * 10])
        self.assertTrue(report["matched"])
        self.assertEqual(report["sample_mismatch_count"], 0)

    def test_label_only_rank_change_marks_the_sample(self):
        reference = [[0.1 + i * .02 for i in range(10)]]
        reference[0][:2] = [.1000, .1002]
        candidate = [reference[0].copy()]
        candidate[0][:2] = [.1002, .1000]
        report = self.compare(reference, candidate)
        self.assertFalse(report["matched"])
        self.assertEqual(report["classification_mismatch_count"], 0)
        self.assertEqual(report["probability_tolerance_mismatch_count"], 0)
        self.assertEqual(report["sample_mismatch_count"], 1)
        self.assertTrue(report["first_mismatches"])

    def test_tolerance_is_applied_after_rounding_including_boundary(self):
        self.assertTrue(self.compare([[.20004] * 10], [[.20104] * 10])["matched"])
        self.assertFalse(self.compare([[.20004] * 10], [[.20106] * 10])["matched"])

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


class SubmissionProtocolTests(unittest.TestCase):
    def test_earlier_repeat_rank_failure_is_retained_and_counted_by_image(self):
        helper = ComparisonTests()
        self.addCleanup(helper.doCleanups)
        failed = helper.compare([[.2] * 10, [.2002] * 10], [[.2002] * 10, [.2] * 10])
        passed = helper.compare([[.2] * 10, [.2002] * 10], [[.2] * 10, [.2002] * 10])
        rows = [{"Study_id": "study_0"}, {"Study_id": "study_0"}, {"Study_id": "study_1"}]
        report = cost._summarize_comparisons([failed, passed], rows)
        self.assertFalse(report["matched"])
        self.assertEqual(report["sample_mismatch_count"], 2)
        self.assertEqual(report["image_mismatch_count"], 3)
        self.assertEqual(report["checked_repeats"], 2)

    def test_shared_writer_uses_four_decimal_half_even_rounding(self):
        from scripts.competition_inference import write_competition_submission
        values = np.asarray([[.12345, .12355, .49996, .50004, 0, 1, .2, .3, .4, .6]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            write_competition_submission(path, ["study_1"], values)
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual([r["Probability"] for r in rows[:6]],
                         ["0.1234", "0.1236", "0.5000", "0.5000", "0.0000", "1.0000"])

    def test_writer_does_not_hide_invalid_raw_values_by_rounding(self):
        from scripts.competition_inference import write_competition_submission
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            for value in [1.000001, -.000001, float("nan"), float("inf")]:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    write_competition_submission(path, ["study_1"], np.full((1, 10), value))
                self.assertFalse(path.exists())

    def test_export_script_matches_shared_writer(self):
        from scripts import export_competition_submission as exporter
        from scripts.competition_inference import write_competition_submission
        values = np.asarray([[.12345, .12355, .49996, .50004, 0, 1, .2, .3, .4, .6]])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template, predictions, output = [root / n for n in ("template.csv", "predictions.csv", "output.csv")]
            write_competition_submission(template, ["study_1"], values)
            with predictions.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Study_id", *[f"label_{i}" for i in range(10)]])
                writer.writerow(["study_1", *values[0]])
            with patch.object(sys, "argv", ["export", "--predictions", str(predictions),
                                           "--template", str(template), "--output", str(output)]):
                with contextlib.redirect_stdout(io.StringIO()):
                    exporter.main()
            self.assertEqual(template.read_bytes(), output.read_bytes())

    def test_official_pipeline_rejects_subset_dependent_patient_smoothing(self):
        from scripts.competition_inference import InferencePipeline
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            path.write_text(json.dumps({"version": 1, "patient_smoothing_alpha": .1,
                                        "models": [{}]}))
            with patch.object(InferencePipeline, "_load_model") as loader:
                with self.assertRaisesRegex(ValueError, "patient.*smooth|patient_smoothing"):
                    InferencePipeline(path, "cpu")
                loader.assert_not_called()

    def test_fusion_is_rounded_only_after_averaging(self):
        from types import SimpleNamespace
        import torch
        from scripts import competition_inference as inference
        pipeline = object.__new__(inference.InferencePipeline)
        pipeline.device = torch.device("cpu")
        pipeline.models = [SimpleNamespace(model=value, config={"amp": False}, name=str(value))
                           for value in [.12344, .12354]]
        pipeline._normalized_weights = np.asarray([.5, .5])
        pipeline.tta = "none"
        prepared = SimpleNamespace(datasets=[None, None], study_ids=["study_1"])
        with patch.object(inference, "DataLoader", return_value=[{"study_ids": ["study_1"]}]), \
             patch.object(inference, "predict_batch", side_effect=lambda model, *a, **kw:
                          torch.full((1, 10), model, dtype=torch.float64)):
            _, values = pipeline.predict(prepared)
        np.testing.assert_array_equal(values, np.full((1, 10), .1235))

    def test_predict_and_cost_clis_agree_on_complete_study_subset(self):
        """Exercise both CLI entrypoints, real image decoding, Study pooling and CSV comparison."""
        from types import SimpleNamespace
        import shutil
        import torch
        from PIL import Image
        from torchvision.transforms import ToTensor
        from scripts import predict
        from scripts.competition_inference import InferencePipeline

        class TinyModel(torch.nn.Module):
            hard_label_indices = ()
            image_chunk_size = 1

            def forward(self, images, offsets, view_ids):
                logits = torch.stack([images[start:end].mean() for start, end in
                                      zip(offsets[:-1], offsets[1:])]).unsqueeze(1)
                logits = logits + torch.linspace(-1, 1, 10).unsqueeze(0)
                return logits, logits[:, :0]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle.json"
            bundle.write_text(json.dumps({"version": 1, "patient_smoothing_alpha": 0,
                                          "models": [{}]}))

            def runtime(pipeline, spec, index):
                return SimpleNamespace(name="fixture", model=TinyModel().eval(),
                    config={"amp": False, "input_size": 16, "image_chunk_size": 1},
                    transform=ToTensor(), config_path=bundle, checkpoint_path=bundle,
                    config_sha256="fixture", checkpoint_sha256="fixture", weight_source="ema", fusion_weight=1.)

            full, measured, warmup = [root / n for n in ("full", "measured", "warmup")]
            for study, count in [(1, 2), (2, 1), (3, 1)]:
                for image in range(count):
                    p = full / "subject_1" / f"study_{study}" / f"{image}.png"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (16, 16), (study * 50 + image, 80, 100)).save(p)
            for study in [1, 3]:
                shutil.copytree(full / "subject_1" / f"study_{study}",
                                measured / "subject_1" / f"study_{study}")
            for study in range(5):
                p = warmup / "subject_2" / f"study_{100 + study}" / "0.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (16, 16), (30, 80, 100)).save(p)
            submission = root / "submission.csv"
            common = ["--bundle", str(bundle), "--device", "cpu"]
            with patch.object(InferencePipeline, "_load_model", runtime), \
                 contextlib.redirect_stdout(io.StringIO()):
                with patch.object(sys, "argv", ["predict", *common, "--data-dir", str(full),
                                               "--output", str(submission)]):
                    predict.main()
                with patch.object(sys, "argv", ["cost", *common, "--data-dir", str(measured),
                      "--warmup-dir", str(warmup), "--output-dir", str(root / "cost"),
                      "--submission", str(submission), "--allow-short-sample"]):
                    code = cost.main()
            report = json.loads((root / "cost/cost_report.json").read_text())
            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "diagnostic_only")
            self.assertEqual(report["consistency_status"], "passed")
            self.assertEqual(report["submission_comparison"]["image_mismatch_count"], 0)
            self.assertEqual(report["submission_comparison"]["image_sample_count"], 3)
            self.assertEqual(report["submission_comparison"]["sample_count"], 2)
            self.assertEqual(len(report["repeat_submission_comparisons"]), 3)
            records = list(csv.DictReader(io.StringIO(submission.read_text())))
            records[0]["Probability"] = "0.9999"
            with submission.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Study_id", "Predict_class", "Probability"])
                writer.writeheader()
                writer.writerows(records)
            with patch.object(InferencePipeline, "_load_model", runtime), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 patch.object(sys, "argv", ["cost", *common, "--data-dir", str(measured),
                      "--warmup-dir", str(warmup), "--output-dir", str(root / "cost_failed"),
                      "--submission", str(submission), "--allow-short-sample"]):
                code = cost.main()
            failed = json.loads((root / "cost_failed/cost_report.json").read_text())
            self.assertEqual(code, 1)
            self.assertEqual(failed["status"], "consistency_failed")
            self.assertGreater(failed["submission_comparison"]["image_mismatch_count"], 0)


if __name__ == "__main__":
    unittest.main()
