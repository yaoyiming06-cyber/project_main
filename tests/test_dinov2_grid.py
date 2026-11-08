import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
import io

import yaml

from cxr_project.dinov2_grid import (
    build_stage_configs,
    select_best_learning_rate,
    write_stage_configs,
)
from cxr_project.general_weights import DINOV2_MODEL, DINOV2_URL
from cxr_project import dinov2_weights
from scripts.dinov2_grid import execute_grid, main as grid_main


SEEDS = (42, 2027, 3407)
LEARNING_RATES = (2.5e-5, 5e-5, 1e-4, 2e-4, 4e-4)


class DINOv2GridTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.weights = self.root / "dinov2_vitb14_pretrain.pth"
        payload = b"synthetic official checkpoint"
        self.weights.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        self.checkpoint = {
            "model": DINOV2_MODEL,
            "url": DINOV2_URL,
            "bytes": len(payload),
            "sha256": digest,
            "path": str(self.weights.resolve()),
        }
        self.weights.with_suffix(self.weights.suffix + ".source.json").write_text(
            json.dumps(self.checkpoint), encoding="utf-8",
        )
        self.pinned_weights_patch = patch.multiple(
            dinov2_weights,
            WEIGHTS_BYTES=len(payload),
            WEIGHTS_SHA256=digest,
        )
        self.pinned_weights_patch.start()
        self.baseline = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "configs" / "head_baseline_l5.yaml")
            .read_text(encoding="utf-8")
        )
        self.baseline.update(
            project_root=str(self.root),
            model=DINOV2_MODEL,
            pretrained=True,
            pretrained_checkpoint=str(self.weights),
            pretrained_sha256=None,
            input_size=252,
            batch_size_studies=16,
            accum_steps=1,
            image_chunk_size=2,
            normalization="pretrained",
            image_root=str(self.root / "cxr_first"),
            train_manifest=str(self.root / "train.csv"),
            valid_manifest=str(self.root / "valid.csv"),
            test_manifest=str(self.root / "test.csv"),
        )

    def tearDown(self):
        self.pinned_weights_patch.stop()
        self.temp_dir.cleanup()

    def test_stage_one_generates_fifteen_lr_seed_configs_without_other_changes(self):
        configs = build_stage_configs(
            self.baseline,
            stage="stage1",
            project_root=self.root,
            output_root=self.root / "runs" / "stage1",
            checkpoint_path=self.weights,
        )
        self.assertEqual(len(configs), 15)
        self.assertEqual(len({c["output_dir"] for c in configs.values()}), 15)
        self.assertEqual({c["lr"] for c in configs.values()}, set(LEARNING_RATES))
        self.assertEqual({c["seed"] for c in configs.values()}, set(SEEDS))
        self.assertEqual({c["input_size"] for c in configs.values()}, {252})
        self.assertTrue(all(c["pretrained_sha256"] == self.checkpoint["sha256"] for c in configs.values()))
        for config in configs.values():
            self.assertEqual(config["epochs"], 15)
            self.assertEqual(config["warmup_epochs"], 5)
            self.assertEqual(config["min_lr"], 1e-6)
            self.assertTrue(config["ema"])
            self.assertEqual(config["ema_decay"], 0.9998)
            self.assertEqual(config["validation_weights"], "ema")
            self.assertEqual(config["batch_size_studies"], 16)
            self.assertEqual(config["accum_steps"], 1)
            self.assertEqual(config["image_chunk_size"], 2)
            self.assertEqual(config["augmentation"], "G1")
            self.assertEqual(config["loss"], "focal")
            self.assertEqual(config["focal_gamma"], 2.0)
            self.assertEqual(config["pos_weight_mode"], "ratio")
            self.assertEqual(config["max_pos_weight"], 20.0)
            self.assertFalse(config["use_view_embedding"])
            self.assertIsNone(config["view_filter"])
            self.assertIsNone(config["clip_grad"])
            self.assertEqual(config["early_stopping_patience"], 0)
            self.assertEqual(config["normalization"], "pretrained")

    def test_stage_two_generates_six_resolutions_for_selected_lr(self):
        configs = build_stage_configs(
            self.baseline,
            stage="stage2",
            project_root=self.root,
            output_root=self.root / "runs" / "stage2",
            checkpoint_path=self.weights,
            best_lr=1e-4,
        )
        self.assertEqual(len(configs), 6)
        self.assertEqual({c["input_size"] for c in configs.values()}, {224, 280})
        self.assertEqual({c["lr"] for c in configs.values()}, {1e-4})
        self.assertEqual({c["seed"] for c in configs.values()}, set(SEEDS))
        self.assertEqual(len({c["output_dir"] for c in configs.values()}), 6)

    def test_stage_two_rejects_learning_rate_outside_stage_one(self):
        with self.assertRaisesRegex(ValueError, "one of the stage-1 learning rates"):
            build_stage_configs(
                self.baseline,
                stage="stage2",
                project_root=self.root,
                output_root=self.root / "runs",
                checkpoint_path=self.weights,
                best_lr=3e-4,
            )

    def test_writer_validates_configs_and_records_non_execution_manifest(self):
        configs = build_stage_configs(
            self.baseline,
            stage="stage1",
            project_root=self.root,
            output_root=self.root / "runs" / "written",
            checkpoint_path=self.weights,
        )
        baseline_path = self.root / "baseline.yaml"
        baseline_path.write_text(yaml.safe_dump(self.baseline), encoding="utf-8")
        directory = self.root / "generated"
        manifest = write_stage_configs(
            configs,
            stage="stage1",
            config_dir=directory,
            baseline_path=baseline_path,
            checkpoint_path=self.weights,
            python_executable="python-test",
        )
        self.assertEqual(len(manifest["jobs"]), 15)
        self.assertFalse(manifest["execution_requested"])
        self.assertEqual(manifest["jobs"][0]["command"][0], "python-test")
        self.assertIn("export-fp16", manifest["jobs"][0]["export_fp16_command"])
        self.assertIn("checkpoint-best-ema-fp16.pth", " ".join(manifest["jobs"][0]["export_fp16_command"]))
        self.assertTrue((directory / "grid_manifest.json").is_file())
        self.assertTrue(all((directory / f"{name}.yaml").is_file() for name in configs))

    def test_writer_refuses_nonempty_config_directory(self):
        directory = self.root / "occupied"
        directory.mkdir()
        (directory / "keep.txt").write_text("existing", encoding="utf-8")
        configs = build_stage_configs(
            self.baseline,
            stage="stage1",
            project_root=self.root,
            output_root=self.root / "runs" / "occupied",
            checkpoint_path=self.weights,
        )
        baseline_path = self.root / "baseline.yaml"
        baseline_path.write_text(yaml.safe_dump(self.baseline), encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "not empty"):
            write_stage_configs(
                configs,
                stage="stage1",
                config_dir=directory,
                baseline_path=baseline_path,
                checkpoint_path=self.weights,
            )

    def test_cli_generation_is_config_only_and_run_exports_after_each_job(self):
        baseline_path = self.root / "baseline.yaml"
        baseline_path.write_text(yaml.safe_dump(self.baseline), encoding="utf-8")
        config_dir = self.root / "cli_stage1"
        output_root = self.root / "cli_runs"
        with patch("scripts.dinov2_grid.subprocess.run") as run_jobs:
            with redirect_stdout(io.StringIO()):
                generated = grid_main([
                    "stage1",
                    "--baseline", str(baseline_path),
                    "--weights", str(self.weights),
                    "--config-dir", str(config_dir),
                    "--output-root", str(output_root),
                    "--device", "cuda",
                ])
            run_jobs.assert_not_called()
            self.assertEqual(len(generated["jobs"]), 15)
            executed = execute_grid(config_dir)
        self.assertEqual(run_jobs.call_count, 30)
        self.assertIn("cxr_project.train", run_jobs.call_args_list[0].args[0])
        self.assertIn("export-fp16", run_jobs.call_args_list[1].args[0])
        self.assertEqual(executed["execution_status"], "completed")

    def test_best_lr_uses_three_completed_ema_runs_per_rate(self):
        config_dir = self.root / "stage1"
        config_dir.mkdir()
        jobs = []
        for lr_index, lr in enumerate(LEARNING_RATES):
            for seed_index, seed in enumerate(SEEDS):
                config_path = config_dir / f"lr{lr_index}_seed{seed}.yaml"
                config_path.write_text(yaml.safe_dump({
                    "model": DINOV2_MODEL,
                    "input_size": 252,
                    "lr": lr,
                    "seed": seed,
                }), encoding="utf-8")
                run_dir = self.root / "runs" / f"lr{lr_index}_seed{seed}"
                run_dir.mkdir(parents=True)
                score = 0.70 + 0.02 * (lr_index == 1) + seed_index * 0.001
                (run_dir / "run_manifest.json").write_text(json.dumps({
                    "status": "completed",
                    "selected_weights": "ema",
                    "best_macro_auc": score,
                    "best_epoch": 7,
                }), encoding="utf-8")
                jobs.append({
                    "learning_rate": lr,
                    "seed": seed,
                    "output_dir": str(run_dir),
                    "config": str(config_path),
                    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                })
        (config_dir / "grid_manifest.json").write_text(json.dumps({
            "stage": "stage1",
            "jobs": jobs,
        }), encoding="utf-8")
        result = select_best_learning_rate(config_dir)
        self.assertEqual(result["best_lr"], 5e-5)
        self.assertEqual(result["runs_per_learning_rate"], 3)
        self.assertEqual(len(result["by_learning_rate"]), 5)
        self.assertTrue(all(row["sample_std"] >= 0 for row in result["by_learning_rate"]))

    def test_best_lr_rejects_incomplete_stage_one(self):
        config_dir = self.root / "incomplete"
        config_dir.mkdir()
        (config_dir / "grid_manifest.json").write_text(json.dumps({
            "stage": "stage1",
            "jobs": [],
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly 15"):
            select_best_learning_rate(config_dir)


if __name__ == "__main__":
    unittest.main()
