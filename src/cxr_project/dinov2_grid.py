"""Build and select the staged DINOv2 ViT-B/14 learning-rate grid."""

from __future__ import annotations

import json
import hashlib
import math
import statistics
import sys
from pathlib import Path

from .dinov2_weights import validate_checkpoint
from .general_weights import DINOV2_MODEL


SEEDS = (42, 2027, 3407)
LEARNING_RATES = (5e-5, 1e-4, 2e-4)
RESOLUTIONS = (224, 252, 280)


def build_stage_configs(
    baseline,
    *,
    stage,
    project_root,
    output_root,
    checkpoint_path,
    best_lr=None,
):
    """Return the stage's resolved configs without writing or starting training."""
    if stage not in {"stage1", "stage2"}:
        raise ValueError("stage must be stage1 or stage2")
    if baseline.get("model") != DINOV2_MODEL:
        raise ValueError(f"baseline model must be {DINOV2_MODEL}")
    if stage == "stage2":
        try:
            best_lr = float(best_lr)
        except (TypeError, ValueError) as error:
            raise ValueError("stage2 requires a selected stage-1 learning rate") from error
        if best_lr not in LEARNING_RATES:
            raise ValueError("best_lr must be one of the stage-1 learning rates")

    root = Path(project_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    source_sidecar = checkpoint.with_suffix(checkpoint.suffix + ".source.json")
    try:
        source = json.loads(source_sidecar.read_text(encoding="utf-8"))
        digest = source.get("sha256")
    except (OSError, json.JSONDecodeError, AttributeError) as error:
        raise ValueError(f"DINOv2 checkpoint provenance is unavailable: {source_sidecar}") from error
    provenance = validate_checkpoint(checkpoint, digest)

    path_keys = ("image_root", "train_manifest", "valid_manifest", "test_manifest")
    base = {key: value for key, value in baseline.items() if not key.startswith("_")}
    for key in path_keys:
        value = Path(base[key]).expanduser()
        base[key] = str(value.resolve() if value.is_absolute() else (root / value).resolve())
    base["project_root"] = str(root)
    base["pretrained_checkpoint"] = str(checkpoint)
    base["pretrained_sha256"] = provenance["sha256"]
    base["model"] = DINOV2_MODEL
    base["pretrained"] = True
    base["normalization"] = "pretrained"

    factors = (
        [(lr, 252, seed) for lr in LEARNING_RATES for seed in SEEDS]
        if stage == "stage1"
        else [(best_lr, size, seed) for size in (224, 280) for seed in SEEDS]
    )
    configs = {}
    for lr, size, seed in factors:
        tag = f"lr{lr:.0e}".replace("-0", "-")
        name = f"DINOv2_{tag}_seed{seed}" if stage == "stage1" else f"DINOv2_res{size}_{tag}_seed{seed}"
        config = dict(base)
        config.update(
            run_name=name,
            seed=seed,
            input_size=size,
            lr=lr,
            output_dir=str(output / name),
        )
        configs[name] = config
    return configs


def select_best_learning_rate(stage1_config_dir):
    """Select LR by mean best EMA validation macro AUC across all three seeds."""
    directory = Path(stage1_config_dir).expanduser().resolve()
    manifest_path = directory / "grid_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read stage-1 grid manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("stage-1 grid manifest must contain a JSON object")
    jobs = manifest.get("jobs")
    if manifest.get("stage") != "stage1" or not isinstance(jobs, list) or len(jobs) != 9:
        raise ValueError("stage-1 manifest must contain exactly nine LR/seed runs")

    records = {}
    for job in jobs:
        config_path = Path(job.get("config", ""))
        if not config_path.is_file():
            raise ValueError(f"stage-1 config is missing: {config_path}")
        config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
        if config_digest != job.get("config_sha256"):
            raise ValueError(f"stage-1 config changed after generation: {config_path}")
        import yaml
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        lr = float(job["learning_rate"])
        seed = int(job["seed"])
        if lr not in LEARNING_RATES or seed not in SEEDS:
            raise ValueError("stage-1 manifest contains an unexpected learning rate or seed")
        if (config.get("model") != DINOV2_MODEL or config.get("seed") != seed
                or float(config.get("lr")) != lr or config.get("input_size") != 252):
            raise ValueError(f"stage-1 config does not match its LR/seed metadata: {config_path}")
        key = (lr, seed)
        if key in records:
            raise ValueError("stage-1 manifest contains duplicate LR/seed runs")
        run_manifest_path = Path(job["output_dir"]) / "run_manifest.json"
        try:
            run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"missing or invalid run manifest: {run_manifest_path}") from error
        if run.get("status") != "completed" or run.get("selected_weights") != "ema":
            raise ValueError(f"run is incomplete or did not select EMA weights: {run_manifest_path}")
        score = run.get("best_macro_auc")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"invalid best EMA validation AUC: {run_manifest_path}")
        records[key] = {"seed": seed, "macro_auc": float(score), "best_epoch": run.get("best_epoch")}

    expected = {(lr, seed) for lr in LEARNING_RATES for seed in SEEDS}
    if set(records) != expected:
        raise ValueError("stage-1 results must include each of three learning rates and all three seeds")

    summaries = []
    for lr in LEARNING_RATES:
        rows = [records[(lr, seed)] for seed in SEEDS]
        scores = [row["macro_auc"] for row in rows]
        summaries.append({
            "learning_rate": lr,
            "seed_results": rows,
            "mean": statistics.mean(scores),
            "sample_std": statistics.stdev(scores),
        })
    winner = min(summaries, key=lambda row: (-row["mean"], row["sample_std"], row["learning_rate"]))
    return {
        "selection_metric": "mean of three best EMA validation macro AUCs",
        "tie_break": "lower sample standard deviation, then lower learning rate",
        "runs_per_learning_rate": 3,
        "best_lr": winner["learning_rate"],
        "by_learning_rate": summaries,
    }


def write_stage_configs(
    configs,
    *,
    stage,
    config_dir,
    baseline_path,
    checkpoint_path,
    selection=None,
    device="cuda",
    python_executable=None,
    execution_requested=False,
):
    """Write validated YAMLs and an immutable command/provenance manifest."""
    import yaml

    from .config import load_config

    directory = Path(config_dir).expanduser().resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"config directory is not empty: {directory}")
    for config in configs.values():
        output = Path(config["output_dir"])
        if output.exists():
            raise FileExistsError(f"run output directory already exists: {output}")

    directory.mkdir(parents=True, exist_ok=True)
    jobs = []
    baseline_path = Path(baseline_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    for name, config in configs.items():
        config_path = directory / f"{name}.yaml"
        payload = {key: value for key, value in config.items() if not key.startswith("_")}
        config_path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8",
        )
        validated = load_config(config_path)
        command = [
            str(python_executable or sys.executable), "-m", "cxr_project.train", "--config", str(config_path),
            "--device", device,
        ]
        run_dir = Path(validated["output_dir"])
        export_command = [
            str(python_executable or sys.executable),
            str(Path(validated["project_root"]) / "scripts" / "evaluate_grid.py"),
            "export-fp16",
            "--checkpoint", str(run_dir / "checkpoint-best-ema.pth"),
            "--output", str(run_dir / "checkpoint-best-ema-fp16.pth"),
            "--weights", "ema",
        ]
        jobs.append({
            "name": name,
            "config": str(config_path),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "output_dir": validated["output_dir"],
            "seed": validated["seed"],
            "learning_rate": validated["lr"],
            "input_size": validated["input_size"],
            "command": command,
            "export_fp16_command": export_command,
        })

    source_sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".source.json")
    checkpoint_info = json.loads(source_sidecar.read_text(encoding="utf-8"))
    baseline_digest = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    manifest = {
        "stage": stage,
        "baseline": str(baseline_path),
        "baseline_sha256": baseline_digest,
        "checkpoint": checkpoint_info,
        "selection": selection,
        "execution_requested": bool(execution_requested),
        "jobs": jobs,
    }
    (directory / "grid_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return manifest
