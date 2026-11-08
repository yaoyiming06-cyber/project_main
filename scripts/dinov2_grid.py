"""Generate or run the staged DINOv2 ViT-B/14 LR and resolution grid."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from cxr_project.config import load_config, resolve_path
from cxr_project.dinov2_grid import (
    build_stage_configs,
    select_best_learning_rate,
    write_stage_configs,
)
from cxr_project.dinov2_weights import validate_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = PROJECT_ROOT / "configs" / "dinov2_vitb14_baseline.yaml"


def add_grid_arguments(parser, *, stage2=False):
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--weights", type=Path, help="Defaults to the checkpoint path in the baseline")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu", "auto"), default="cuda")
    if stage2:
        parser.add_argument("--stage1-config-dir", type=Path, required=True)


def execute_grid(config_dir):
    directory = Path(config_dir).expanduser().resolve()
    manifest_path = directory / "grid_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = manifest.get("jobs") if isinstance(manifest, dict) else None
    expected_runs = {"stage1": 15, "stage2": 6}.get(manifest.get("stage")) if isinstance(manifest, dict) else None
    if expected_runs is None or not isinstance(jobs, list) or len(jobs) != expected_runs:
        raise ValueError("grid manifest has an invalid stage or run count")
    for job in jobs:
        config_path = Path(job["config"])
        if not config_path.is_file():
            raise FileNotFoundError(f"generated config is missing: {config_path}")
        if hashlib.sha256(config_path.read_bytes()).hexdigest() != job["config_sha256"]:
            raise ValueError(f"generated config changed after manifest creation: {config_path}")
        if Path(job["output_dir"]).exists():
            raise FileExistsError(f"run output already exists: {job['output_dir']}")
        if not isinstance(job.get("command"), list) or not isinstance(job.get("export_fp16_command"), list):
            raise ValueError(f"training/export commands are missing for {job.get('name')}")
    baseline_config = load_config(jobs[0]["config"])
    project_root = baseline_config["_project_root"]
    manifest.update(execution_requested=True, execution_status="running")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        for job in jobs:
            subprocess.run(job["command"], cwd=project_root, check=True)
            subprocess.run(job["export_fp16_command"], cwd=project_root, check=True)
            manifest["last_completed_run"] = job["name"]
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    except subprocess.CalledProcessError as error:
        manifest.update(execution_status="failed", failed_returncode=error.returncode)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    manifest["execution_status"] = "completed"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    stage1 = commands.add_parser("stage1", help="Generate LR selection configs at 252px")
    add_grid_arguments(stage1)
    stage2 = commands.add_parser("stage2", help="Generate resolution configs from completed stage 1")
    add_grid_arguments(stage2, stage2=True)
    run = commands.add_parser("run", help="Execute a previously generated stage sequentially and export FP16 EMA")
    run.add_argument("--config-dir", type=Path, required=True)
    select = commands.add_parser("select-lr", help="Summarize completed stage-1 EMA validation scores")
    select.add_argument("--stage1-config-dir", type=Path, required=True)
    select.add_argument("--report", type=Path)

    args = parser.parse_args(argv)
    if args.command == "run":
        result = execute_grid(args.config_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    if args.command == "select-lr":
        result = select_best_learning_rate(args.stage1_config_dir)
        if args.report:
            report = args.report.expanduser().resolve()
            if report.exists():
                raise FileExistsError(report)
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    baseline_path = args.baseline.expanduser().resolve()
    baseline = load_config(baseline_path)
    root = args.project_root.expanduser().resolve() if args.project_root else baseline["_project_root"]
    checkpoint = args.weights.expanduser().resolve() if args.weights else resolve_path(baseline, "pretrained_checkpoint")
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".source.json")
    source = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance = validate_checkpoint(checkpoint, source.get("sha256"))

    selection = None
    best_lr = None
    if args.command == "stage2":
        selection = select_best_learning_rate(args.stage1_config_dir)
        stage1_manifest = json.loads(
            (args.stage1_config_dir.expanduser().resolve() / "grid_manifest.json")
            .read_text(encoding="utf-8")
        )
        baseline_digest = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        if stage1_manifest.get("baseline_sha256") != baseline_digest:
            raise ValueError("stage-1 and stage-2 must use the same baseline config")
        if stage1_manifest.get("checkpoint", {}).get("sha256") != provenance["sha256"]:
            raise ValueError("stage-1 and stage-2 must use the same verified DINOv2 checkpoint")
        best_lr = selection["best_lr"]
    configs = build_stage_configs(
        baseline,
        stage=args.command,
        project_root=root,
        output_root=args.output_root,
        checkpoint_path=checkpoint,
        best_lr=best_lr,
    )
    manifest = write_stage_configs(
        configs,
        stage=args.command,
        config_dir=args.config_dir,
        baseline_path=baseline_path,
        checkpoint_path=checkpoint,
        selection=selection,
        device=args.device,
        python_executable=sys.executable,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    main()
