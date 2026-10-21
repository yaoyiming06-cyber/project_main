"""Full Windows/CUDA training entry point for the G1 seed42 experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config, resolve_path
from .dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    StudyDataset,
    build_transform,
    collate_studies,
    compute_dataset_normalization,
)
from .labels import LABEL_NAMES
from .losses import combine_main_and_expert_logits, compute_training_loss
from .manifest import resolve_pos_weights
from .metrics import compute_macro_auc
from .model import MODEL_NAME, ConvNeXtStudyClassifier
from .provenance import (
    dataset_normalization_source,
    git_working_tree_state,
    validate_dataset_normalization_cache,
)
from .schedule import scheduler_step_counts


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optimizer_layer_id(name: str):
    """Map timm ConvNeXt stem and four stages to layer-decay levels."""
    if not name.startswith("backbone."):
        return 5
    parts = name.split(".")
    if len(parts) >= 2 and parts[1] == "stem":
        return 0
    if len(parts) >= 3 and parts[1] == "stages" and parts[2].isdigit():
        return int(parts[2]) + 1
    return 5


def build_optimizer(model, lr, weight_decay, layer_decay):
    """AdamW groups with stage-wise ConvNeXt layer decay."""
    groups = {}
    max_layer = 5
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        layer_id = _optimizer_layer_id(name)
        no_decay = parameter.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower()
        key = (layer_id, no_decay)
        if key not in groups:
            scale = float(layer_decay) ** (max_layer - layer_id)
            groups[key] = {
                "params": [],
                "lr": float(lr) * scale,
                "lr_scale": scale,
                "weight_decay": 0.0 if no_decay else float(weight_decay),
            }
        groups[key]["params"].append(parameter)
    if not groups:
        raise ValueError("model has no trainable parameters")
    return torch.optim.AdamW(list(groups.values()), lr=float(lr), weight_decay=float(weight_decay))


def learning_rate_at_step(step, total_steps, warmup_steps, base_lr, min_lr):
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if warmup_steps > 0 and step < warmup_steps:
        return float(base_lr) * (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr) + (float(base_lr) - float(min_lr)) * cosine


def _set_group_lrs(optimizer, base_lr):
    for group in optimizer.param_groups:
        group["lr"] = base_lr * group["lr_scale"]


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def evaluate(model, loader, device, use_amp):
    model.eval()
    all_targets = []
    all_probabilities = []
    all_study_ids = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            offsets = batch["offsets"]
            view_ids = batch["view_ids"].to(device, non_blocking=True)
            with _autocast(device, use_amp):
                main_logits, expert_logits = model(images, offsets, view_ids)
            combined = combine_main_and_expert_logits(
                main_logits.float(), expert_logits.float(), model.hard_label_indices,
            )
            probabilities = combined.sigmoid().cpu().numpy()
            all_probabilities.append(probabilities)
            all_targets.append(batch["targets"].numpy())
            all_study_ids.extend(batch["study_ids"])
    targets = np.concatenate(all_targets, axis=0)
    probabilities = np.concatenate(all_probabilities, axis=0)
    metrics = compute_macro_auc(targets, probabilities)
    return metrics, all_study_ids, targets, probabilities


def _write_validation_predictions(path, study_ids, targets, probabilities):
    columns = ["Study_id"]
    columns += [f"target_{index}" for index in range(10)]
    columns += [f"probability_{index}" for index in range(10)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for study_id, target, probability in zip(study_ids, targets, probabilities):
            writer.writerow([study_id, *target.tolist(), *probability.tolist()])


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _normalization_stats(config, train_manifest, image_root, run_dir):
    if str(config["normalization"]).lower() != "dataset":
        return None
    stats_path = run_dir / "dataset_normalization.json"
    source = dataset_normalization_source(
        train_manifest,
        image_root,
        input_size=int(config["input_size"]),
    )
    if stats_path.is_file():
        with stats_path.open("r", encoding="utf-8") as handle:
            stats = json.load(handle)
        return validate_dataset_normalization_cache(stats, source, stats_path)
    stats = compute_dataset_normalization(
        train_manifest,
        image_root,
        input_size=int(config["input_size"]),
    )
    stats["source"] = source
    _write_json(stats_path, stats)
    return stats


def _plain_config(config):
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _train_epoch(model, loader, dataset_size, optimizer, scaler, device, config,
                 epoch, global_step, total_steps, warmup_steps, pos_weights):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    seen_studies = 0
    steps_per_epoch = len(loader)

    for step, batch in enumerate(loader):
        if step % config["accum_steps"] == 0:
            window_start = step
            window_end = min(steps_per_epoch, window_start + config["accum_steps"])
            window_start_sample = window_start * config["batch_size_studies"]
            window_studies = min(
                (window_end - window_start) * config["batch_size_studies"],
                dataset_size - window_start_sample,
            )
            base_lr = learning_rate_at_step(
                global_step, total_steps, warmup_steps, config["lr"], config["min_lr"],
            )
            _set_group_lrs(optimizer, base_lr)

        images = batch["images"].to(device, non_blocking=True)
        offsets = batch["offsets"]
        view_ids = batch["view_ids"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        with _autocast(device, config["amp"]):
            main_logits, expert_logits = model(images, offsets, view_ids)
        loss, _ = compute_training_loss(
            main_logits.float(), expert_logits.float(), targets,
            pos_weights,
            config["hard_label_indices"],
            config["expert_loss_weight"],
            config["loss"],
            config["focal_gamma"],
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}, batch {step + 1}")
        current_studies = int(targets.shape[0])
        scaler.scale(loss * current_studies / window_studies).backward()
        running_loss += loss.detach().item() * current_studies
        seen_studies += current_studies

        should_step = (step + 1) % config["accum_steps"] == 0 or step + 1 == steps_per_epoch
        if should_step:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            clip_grad = config.get("clip_grad")
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        if (step + 1) % 100 == 0 or step + 1 == steps_per_epoch:
            print(
                f"epoch {epoch + 1}/{config['epochs']} batch {step + 1}/{steps_per_epoch} "
                f"loss={loss.detach().item():.5f} lr={optimizer.param_groups[-1]['lr']:.7g}",
                flush=True,
            )
    return running_loss / max(1, seen_studies), global_step


def train(config_path, data_root=None, output_dir=None, resume=None, device_name="cuda"):
    config = load_config(config_path)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA GPU; run scripts/check_environment.py")
    if device_name not in ("cuda", "cpu"):
        raise ValueError("device must be 'cuda' or 'cpu'")
    device = torch.device(device_name)
    set_seed(int(config["seed"]))

    train_manifest = resolve_path(config, "train_manifest")
    valid_manifest = resolve_path(config, "valid_manifest")
    image_root = Path(data_root).resolve() if data_root else resolve_path(config, "image_root")
    run_dir = Path(output_dir).resolve() if output_dir else resolve_path(config, "output_dir")
    if run_dir.exists() and resume is None:
        raise FileExistsError(f"run directory already exists; choose a new --output-dir or pass --resume: {run_dir}")
    if config["model"] != MODEL_NAME:
        raise ValueError(f"unsupported model in config: {config['model']!r}; expected {MODEL_NAME!r}")
    run_dir.mkdir(parents=True, exist_ok=True)

    normalization_stats = _normalization_stats(config, train_manifest, image_root, run_dir)
    train_transform = build_transform(
        True, int(config["input_size"]), config=config, normalization=normalization_stats,
    )
    valid_transform = build_transform(
        False, int(config["input_size"]), config=config, normalization=normalization_stats,
    )
    train_data = StudyDataset(train_manifest, image_root, train_transform, labeled=True)
    valid_data = StudyDataset(valid_manifest, image_root, valid_transform, labeled=True)
    pos_weights_list = resolve_pos_weights(
        train_data.studies,
        mode=config["pos_weight_mode"],
        cap=float(config["max_pos_weight"]),
        manual=config["manual_pos_weights"],
    )
    pos_weights = torch.tensor(pos_weights_list, dtype=torch.float32, device=device)

    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    loader_options = {
        "num_workers": int(config["num_workers"]),
        "pin_memory": bool(config["pin_memory"] and device.type == "cuda"),
        "collate_fn": collate_studies,
        "persistent_workers": int(config["num_workers"]) > 0,
    }
    if int(config["num_workers"]) > 0:
        loader_options["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_data,
        batch_size=int(config["batch_size_studies"]),
        shuffle=True,
        drop_last=False,
        generator=generator,
        **loader_options,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=int(config["batch_size_studies"]),
        shuffle=False,
        drop_last=False,
        **loader_options,
    )

    weight_path = resolve_path(config, "pretrained_checkpoint")
    model = ConvNeXtStudyClassifier(
        pretrained=bool(config["pretrained"]) and resume is None,
        pretrained_checkpoint=weight_path if weight_path.is_file() else None,
        drop_path=float(config["drop_path"]),
        hard_label_indices=config["hard_label_indices"],
        gradient_checkpointing=bool(config["gradient_checkpointing"]),
        image_chunk_size=int(config["image_chunk_size"]),
    ).to(device)
    optimizer = build_optimizer(
        model,
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        layer_decay=float(config["layer_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(config["amp"]))

    (run_dir / "config_resolved.yaml").write_text(
        Path(config_path).read_text(encoding="utf-8"), encoding="utf-8",
    )
    start_epoch = 0
    global_step = 0
    best_auc = float("-inf")
    best_epoch = None
    if resume:
        resume_path = Path(resume).resolve()
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_auc = float(checkpoint.get("best_auc", float("-inf")))
        best_epoch = checkpoint.get("best_epoch")
        if resume_path.parent != run_dir:
            previous_best = resume_path.parent / "checkpoint-best.pth"
            previous_predictions = resume_path.parent / "valid_best_predictions.csv"
            if previous_best.is_file() and not (run_dir / "checkpoint-best.pth").exists():
                import shutil
                shutil.copy2(previous_best, run_dir / "checkpoint-best.pth")
            if previous_predictions.is_file() and not (run_dir / "valid_best_predictions.csv").exists():
                import shutil
                shutil.copy2(previous_predictions, run_dir / "valid_best_predictions.csv")

    steps_per_epoch = len(train_loader)
    total_steps, warmup_steps = scheduler_step_counts(
        steps_per_epoch,
        int(config["accum_steps"]),
        int(config["epochs"]),
        int(config["warmup_epochs"]),
    )
    data_summary = {
        "image_root": str(image_root),
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": _file_sha256(train_manifest),
        "valid_manifest": str(valid_manifest),
        "valid_manifest_sha256": _file_sha256(valid_manifest),
        "train_studies": len(train_data),
        "train_images": sum(len(study.image_paths) for study in train_data.studies),
        "valid_studies": len(valid_data),
        "valid_images": sum(len(study.image_paths) for study in valid_data.studies),
        "excluded_train_paths": (Path(config["_project_root"]) / "data/manifests/excluded_train_paths.txt").read_text(encoding="utf-8").splitlines(),
    }
    git_state = git_working_tree_state(config["_project_root"])
    run_manifest = {
        "status": "running",
        "config": _plain_config(config),
        "data": data_summary,
        "labels": list(LABEL_NAMES),
        "hard_labels": [LABEL_NAMES[index] for index in config["hard_label_indices"]],
        "pos_weights": pos_weights_list,
        "pos_weight_source": config["pos_weight_mode"],
        "loss": {
            "name": config["loss"],
            "focal_gamma": float(config["focal_gamma"]),
        },
        "normalization": normalization_stats or {
            "name": config["normalization"],
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "pretrained_weights": {
            "type": "pure ImageNet-22K ConvNeXt-Base (non-medical; 21,841 classes)",
            "model": "convnext_base.fb_in22k",
            "path": str(weight_path),
            "sha256": _file_sha256(weight_path) if weight_path.is_file() else None,
            "resumed_from": str(Path(resume).resolve()) if resume else None,
        },
        "device": str(device),
        "git": git_state,
        "git_revision": git_state["revision"],
        "started_at_unix": time.time(),
    }
    _write_json(run_dir / "run_manifest.json", run_manifest)

    metrics_path = run_dir / "epoch_metrics.csv"
    fields = ["epoch", "train_loss", "macro_auc", *[f"auc_label_{i}" for i in range(10)]]
    if not metrics_path.exists():
        with metrics_path.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()

    try:
        for epoch in range(start_epoch, int(config["epochs"])):
            train_loss, global_step = _train_epoch(
                model, train_loader, len(train_data), optimizer, scaler, device, config,
                epoch, global_step, total_steps, warmup_steps, pos_weights,
            )
            val_metrics, study_ids, targets, probabilities = evaluate(
                model, valid_loader, device, bool(config["amp"]),
            )
            score = val_metrics["macro_auc"]
            if score is None:
                raise RuntimeError("validation macro AUC is undefined; every label needs positive and negative Studies")
            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "macro_auc": score,
                **{f"auc_label_{i}": value for i, value in enumerate(val_metrics["auc_per_label"])},
            }
            with metrics_path.open("a", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=fields).writerow(row)

            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "best_auc": max(best_auc, score),
                "best_epoch": epoch if score >= best_auc else best_epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "pos_weights": pos_weights_list,
                "hard_label_indices": list(config["hard_label_indices"]),
                "config": _plain_config(config),
            }
            torch.save(checkpoint, run_dir / "checkpoint-last.pth")
            if score >= best_auc:
                best_auc = score
                best_epoch = epoch
                torch.save(checkpoint, run_dir / "checkpoint-best.pth")
                _write_validation_predictions(
                    run_dir / "valid_best_predictions.csv", study_ids, targets, probabilities,
                )
            run_manifest["last_epoch"] = epoch + 1
            run_manifest["best_epoch"] = best_epoch + 1 if best_epoch is not None else None
            run_manifest["best_macro_auc"] = best_auc
            run_manifest["last_validation"] = val_metrics
            _write_json(run_dir / "run_manifest.json", run_manifest)
            print(
                f"epoch {epoch + 1}/{config['epochs']} train_loss={train_loss:.5f} "
                f"val_macro_auc={score:.5f} best={best_auc:.5f}",
                flush=True,
            )
        run_manifest["status"] = "completed"
    except BaseException as error:
        run_manifest["status"] = "failed"
        run_manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        run_manifest["finished_at_unix"] = time.time()
        _write_json(run_dir / "run_manifest.json", run_manifest)
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the G1 seed42 ConvNeXt-Base model.")
    parser.add_argument("--config", type=Path, default=Path("configs/g1_seed42.yaml"))
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Override the image root containing train/, valid/, and test/ folders")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    run_dir = train(args.config, args.data_root, args.output_dir, args.resume, args.device)
    print(f"Training outputs: {run_dir}")


if __name__ == "__main__":
    main()
