"""Full Windows/CUDA training entry point for the G1 seed42 experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
import yaml
from contextlib import contextmanager, nullcontext
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
from .losses import combine_main_and_expert_logits, compute_training_loss, mixed_training_loss
from .sampling import build_train_sampler
from .runtime import optimizer_update
from .manifest import resolve_pos_weights
from .metrics import compute_macro_auc
from .model import build_model, SUPPORTED_MODELS
from .provenance import (
    dataset_normalization_source,
    git_working_tree_state,
    validate_dataset_normalization_cache,
)
from .schedule import scheduler_step_counts


class ModelEMA:
    """Keep a moving-average copy of model state for validation and inference."""

    def __init__(self, model, decay=0.9998):
        if not 0.0 < float(decay) < 1.0:
            raise ValueError("ema_decay must be between 0 and 1")
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    def reset(self, model):
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    def update(self, model):
        with torch.no_grad():
            for key, value in model.state_dict().items():
                if value.is_floating_point():
                    self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
                else:
                    self.shadow[key] = value.detach().clone()

    def state_dict(self):
        return {key: value.detach().clone() for key, value in self.shadow.items()}

    def load_state_dict(self, state):
        if set(state) != set(self.shadow):
            raise ValueError("EMA state does not match model state")
        self.shadow = {key: value.detach().clone() for key, value in state.items()}

    @contextmanager
    def use(self, model):
        raw_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(raw_state, strict=True)


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


def _optimizer_layer_id(name: str, max_layer=5):
    """Map timm ConvNeXt stem and four stages to layer-decay levels."""
    if not name.startswith("backbone."):
        return max_layer
    parts = name.split(".")
    if len(parts) >= 2 and parts[1] in {"stem", "patch_embed", "pos_embed", "cls_token"}:
        return 0
    if len(parts) >= 3 and parts[1] in {"stages", "layers", "blocks"} and parts[2].isdigit():
        return int(parts[2]) + 1
    return max_layer


def build_optimizer(model, lr, weight_decay, layer_decay):
    """AdamW groups with stage-wise ConvNeXt layer decay."""
    groups = {}
    blocks = getattr(model.backbone, "blocks", None)
    max_layer = len(blocks) + 1 if blocks is not None else 5
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        layer_id = _optimizer_layer_id(name, max_layer)
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


def evaluate(model, loader, device, use_amp, label_indices=None):
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
    metric_targets = targets.copy()
    if label_indices is not None:
        metric_targets[:, [i for i in range(10) if i not in label_indices]] = -1
    metrics = compute_macro_auc(metric_targets, probabilities)
    metrics["evaluated_label_indices"] = list(label_indices) if label_indices is not None else list(range(10))
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
    source["preprocessing"] = {key: config.get(key) for key in ("view_filter", "clahe", "clahe_clip_limit", "clahe_grid_size")}
    if stats_path.is_file():
        with stats_path.open("r", encoding="utf-8") as handle:
            stats = json.load(handle)
        return validate_dataset_normalization_cache(stats, source, stats_path)
    stats = compute_dataset_normalization(
        train_manifest,
        image_root,
        input_size=int(config["input_size"]),
        config=config,
    )
    stats["source"] = source
    _write_json(stats_path, stats)
    return stats


def _plain_config(config):
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _train_epoch(model, loader, dataset_size, optimizer, scaler, device, config,
                 epoch, global_step, total_steps, warmup_steps, pos_weights, ema=None):
    model.train()
    if config.get("freeze_backbone") or config.get("unfreeze_last_stages") or config.get("adaptformer"):
        model.backbone.eval()
        if config.get("unfreeze_last_stages"):
            stages = getattr(model.backbone, "stages", None)
            if stages is None:
                stages = getattr(model.backbone, "layers", None)
            for stage in list(stages)[-int(config["unfreeze_last_stages"]):]:
                stage.train()
        for module in model.backbone.modules():
            if any(p.requires_grad for p in module.parameters(recurse=False)):
                module.train()
        if config.get("adaptformer"):
            from .adapters import BottleneckAdapter
            for module in model.backbone.modules():
                if isinstance(module, BottleneckAdapter):
                    module.train()
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
        indices = config.get("train_label_indices")
        if indices is not None:
            targets = targets.clone()
            targets[:, [i for i in range(10) if i not in indices]] = -1
        alpha = float(config.get("mixup_alpha", 0.))
        use_mixup = alpha > 0 and targets.shape[0] > 1
        with _autocast(device, config["amp"]):
            if use_mixup:
                pooled = model.encode_studies(model._encode_images(images), offsets, view_ids)
                permutation = torch.randperm(pooled.shape[0], device=device)
                lam = float(np.random.beta(alpha, alpha))
                main_logits, expert_logits = model.classify_features(lam*pooled + (1-lam)*pooled[permutation])
            else:
                main_logits, expert_logits = model(images, offsets, view_ids)
        loss_kwargs = dict(hard_label_indices=model.hard_label_indices,
                           expert_loss_weight=config["expert_loss_weight"], loss_name=config["loss"],
                           focal_gamma=config["focal_gamma"], asl_gamma_neg=config.get("asl_gamma_neg",4.),
                           asl_gamma_pos=config.get("asl_gamma_pos",0.), asl_clip=config.get("asl_clip",.05))
        if use_mixup:
            loss, _ = mixed_training_loss(main_logits.float(), expert_logits.float(), targets,
                                         pos_weights, permutation, lam, **loss_kwargs)
        else:
            loss, _ = compute_training_loss(main_logits.float(), expert_logits.float(), targets, pos_weights, **loss_kwargs)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}, batch {step + 1}")
        current_studies = int(targets.shape[0])
        scaler.scale(loss * current_studies / window_studies).backward()
        running_loss += loss.detach().item() * current_studies
        seen_studies += current_studies

        should_step = (step + 1) % config["accum_steps"] == 0 or step + 1 == steps_per_epoch
        if should_step:
            if optimizer_update(model, optimizer, scaler, config.get("clip_grad"), ema):
                global_step += 1

        if (step + 1) % 100 == 0 or step + 1 == steps_per_epoch:
            print(
                f"epoch {epoch + 1}/{config['epochs']} batch {step + 1}/{steps_per_epoch} "
                f"loss={loss.detach().item():.5f} lr={optimizer.param_groups[-1]['lr']:.7g}",
                flush=True,
            )
    return running_loss / max(1, seen_studies), global_step


def _rng_state(generator, sampler):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader": generator.get_state(),
            "sampler": sampler.generator.get_state() if sampler is not None else None}


def _restore_rng(state, generator, sampler):
    if not state:
        return
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])
    generator.set_state(state["loader"].cpu())
    if sampler is not None and state.get("sampler") is not None:
        sampler.generator.set_state(state["sampler"].cpu())


def _validate_resume(checkpoint, config, data_summary):
    previous = checkpoint.get("config", {})
    ignored = {"output_dir", "run_name", "project_root", "num_workers", "pin_memory",
               "image_root", "pretrained_checkpoint", "train_manifest", "valid_manifest", "test_manifest"}
    changed = [key for key, value in previous.items() if key not in ignored and config.get(key) != value]
    if changed:
        raise ValueError(f"resume configuration differs: {changed}; use a new experiment")
    previous_data = checkpoint.get("data")
    if previous_data:
        for key in ("train_manifest_sha256", "valid_manifest_sha256"):
            if previous_data.get(key) != data_summary.get(key):
                raise ValueError(f"resume data differs: {key}")


def train(config_path, data_root=None, output_dir=None, resume=None, device_name="cuda"):
    config = load_config(config_path)
    if device_name not in ("cuda", "cpu"):
        raise ValueError("device must be cuda or cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    set_seed(int(config["seed"]))
    final = bool(config.get("final_training", False))
    train_manifest = resolve_path(config, "train_manifest")
    valid_manifest = None if final else resolve_path(config, "valid_manifest")
    image_root = Path(data_root).resolve() if data_root else resolve_path(config, "image_root")
    run_dir = Path(output_dir).resolve() if output_dir else resolve_path(config, "output_dir")
    if run_dir.exists() and resume is None:
        raise FileExistsError(f"run directory already exists: {run_dir}")
    data_summary = {"image_root": str(image_root), "train_manifest": str(train_manifest),
                    "valid_manifest": str(valid_manifest) if valid_manifest else None,
                    "train_manifest_sha256": _file_sha256(train_manifest),
                    "valid_manifest_sha256": _file_sha256(valid_manifest) if valid_manifest else None}
    checkpoint = torch.load(Path(resume), map_location="cpu", weights_only=False) if resume else None
    if checkpoint is not None:
        _validate_resume(checkpoint, config, data_summary)
    run_dir.mkdir(parents=True, exist_ok=True)
    normalization_stats = _normalization_stats(config, train_manifest, image_root, run_dir)
    train_transform = build_transform(True, int(config["input_size"]), config, normalization_stats)
    valid_transform = build_transform(False, int(config["input_size"]), config, normalization_stats)
    train_data = StudyDataset(
        train_manifest, image_root, train_transform,
        view_filter=config.get("view_filter"),
        use_view_embedding=config.get("use_view_embedding", True),
    )
    valid_data = None if final else StudyDataset(
        valid_manifest, image_root, valid_transform,
        view_filter=config.get("view_filter"),
        use_view_embedding=config.get("use_view_embedding", True),
    )
    for prefix, dataset in (("train", train_data), ("valid", valid_data)):
        data_summary[prefix+"_studies"] = len(dataset) if dataset else 0
        data_summary[prefix+"_images"] = sum(len(s.image_paths) for s in dataset.studies) if dataset else 0
        data_summary[prefix+"_coverage"] = dataset.coverage if dataset else None
    pos_weights_list = resolve_pos_weights(train_data.studies, mode=config["pos_weight_mode"],
                                          cap=float(config["max_pos_weight"]), manual=config["manual_pos_weights"])
    pos_weights = torch.tensor(pos_weights_list, dtype=torch.float32, device=device)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    sampler = build_train_sampler(train_data.studies, config, int(config["seed"]))
    loader_options = dict(num_workers=int(config["num_workers"]), pin_memory=bool(config["pin_memory"] and device.type == "cuda"),
                          collate_fn=collate_studies, persistent_workers=False)
    if config["num_workers"]:
        loader_options["prefetch_factor"] = 2
    def training_loader(active_sampler):
        return DataLoader(train_data, batch_size=int(config["batch_size_studies"]), shuffle=active_sampler is None,
                          sampler=active_sampler, generator=generator, drop_last=False, **loader_options)
    train_loader = training_loader(sampler)
    warm_loader = training_loader(None) if sampler is not None and config.get("sampling_warmup_epochs",0) else train_loader
    valid_loader = None if final else DataLoader(valid_data, batch_size=int(config["batch_size_studies"]), shuffle=False, **loader_options)
    weight_path = resolve_path(config, "pretrained_checkpoint") if config.get("pretrained_checkpoint") else None
    model = build_model(config, pretrained_checkpoint=weight_path, pretrained=bool(config["pretrained"]) and not resume).to(device)
    optimizer = build_optimizer(model, config["lr"], config["weight_decay"], config["layer_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config["amp"])
    ema = ModelEMA(model, config["ema_decay"]) if config["ema"] else None
    ema_active = False
    start_epoch = global_step = bad_epochs = 0
    best_auc, best_epoch = float("-inf"), None
    best_tracks = {}
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        if ema is not None and checkpoint.get("ema") is not None:
            ema.load_state_dict({k:v.to(device) for k,v in checkpoint["ema"].items()})
            ema_active = bool(checkpoint.get("ema_active", True))
        start_epoch = int(checkpoint["epoch"])+1
        global_step = int(checkpoint.get("global_step",0))
        best_auc, best_epoch = checkpoint.get("best_auc",float("-inf")), checkpoint.get("best_epoch")
        bad_epochs, best_tracks = checkpoint.get("bad_epochs",0), checkpoint.get("best_tracks",{})
        _restore_rng(checkpoint.get("rng_state"), generator, sampler)
        if Path(resume).resolve().parent != run_dir:
            import shutil
            for pattern in ("checkpoint-best*.pth", "valid_best*predictions.csv", "epoch_metrics*.csv"):
                for previous in Path(resume).resolve().parent.glob(pattern):
                    if not (run_dir/previous.name).exists(): shutil.copy2(previous,run_dir/previous.name)
    (run_dir/"config_resolved.yaml").write_text(yaml.safe_dump(_plain_config(config),allow_unicode=True,sort_keys=False),encoding="utf-8")
    total_steps, warmup_steps = scheduler_step_counts(len(train_loader), config["accum_steps"], config["epochs"], config["warmup_epochs"])
    git_state = git_working_tree_state(config["_project_root"])
    norm = next(op for op in reversed(valid_transform.transforms) if hasattr(op,"mean") and hasattr(op,"std"))
    run_manifest = {"status":"running", "config":_plain_config(config), "data":data_summary,
                    "labels":list(LABEL_NAMES), "hard_labels":[LABEL_NAMES[i] for i in model.hard_label_indices],
                    "pos_weights":pos_weights_list, "pos_weight_source":config["pos_weight_mode"],
                    "normalization":normalization_stats or {"name":config["normalization"],"mean":list(norm.mean),"std":list(norm.std)},
                    "pretrained_weights":{"type":SUPPORTED_MODELS.get(config["model"]),"model":config["model"],"path":str(weight_path),
                                          "sha256":_file_sha256(weight_path) if weight_path and weight_path.is_file() else None},
                    "device":str(device),"git":git_state,"git_revision":git_state["revision"],"started_at_unix":time.time()}
    fields = ["epoch","train_loss","macro_auc", *[f"auc_label_{i}" for i in range(10)]]
    def append_metrics(path,epoch,loss,metrics):
        exists=path.exists()
        with path.open("a",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=fields)
            if not exists: writer.writeheader()
            writer.writerow({"epoch":epoch+1,"train_loss":loss,"macro_auc":metrics.get("macro_auc"),
                             **{f"auc_label_{i}":v for i,v in enumerate(metrics.get("auc_per_label",[None]*10))}})
    _write_json(run_dir/"run_manifest.json",run_manifest)
    try:
        for epoch in range(start_epoch,config["epochs"]):
            if ema is not None and not ema_active and epoch+1 >= config["ema_start_epoch"]:
                ema.reset(model); ema_active=True
            loader=warm_loader if epoch < config.get("sampling_warmup_epochs",0) else train_loader
            train_loss,global_step = _train_epoch(model,loader,len(train_data),optimizer,scaler,device,config,epoch,global_step,total_steps,warmup_steps,pos_weights,ema if ema_active else None)
            chosen = "ema" if ema_active and config["validation_weights"] != "raw" else "raw"
            results, improved = {}, []
            if not final:
                sources = ["raw","ema"] if config["evaluate_raw_and_ema"] and ema_active else [chosen]
                for source in sources:
                    with ema.use(model) if source == "ema" else nullcontext():
                        result=evaluate(model,valid_loader,device,config["amp"],config.get("train_label_indices"))
                    metrics=result[0]; score=metrics["macro_auc"]
                    if score is None: raise RuntimeError("validation AUC undefined for selected labels")
                    results[source]=result
                    append_metrics(run_dir/f"epoch_metrics_{source}.csv",epoch,train_loss,metrics)
                    if score >= best_tracks.get(source,{}).get("auc",float("-inf")):
                        best_tracks[source]={"auc":score,"epoch":epoch}; improved.append(source)
                score=results[chosen][0]["macro_auc"]
                previous_best=best_auc
                is_best=score >= best_auc
                if is_best: best_auc,best_epoch=score,epoch
                meaningful=previous_best == float("-inf") or score > previous_best+config["early_stopping_min_delta"]
                bad_epochs=0 if meaningful else bad_epochs+1
                append_metrics(run_dir/"epoch_metrics.csv",epoch,train_loss,results[chosen][0])
            else:
                is_best=False
                append_metrics(run_dir/"epoch_metrics.csv",epoch,train_loss,{})
            state={"epoch":epoch,"global_step":global_step,"best_auc":best_auc,"best_epoch":best_epoch,"bad_epochs":bad_epochs,"best_tracks":best_tracks,
                   "model":model.state_dict(),"ema":ema.state_dict() if ema else None,"ema_active":ema_active,"selected_weights":chosen,
                   "optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),"pos_weights":pos_weights_list,
                   "hard_label_indices":list(model.hard_label_indices),"config":_plain_config(config),"data":data_summary,
                   "rng_state":_rng_state(generator,sampler),"normalization":run_manifest["normalization"]}
            torch.save(state,run_dir/"checkpoint-last.pth")
            for source in improved:
                torch.save({**state,"selected_weights":source},run_dir/f"checkpoint-best-{source}.pth")
                _write_validation_predictions(run_dir/f"valid_best_{source}_predictions.csv",*results[source][1:])
            if is_best:
                torch.save(state,run_dir/"checkpoint-best.pth")
                _write_validation_predictions(run_dir/"valid_best_predictions.csv",*results[chosen][1:])
            if final and epoch+1 == config["epochs"]: torch.save(state,run_dir/"checkpoint-final.pth")
            run_manifest.update(last_epoch=epoch+1,best_epoch=best_epoch+1 if best_epoch is not None else None,
                                best_macro_auc=best_auc if best_epoch is not None else None,best_tracks=best_tracks,
                                selected_weights=chosen,early_stopping_bad_epochs=bad_epochs,
                                last_validation=results[chosen][0] if results else None)
            _write_json(run_dir/"run_manifest.json",run_manifest)
            print(f"epoch {epoch+1}/{config['epochs']} train_loss={train_loss:.5f} selected={chosen} best={best_auc}",flush=True)
            if not final and config["early_stopping_patience"] and bad_epochs >= config["early_stopping_patience"]: break
        run_manifest["status"]="completed"
    except BaseException as error:
        run_manifest.update(status="failed",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        run_manifest["finished_at_unix"]=time.time(); _write_json(run_dir/"run_manifest.json",run_manifest)
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
