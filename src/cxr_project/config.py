"""Load and validate the standalone YAML training configuration."""

import math
from pathlib import Path

import yaml


REQUIRED_KEYS = {
    "run_name", "seed", "model", "pretrained", "pretrained_checkpoint",
    "input_size", "epochs", "warmup_epochs", "batch_size_studies", "accum_steps",
    "lr", "min_lr", "weight_decay", "layer_decay", "drop_path", "num_workers",
    "max_pos_weight", "pos_weight_mode", "manual_pos_weights",
    "loss", "focal_gamma", "expert_loss_weight", "hard_label_indices", "image_root",
    "train_manifest", "valid_manifest", "test_manifest", "output_dir",
    "label_strategy", "blank_policy", "uncertain_policy",
    "optimizer", "amp", "pin_memory", "gradient_checkpointing", "image_chunk_size",
    "clip_grad", "augmentation", "normalization",
}


def _finite_float(config, key):
    try:
        value = float(config[key])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be a finite number") from error
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def _validate_finite_values(value, path="config"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_finite_values(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_finite_values(child, f"{path}[{index}]")


def load_config(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config must contain a YAML mapping: {path}")
    _validate_finite_values(config)
    missing = REQUIRED_KEYS - set(config)
    if missing:
        raise ValueError(f"{path} is missing config keys: {sorted(missing)}")

    numeric_keys = (
        "lr", "min_lr", "weight_decay", "layer_decay", "drop_path",
        "max_pos_weight", "expert_loss_weight", "focal_gamma", "clip_grad",
        "augmentation_translate", "augmentation_degrees",
        "color_jitter_brightness", "color_jitter_contrast",
    )
    for key in numeric_keys:
        if key in config and config[key] is not None:
            config[key] = _finite_float(config, key)

    for key in ("input_size", "epochs", "batch_size_studies", "accum_steps", "num_workers"):
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be at least 1")
    if int(config["warmup_epochs"]) < 0 or int(config["warmup_epochs"]) >= int(config["epochs"]):
        raise ValueError("warmup_epochs must be non-negative and less than epochs")
    for key in ("lr", "min_lr", "weight_decay", "max_pos_weight", "focal_gamma"):
        if float(config[key]) < 0:
            raise ValueError(f"{key} must be non-negative")
    if float(config["lr"]) <= 0 or float(config["max_pos_weight"]) <= 0:
        raise ValueError("lr and max_pos_weight must be positive")
    if float(config["min_lr"]) > float(config["lr"]):
        raise ValueError("min_lr must not exceed lr")
    if not 0 < float(config["layer_decay"]) <= 1:
        raise ValueError("layer_decay must be in (0, 1]")
    if not 0 <= float(config["drop_path"]) < 1:
        raise ValueError("drop_path must be in [0, 1)")
    if float(config["focal_gamma"]) <= 0:
        raise ValueError("focal_gamma must be positive")
    if float(config["expert_loss_weight"]) < 0:
        raise ValueError("expert_loss_weight must be non-negative")
    if str(config["label_strategy"]).upper() != "E0":
        raise ValueError("this training entry point only implements label_strategy E0")
    if str(config["blank_policy"]).lower() != "zero":
        raise ValueError("this training entry point only implements blank_policy zero")
    if str(config["uncertain_policy"]).lower() != "ignore":
        raise ValueError("this training entry point only implements uncertain_policy ignore")
    if str(config["optimizer"]).lower() != "adamw":
        raise ValueError("only the configured AdamW optimizer is supported")
    if str(config["loss"]).lower() not in {"bce", "focal"}:
        raise ValueError("loss must be 'bce' or 'focal'")
    if str(config["pos_weight_mode"]).lower() not in {"ratio", "unit", "sqrt", "manual"}:
        raise ValueError("pos_weight_mode must be ratio, unit, sqrt, or manual")
    manual_weights = config["manual_pos_weights"]
    if manual_weights is not None:
        if not isinstance(manual_weights, (list, tuple)) or len(manual_weights) != 10:
            raise ValueError("manual_pos_weights must contain ten positive values or be null")
        manual_weights = [float(value) for value in manual_weights]
        if any(not math.isfinite(value) for value in manual_weights):
            raise ValueError("manual_pos_weights values must be finite")
        if any(value <= 0 for value in manual_weights):
            raise ValueError("manual_pos_weights must contain ten positive values or be null")
        config["manual_pos_weights"] = manual_weights
    if config["clip_grad"] is not None and float(config["clip_grad"]) <= 0:
        raise ValueError("clip_grad must be positive or null")
    if str(config["augmentation"]).upper() not in {"G0", "G1", "AUTOAUG"}:
        raise ValueError("augmentation must be G0, G1, or AutoAug")
    scale = config.get("augmentation_scale", (0.98, 1.02))
    if not isinstance(scale, (list, tuple)) or len(scale) != 2:
        raise ValueError("augmentation_scale must be a finite increasing positive pair")
    try:
        scale = [float(value) for value in scale]
    except (TypeError, ValueError) as error:
        raise ValueError("augmentation_scale must be a finite increasing positive pair") from error
    if any(not math.isfinite(value) for value in scale):
        raise ValueError("augmentation_scale values must be finite")
    if scale[0] <= 0 or scale[1] < scale[0]:
        raise ValueError("augmentation_scale must be an increasing positive pair")
    config["augmentation_scale"] = scale
    for key in (
        "augmentation_translate", "augmentation_degrees",
        "color_jitter_brightness", "color_jitter_contrast",
    ):
        if key in config and config[key] is not None and float(config[key]) < 0:
            raise ValueError(f"{key} must be non-negative")
    if str(config["normalization"]).lower() not in {"imagenet1k", "dataset"}:
        raise ValueError("normalization must be imagenet1k or dataset")
    if int(config["image_chunk_size"]) < 1:
        raise ValueError("image_chunk_size must be at least 1")
    if str(config["pos_weight_mode"]).lower() == "manual" and manual_weights is None:
        raise ValueError("manual pos_weight_mode requires manual_pos_weights")
    if not config["hard_label_indices"] or any(
        int(index) not in range(10) for index in config["hard_label_indices"]
    ):
        raise ValueError("hard_label_indices must be non-empty and within 0..9")
    config["_config_path"] = path
    config["_project_root"] = path.parent.parent
    return config


def resolve_path(config, key):
    value = Path(config[key]).expanduser()
    return value.resolve() if value.is_absolute() else (config["_project_root"] / value).resolve()
