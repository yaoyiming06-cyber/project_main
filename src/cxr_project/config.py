"""Load and validate the standalone YAML training configuration."""

from pathlib import Path

import yaml


REQUIRED_KEYS = {
    "run_name", "seed", "model", "pretrained", "pretrained_checkpoint",
    "input_size", "epochs", "warmup_epochs", "batch_size_studies", "accum_steps",
    "lr", "min_lr", "weight_decay", "layer_decay", "drop_path", "num_workers",
    "max_pos_weight", "expert_loss_weight", "hard_label_indices", "image_root",
    "train_manifest", "valid_manifest", "test_manifest", "output_dir",
}


def load_config(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config must contain a YAML mapping: {path}")
    missing = REQUIRED_KEYS - set(config)
    if missing:
        raise ValueError(f"{path} is missing config keys: {sorted(missing)}")
    for key in ("input_size", "epochs", "batch_size_studies", "accum_steps", "num_workers"):
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be at least 1")
    if int(config["warmup_epochs"]) < 0 or int(config["warmup_epochs"]) >= int(config["epochs"]):
        raise ValueError("warmup_epochs must be non-negative and less than epochs")
    for key in ("lr", "min_lr", "weight_decay", "max_pos_weight"):
        if float(config[key]) < 0:
            raise ValueError(f"{key} must be non-negative")
    if float(config["lr"]) <= 0 or float(config["max_pos_weight"]) <= 0:
        raise ValueError("lr and max_pos_weight must be positive")
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
