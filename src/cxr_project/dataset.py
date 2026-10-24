"""Study-level dataset and configuration-driven image preprocessing."""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import replace

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from .labels import VIEW_TO_ID, normalize_view_position
from .augmentation import CLAHE, RandomGamma
from .manifest import StudyRecord, read_study_manifest

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResizePadToSquare:
    """Resize the full radiograph into a square canvas without cropping."""

    def __init__(self, size: int, fill: int = 0):
        self.size = int(size)
        self.fill = int(fill)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = self.size / max(width, height)
        resized = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BICUBIC,
        )
        pad_width = self.size - resized.width
        pad_height = self.size - resized.height
        border = (
            pad_width // 2,
            pad_height // 2,
            pad_width - pad_width // 2,
            pad_height - pad_height // 2,
        )
        return ImageOps.expand(resized, border=border, fill=self.fill)


def build_transform(
    is_train: bool,
    input_size: int = 224,
    config: dict | None = None,
    normalization: dict | None = None,
):
    """Build the configured transform.

    Validation always uses deterministic resize, tensor conversion and
    normalization.  Training-only operations are selected by ``augmentation``
    and are intentionally explicit so an experiment cannot silently fall back
    to G1 after a typo.
    """
    config = config or {}
    augmentation = str(config.get("augmentation", "G1")).strip().upper()
    if augmentation not in {"G0", "G1", "G2", "G3", "AUTOAUG"}:
        raise ValueError(f"unsupported augmentation strategy: {augmentation!r}")

    normalization_name = str(config.get("normalization", "imagenet1k")).strip().lower()
    if normalization_name == "pretrained":
        import timm
        name = config.get("model", "convnext_base.fb_in22k")
        pretrained_cfg = timm.models.get_pretrained_cfg(name)
        if pretrained_cfg is None:
            raise ValueError(f"No pretrained normalization registered for {name}")
        mean, std = pretrained_cfg.mean, pretrained_cfg.std
    elif normalization_name == "imagenet1k":
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    elif normalization_name == "dataset":
        if normalization is None:
            mean = config.get("normalization_mean")
            std = config.get("normalization_std")
        else:
            mean = normalization.get("mean")
            std = normalization.get("std")
        if mean is None or std is None or len(mean) != 3 or len(std) != 3:
            raise ValueError(
                "dataset normalization requires three-channel normalization mean/std"
            )
        mean = tuple(float(value) for value in mean)
        std = tuple(float(value) for value in std)
        if any(not math.isfinite(value) for value in (*mean, *std)):
            raise ValueError("dataset normalization mean/std values must be finite")
        if any(value <= 0 for value in std):
            raise ValueError("normalization standard deviations must be positive")
    else:
        raise ValueError(f"unsupported normalization: {normalization_name!r}")

    ops = []
    if bool(config.get("clahe", False)):
        ops.append(CLAHE(config.get("clahe_clip_limit", 2.), config.get("clahe_grid_size", (8, 8))))
    ops.append(ResizePadToSquare(input_size))
    if is_train and augmentation != "G0":
        if augmentation in {"G1", "G2", "G3"}:
            scale = config.get("augmentation_scale", (0.95, 1.05) if augmentation == "G2" else (0.98, 1.02))
            if len(scale) != 2 or float(scale[0]) <= 0 or float(scale[1]) < float(scale[0]):
                raise ValueError("augmentation_scale must be an increasing positive pair")
            ops.extend([
                transforms.RandomAffine(
                    degrees=float(config.get("augmentation_degrees", 5 if augmentation == "G2" else 3)),
                    translate=(
                        float(config.get("augmentation_translate", 0.04 if augmentation == "G2" else 0.02)),
                        float(config.get("augmentation_translate", 0.04 if augmentation == "G2" else 0.02)),
                    ),
                    scale=(float(scale[0]), float(scale[1])),
                    fill=0,
                ),
                transforms.ColorJitter(
                    brightness=float(config.get("color_jitter_brightness", 0.08)),
                    contrast=float(config.get("color_jitter_contrast", 0.08)),
                ),
            ])
        elif augmentation == "AUTOAUG":
            ops.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET))
        if bool(config.get("horizontal_flip", False)) or augmentation == "G3":
            ops.append(transforms.RandomHorizontalFlip(p=float(config.get("horizontal_flip_prob", .5))))
        if float(config.get("gamma_prob", 0.)) > 0:
            ops.append(RandomGamma(config["gamma_prob"], config.get("gamma_range", (.8, 1.2))))
    ops.append(transforms.ToTensor())
    if is_train and augmentation != "G0" and float(config.get("random_erasing_prob", 0.)) > 0:
        ops.append(transforms.RandomErasing(
            p=float(config["random_erasing_prob"]),
            scale=tuple(config.get("random_erasing_scale", (.005, .02))),
            ratio=tuple(config.get("random_erasing_ratio", (.3, 3.3))), value=0.,
        ))
    ops.append(transforms.Normalize(mean, std))
    return transforms.Compose(ops)


def filter_study_views(studies, view_filter=None):
    """Retain selected images, excluding uncovered Studies with explicit coverage."""
    selected = None if view_filter is None else {str(v).strip().upper() for v in view_filter}
    if selected is not None and (not selected or selected - set(VIEW_TO_ID)):
        raise ValueError("view_filter must contain known view names")
    filtered = []
    for study in studies:
        indices = [i for i,v in enumerate(study.view_positions) if selected is None or normalize_view_position(v) in selected]
        if indices:
            filtered.append(replace(study, image_paths=tuple(study.image_paths[i] for i in indices),
                                    view_positions=tuple(study.view_positions[i] for i in indices)))
    retained_ids = {s.study_id for s in filtered}
    coverage = {"total_studies": len(studies), "retained_studies": len(filtered),
                "total_images": sum(len(s.image_paths) for s in studies),
                "retained_images": sum(len(s.image_paths) for s in filtered),
                "view_filter": sorted(selected) if selected is not None else None,
                "dropped_study_ids": [s.study_id for s in studies if s.study_id not in retained_ids]}
    return filtered, coverage


class StudyDataset(Dataset):
    """Load every view in a Study together for attention-based aggregation."""

    def __init__(self, manifest_path, image_root, transform=None, labeled=True, verify_paths=True, view_filter=None):
        self.manifest_path = Path(manifest_path).resolve()
        self.image_root = Path(image_root).resolve()
        self.studies, self.coverage = filter_study_views(read_study_manifest(self.manifest_path, labeled=labeled), view_filter)
        if not self.studies:
            raise ValueError("view_filter leaves no covered Studies")
        self.transform = transform
        self.labeled = labeled
        self.resolved_image_paths = [
            tuple(self._resolve_path(path) for path in study.image_paths)
            for study in self.studies
        ]
        if verify_paths:
            missing = [
                str(path)
                for paths in self.resolved_image_paths
                for path in paths
                if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} manifest images are missing; first paths: {missing[:5]}"
                )

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        path = path if path.is_absolute() else self.image_root / path
        resolved = path.resolve()
        if not path.is_absolute():
            try:
                resolved.relative_to(self.image_root)
            except ValueError as error:
                raise ValueError(f"image path escapes data root: {value!r}") from error
        return resolved

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, index):
        study: StudyRecord = self.studies[index]
        images = []
        for path in self.resolved_image_paths[index]:
            with Image.open(path) as source:
                image = source.convert("RGB")
            images.append(self.transform(image) if self.transform else image)
        view_ids = [
            VIEW_TO_ID[normalize_view_position(view)] for view in study.view_positions
        ]
        labels = None if study.labels is None else torch.tensor(study.labels, dtype=torch.float32)
        return {
            "images": images,
            "view_ids": view_ids,
            "labels": labels,
            "study_id": study.study_id,
        }


def compute_dataset_normalization(manifest_path, image_root, input_size: int = 224, config=None):
    """Compute RGB mean and standard deviation after resize and square padding.

    The statistics are accumulated per pixel without retaining images in
    memory.  They are therefore deterministic for a fixed manifest and image
    tree, and can be cached by the training entry point for resumed runs.
    """
    config = config or {}
    dataset = StudyDataset(
        manifest_path,
        image_root,
        transform=None,
        labeled=True,
        verify_paths=True,
        view_filter=config.get("view_filter"),
    )
    resize = ResizePadToSquare(input_size)
    clahe = CLAHE(config.get("clahe_clip_limit", 2.), config.get("clahe_grid_size", (8, 8))) if config.get("clahe", False) else None
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for paths in dataset.resolved_image_paths:
        for path in paths:
            with Image.open(path) as source:
                image = source.convert("RGB")
                if clahe is not None:
                    image = clahe(image)
                image = np.asarray(resize(image), dtype=np.float32) / 255.0
            pixels = image.reshape(-1, 3)
            channel_sum += pixels.sum(axis=0, dtype=np.float64)
            channel_square_sum += np.square(pixels, dtype=np.float32).sum(axis=0, dtype=np.float64)
            pixel_count += pixels.shape[0]
    if pixel_count == 0:
        raise ValueError("cannot compute normalization statistics from an empty manifest")
    mean = channel_sum / pixel_count
    variance = np.maximum(channel_square_sum / pixel_count - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    return {
        "name": "dataset",
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "input_size": int(input_size),
        "images": sum(len(paths) for paths in dataset.resolved_image_paths),
        "pixels_per_image": int(input_size) * int(input_size),
    }


def collate_studies(batch):
    """Flatten variable-view studies and preserve offsets for model pooling."""
    if not batch:
        raise ValueError("cannot collate an empty study batch")
    images = []
    view_ids = []
    offsets = [0]
    labels = []
    study_ids = []
    for item in batch:
        if not item["images"]:
            raise ValueError(f"Study {item['study_id']!r} contains no images")
        images.extend(item["images"])
        view_ids.extend(item["view_ids"])
        offsets.append(len(images))
        if item["labels"] is not None:
            labels.append(item["labels"])
        study_ids.append(item["study_id"])
    result = {
        "images": torch.stack(images),
        "view_ids": torch.tensor(view_ids, dtype=torch.long),
        "offsets": torch.tensor(offsets, dtype=torch.long),
        "study_ids": study_ids,
    }
    if labels:
        if len(labels) != len(batch):
            raise ValueError("a batch cannot mix labeled and unlabeled studies")
        result["targets"] = torch.stack(labels)
    return result
