"""Study-level PyTorch dataset and G1 image preprocessing."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from .labels import VIEW_TO_ID, normalize_view_position
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


def build_transform(is_train: bool, input_size: int = 224):
    ops = [ResizePadToSquare(input_size)]
    if is_train:
        # G1: light affine and brightness/contrast changes; no horizontal flip.
        ops.extend([
            transforms.RandomAffine(
                degrees=3,
                translate=(0.02, 0.02),
                scale=(0.98, 1.02),
                fill=0,
            ),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
        ])
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transforms.Compose(ops)


class StudyDataset(Dataset):
    """Load every view in a Study together for attention-based aggregation."""

    def __init__(self, manifest_path, image_root, transform=None, labeled=True, verify_paths=True):
        self.manifest_path = Path(manifest_path).resolve()
        self.image_root = Path(image_root).resolve()
        self.studies = read_study_manifest(self.manifest_path, labeled=labeled)
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
