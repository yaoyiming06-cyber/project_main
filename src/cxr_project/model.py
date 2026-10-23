"""ImageNet-22K ConvNeXt-Base with Study attention and label experts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .labels import HARD_LABEL_INDICES, NUM_LABELS, VIEW_NAMES


MODEL_NAME = "convnext_base.fb_in22k"
WEIGHTS_FILENAME = "convnext_base.fb_in22k.safetensors"
WEIGHTS_URL = (
    "https://huggingface.co/timm/convnext_base.fb_in22k/resolve/"
    "9afd4146adabf1b6e861d0f1ac2e7baa7c8dd1b7/model.safetensors"
)
WEIGHTS_SIZE = 439_848_696
WEIGHTS_SHA256 = "0f09df05894c7fd1e78569daaae9e839a7b3bc5666587a08ba1d8e8cea63a8a4"
PRETRAINED_NUM_CLASSES = 21_841


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_imagenet_22k_state_dict(weights_path):
    """Load and verify the pinned, pure ImageNet-22K timm checkpoint."""
    path = Path(weights_path) if weights_path else None
    if path is None or not path.is_file():
        raise FileNotFoundError(
            f"ImageNet-22K weights not found: {path}; run scripts/download_weights.py"
        )
    if path.stat().st_size != WEIGHTS_SIZE:
        raise ValueError(
            f"Unexpected ImageNet-22K checkpoint size: {path.stat().st_size} bytes"
        )
    digest = file_sha256(path)
    if digest != WEIGHTS_SHA256:
        raise ValueError(f"ImageNet-22K checkpoint SHA-256 mismatch: {digest}")

    from safetensors.torch import load_file

    state = load_file(str(path), device="cpu")
    if not state:
        raise ValueError(f"ImageNet-22K checkpoint is empty: {path}")
    return state


class ConvNeXtStudyClassifier(nn.Module):
    """Aggregate every image in a Study, then predict 10 labels plus experts.

    The specialist heads cover Enlarged Cardiomediastinum, Pneumonia, and
    Lung Opacity (label indices 0, 3, 7). Their logits are combined with the
    ordinary head at inference, while an auxiliary loss trains each expert.
    """

    def __init__(
        self,
        backbone=None,
        feature_dim=1024,
        pretrained=True,
        pretrained_checkpoint=None,
        drop_path=0.1,
        hard_label_indices=HARD_LABEL_INDICES,
        gradient_checkpointing=True,
        image_chunk_size=2,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hard_label_indices = tuple(int(index) for index in hard_label_indices)
        if len(set(self.hard_label_indices)) != len(self.hard_label_indices):
            raise ValueError("hard_label_indices must not contain duplicates")
        if any(index < 0 or index >= NUM_LABELS for index in self.hard_label_indices):
            raise ValueError("hard_label_indices must be in 0..9")
        if image_chunk_size < 1:
            raise ValueError("image_chunk_size must be positive")
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.image_chunk_size = int(image_chunk_size)

        if backbone is None:
            import timm

            if pretrained:
                weights_path = Path(pretrained_checkpoint) if pretrained_checkpoint else None
                if weights_path is None or not weights_path.is_file():
                    raise FileNotFoundError(
                        f"ImageNet-22K weights not found: {weights_path}; run scripts/download_weights.py"
                    )
                # timm registers weights-only variants only when a pretrained
                # config is requested; point timm at the verified local file.
                network = timm.create_model(
                    MODEL_NAME,
                    pretrained=True,
                    pretrained_cfg_overlay={"file": str(weights_path)},
                    num_classes=PRETRAINED_NUM_CLASSES,
                    drop_path_rate=float(drop_path),
                )
                network.load_state_dict(
                    load_imagenet_22k_state_dict(pretrained_checkpoint), strict=True,
                )
            else:
                network = timm.create_model(
                    MODEL_NAME,
                    pretrained=False,
                    num_classes=PRETRAINED_NUM_CLASSES,
                    drop_path_rate=float(drop_path),
                )
            network.reset_classifier(0)
            backbone = network
            self.feature_dim = 1024
        elif pretrained:
            raise ValueError("pretrained=True requires the built-in ConvNeXt-Base backbone")
        self.backbone = backbone

        self.view_embedding = nn.Embedding(len(VIEW_NAMES), 32)
        self.attention = nn.Sequential(
            nn.LayerNorm(self.feature_dim + 32),
            nn.Linear(self.feature_dim + 32, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.main_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, NUM_LABELS),
        )
        self.expert_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, 256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1),
            )
            for _ in self.hard_label_indices
        ])

    def _features_from_chunk(self, image_chunk):
        if hasattr(self.backbone, "forward_features"):
            feature_map = self.backbone.forward_features(image_chunk)
            if hasattr(self.backbone, "forward_head"):
                features = self.backbone.forward_head(feature_map, pre_logits=True)
            else:
                features = feature_map
        else:
            features = self.backbone(image_chunk)
        if features.ndim == 4:
            features = features.mean(dim=(-2, -1))
        elif features.ndim != 2:
            raise ValueError(
                "backbone must return [images, channels, height, width] or [images, features]"
            )
        return features

    def _encode_images(self, images):
        feature_chunks = []
        for image_chunk in images.split(self.image_chunk_size, dim=0):
            if self.training and self.gradient_checkpointing:
                features = checkpoint(
                    self._features_from_chunk, image_chunk, use_reentrant=False,
                )
            else:
                features = self._features_from_chunk(image_chunk)
            feature_chunks.append(features)
        return torch.cat(feature_chunks, dim=0)

    def forward_from_features(self, image_features, offsets, view_ids):
        if image_features.ndim != 2 or image_features.shape[1] != self.feature_dim:
            raise ValueError(f"image_features must have shape [images, {self.feature_dim}]")
        if view_ids.ndim != 1 or view_ids.shape[0] != image_features.shape[0]:
            raise ValueError("view_ids must contain one view ID per image")
        boundaries = offsets.tolist() if torch.is_tensor(offsets) else list(offsets)
        if len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != image_features.shape[0]:
            raise ValueError("offsets must start at zero and end at the number of images")
        if any(end <= start for start, end in zip(boundaries, boundaries[1:])):
            raise ValueError("each Study must contain at least one image")

        study_features = []
        for start, end in zip(boundaries, boundaries[1:]):
            study_image_features = image_features[start:end]
            study_view_features = self.view_embedding(view_ids[start:end])
            attention_input = torch.cat((study_image_features, study_view_features), dim=1)
            weights = self.attention(attention_input).squeeze(1).softmax(dim=0)
            study_features.append((weights[:, None] * study_image_features).sum(dim=0))
        pooled = torch.stack(study_features)
        main_logits = self.main_head(pooled)
        expert_logits = torch.cat([head(pooled) for head in self.expert_heads], dim=1)
        return main_logits, expert_logits

    def forward(self, images, offsets, view_ids):
        return self.forward_from_features(self._encode_images(images), offsets, view_ids)
