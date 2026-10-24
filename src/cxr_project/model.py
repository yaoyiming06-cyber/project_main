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
        head_dropout=0.2,
        expert_dropout=0.2,
        experts_enabled=True,
        study_pooling="attention",
        use_view_embedding=True,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hard_label_indices = tuple(int(index) for index in hard_label_indices) if experts_enabled else ()
        self.study_pooling = str(study_pooling).lower()
        self.use_view_embedding = bool(use_view_embedding)
        if self.study_pooling not in {"attention", "mean"}:
            raise ValueError("study_pooling must be attention or mean")
        if not 0 <= float(head_dropout) < 1 or not 0 <= float(expert_dropout) < 1:
            raise ValueError("head_dropout and expert_dropout must be in [0, 1)")
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

        self.view_embedding = (nn.Embedding(len(VIEW_NAMES), 32)
                               if self.use_view_embedding and self.study_pooling == "attention" else None)
        attention_dim = self.feature_dim + (32 if self.view_embedding is not None else 0)
        self.attention = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Linear(attention_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        ) if self.study_pooling == "attention" else None
        self.main_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(float(head_dropout)),
            nn.Linear(self.feature_dim, NUM_LABELS),
        )
        self.expert_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, 256),
                nn.GELU(),
                nn.Dropout(float(expert_dropout)),
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

    def encode_studies(self, image_features, offsets, view_ids):
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
            if self.study_pooling == "mean":
                study_features.append(study_image_features.mean(dim=0))
                continue
            attention_input = study_image_features
            if self.view_embedding is not None:
                study_view_features = self.view_embedding(view_ids[start:end])
                attention_input = torch.cat((study_image_features, study_view_features), dim=1)
            weights = self.attention(attention_input).squeeze(1).softmax(dim=0)
            study_features.append((weights[:, None] * study_image_features).sum(dim=0))
        return torch.stack(study_features)

    def classify_features(self, pooled):
        main_logits = self.main_head(pooled)
        expert_logits = (torch.cat([head(pooled) for head in self.expert_heads], dim=1)
                         if self.expert_heads else pooled.new_empty((pooled.shape[0], 0)))
        return main_logits, expert_logits

    def forward_from_features(self, image_features, offsets, view_ids):
        return self.classify_features(self.encode_studies(image_features, offsets, view_ids))

    def forward(self, images, offsets, view_ids):
        return self.forward_from_features(self._encode_images(images), offsets, view_ids)


SUPPORTED_MODELS = {
    MODEL_NAME: 'ImageNet-22K / Meta ConvNeXt',
    'convnext_small.fb_in22k': 'ImageNet-22K / Meta ConvNeXt',
    'swin_base_patch4_window7_224.ms_in22k': 'ImageNet-22K / Microsoft Swin',
    'vit_base_patch16_224.augreg_in21k': 'ImageNet-21K / Google AugReg ViT',
}


def model_options(config):
    return {key: config.get(key, default) for key, default in {
        'drop_path': .1, 'hard_label_indices': HARD_LABEL_INDICES,
        'gradient_checkpointing': True, 'image_chunk_size': 2,
        'head_dropout': .2, 'expert_dropout': .2, 'experts_enabled': True,
        'study_pooling': 'attention', 'use_view_embedding': True,
    }.items()}


def configure_backbone_training(model, freeze_backbone=False, unfreeze_last_stages=None, adaptformer=False):
    """Freeze only the backbone; Study pooling and classification heads stay trainable."""
    backbone = model.backbone
    if freeze_backbone and unfreeze_last_stages is not None:
        raise ValueError('freeze_backbone and unfreeze_last_stages are mutually exclusive')
    if adaptformer and unfreeze_last_stages is not None:
        raise ValueError('AdaptFormer and partial unfreezing are separate experiments')
    for parameter in backbone.parameters():
        parameter.requires_grad_(not (freeze_backbone or adaptformer or unfreeze_last_stages is not None))
    if unfreeze_last_stages is not None:
        stages = getattr(backbone, 'stages', None)
        if stages is None:
            stages = getattr(backbone, 'layers', None)
        if stages is None:
            raise ValueError('unfreeze_last_stages requires a staged ConvNeXt or Swin backbone')
        count = int(unfreeze_last_stages)
        if not 1 <= count <= len(stages):
            raise ValueError(f'unfreeze_last_stages must be between 1 and {len(stages)}')
        for stage in list(stages)[-count:]:
            for parameter in stage.parameters():
                parameter.requires_grad_(True)
        for name in ('norm_pre', 'norm', 'head'):
            module = getattr(backbone, name, None)
            if isinstance(module, nn.Module):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
    if adaptformer:
        from .adapters import BottleneckAdapter
        adapters = [module for module in backbone.modules() if isinstance(module, BottleneckAdapter)]
        if not adapters:
            raise ValueError('adaptformer requested but backbone has no installed adapters')
        for adapter in adapters:
            for parameter in adapter.parameters():
                parameter.requires_grad_(True)
    return model


def build_model(config, pretrained_checkpoint=None, pretrained=None):
    """Build an explicitly named general-pretrained backbone, with no model fallback.

    New backbones require local safetensors and an expected SHA256 when pretrained.
    The existing Base checkpoint retains its built-in pinned hash. No implicit
    network downloads occur here. ViT adapters are installed after weight loading.
    """
    name = str(config.get('model', MODEL_NAME))
    if name not in SUPPORTED_MODELS:
        raise ValueError(f'Unsupported model {name!r}; choose one of {list(SUPPORTED_MODELS)}')
    adaptformer = bool(config.get('adaptformer', False))
    if adaptformer and not name.startswith('vit_'):
        raise ValueError('adaptformer currently supports vit_base_patch16_224.augreg_in21k only')
    pretrained = bool(config.get('pretrained', True)) if pretrained is None else bool(pretrained)
    path = pretrained_checkpoint if pretrained_checkpoint is not None else config.get('pretrained_checkpoint')
    options = model_options(config)
    if name == MODEL_NAME:
        model = ConvNeXtStudyClassifier(pretrained=pretrained, pretrained_checkpoint=path, **options)
    else:
        import timm
        kwargs = {'num_classes': 0, 'drop_path_rate': float(config.get('drop_path', .1))}
        if name.startswith('swin_'):
            kwargs['img_size'] = int(config.get('input_size', 224))
        elif name.startswith('vit_'):
            kwargs['dynamic_img_size'] = True
        if pretrained:
            checkpoint_path = Path(path) if path else None
            if checkpoint_path is None or not checkpoint_path.is_file():
                raise FileNotFoundError(f'Local general pretrained weights not found: {checkpoint_path}')
            digest = str(config.get('pretrained_sha256') or '').lower()
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('pretrained_sha256 is required for new pretrained backbones')
            if file_sha256(checkpoint_path) != digest:
                raise ValueError('pretrained checkpoint SHA-256 mismatch')
            if checkpoint_path.suffix != '.safetensors':
                raise ValueError('new backbone weights must be timm Hugging Face .safetensors files')
            kwargs['pretrained_cfg_overlay'] = {'file': str(checkpoint_path), 'custom_load': False}
        backbone = timm.create_model(name, pretrained=pretrained, **kwargs)
        if adaptformer:
            from .adapters import install_adaptformer
            install_adaptformer(backbone, int(config.get('adapter_dim', 64)),
                               float(config.get('adapter_dropout', .1)),
                               float(config.get('adapter_scale', 1.)))
        model = ConvNeXtStudyClassifier(backbone=backbone, feature_dim=backbone.num_features,
                                       pretrained=False, **options)
    configure_backbone_training(model, bool(config.get('freeze_backbone', False)),
                                config.get('unfreeze_last_stages'), adaptformer)
    return model
