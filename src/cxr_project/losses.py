"""E0 masked multilabel loss with auxiliary small-data expert losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .labels import HARD_LABEL_INDICES, NUM_LABELS


def combine_main_and_expert_logits(main_logits, expert_logits, hard_label_indices=HARD_LABEL_INDICES):
    if main_logits.ndim != 2 or main_logits.shape[1] != NUM_LABELS:
        raise ValueError(f"main_logits must have shape [studies, {NUM_LABELS}]")
    hard_label_indices = tuple(hard_label_indices)
    if expert_logits.shape != (main_logits.shape[0], len(hard_label_indices)):
        raise ValueError("expert_logits shape must match the configured expert labels")
    combined = main_logits.clone()
    if hard_label_indices:
        indices = torch.as_tensor(hard_label_indices, device=main_logits.device, dtype=torch.long)
        combined[:, indices] = (main_logits[:, indices] + expert_logits) / 2
    return combined


def _masked_weighted_bce(logits, targets, pos_weights):
    if logits.shape != targets.shape:
        raise ValueError(f"logits and targets must have the same shape, got {logits.shape} and {targets.shape}")
    valid = targets >= 0
    if not valid.any():
        return logits.sum() * 0.0
    per_label = F.binary_cross_entropy_with_logits(
        logits,
        targets.clamp(min=0),
        pos_weight=pos_weights.to(device=logits.device, dtype=logits.dtype),
        reduction="none",
    )
    return (per_label * valid).sum() / valid.sum()


def compute_training_loss(
    main_logits,
    expert_logits,
    targets,
    pos_weights,
    hard_label_indices=HARD_LABEL_INDICES,
    expert_loss_weight=0.5,
):
    """Return the combined prediction loss and combined per-study logits.

    E0 ignores explicit -1 uncertain targets. Blank values are converted to
    zero by the manifest reader, matching the selected G1 blank policy.
    """
    if targets.shape != main_logits.shape or targets.shape[1] != NUM_LABELS:
        raise ValueError(f"targets must have shape [studies, {NUM_LABELS}]")
    if len(pos_weights) != NUM_LABELS:
        raise ValueError(f"pos_weights must contain {NUM_LABELS} values")
    if expert_loss_weight < 0:
        raise ValueError("expert_loss_weight must be non-negative")
    hard_label_indices = tuple(int(index) for index in hard_label_indices)
    combined = combine_main_and_expert_logits(main_logits, expert_logits, hard_label_indices)
    weights = torch.as_tensor(pos_weights, device=main_logits.device, dtype=main_logits.dtype)
    primary_loss = _masked_weighted_bce(combined, targets, weights)
    if hard_label_indices:
        index_tensor = torch.as_tensor(hard_label_indices, device=targets.device, dtype=torch.long)
        expert_targets = targets.index_select(1, index_tensor)
        expert_weights = weights.index_select(0, index_tensor)
        expert_loss = _masked_weighted_bce(expert_logits, expert_targets, expert_weights)
    else:
        expert_loss = main_logits.sum() * 0.0
    return primary_loss + expert_loss_weight * expert_loss, combined
