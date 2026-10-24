"""Shared optimizer and inference state contracts."""
import torch


def optimizer_update(model, optimizer, scaler, clip_grad=None, ema=None):
    old_scale = scaler.get_scale() if scaler.is_enabled() else None
    if scaler.is_enabled():
        scaler.unscale_(optimizer)
    norm = None
    if clip_grad is not None:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
    scaler.step(optimizer)
    scaler.update()
    succeeded = old_scale is None or scaler.get_scale() >= old_scale
    if succeeded and ema is not None:
        ema.update(model)
    optimizer.zero_grad(set_to_none=True)
    return succeeded


def select_inference_state(checkpoint, selection="auto"):
    recorded = checkpoint.get("selected_weights", checkpoint.get("weight_source"))
    if selection == "auto" and recorded in {"raw", "ema"}:
        selection = recorded
        if selection == "ema" and checkpoint.get("ema") is None and "model" in checkpoint:
            return checkpoint["model"], checkpoint.get("weight_source", "ema")
    if selection not in {"auto", "raw", "ema"}:
        raise ValueError("weights must be auto, raw, or ema")
    # Old EMA checkpoints predate ema_active and started EMA immediately.
    has_ema = checkpoint.get("ema") is not None
    active = has_ema and bool(checkpoint.get("ema_active", True))
    if selection == "ema" and not active:
        raise ValueError("checkpoint has no active EMA weights")
    if selection == "ema" or (selection == "auto" and active):
        return checkpoint["ema"], "ema"
    if "model" not in checkpoint:
        raise ValueError("checkpoint is missing model weights")
    return checkpoint["model"], checkpoint.get("weight_source", "raw")

