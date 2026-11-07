"""Shared device, precision, optimizer, and inference state contracts."""
import contextlib
import torch


def mps_available():
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def select_device(requested="auto"):
    """Resolve auto to CUDA, then Apple MPS, then CPU."""
    requested = str(requested).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "mps":
        if not mps_available():
            raise RuntimeError("MPS was requested but is unavailable; use Apple Silicon and an MPS-enabled PyTorch build")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be auto, cuda, mps, or cpu")


def device_supports_amp(device, enabled=True):
    """Keep mixed precision on CUDA only; run MPS and CPU in FP32."""
    return bool(enabled) and torch.device(device).type == "cuda"


def autocast_context(device, enabled=True):
    if device_supports_amp(device, enabled):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def create_grad_scaler(device, enabled=True):
    """Create an active scaler only for CUDA AMP."""
    use_scaler = device_supports_amp(device, enabled)
    return torch.amp.GradScaler("cuda", enabled=use_scaler)


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

