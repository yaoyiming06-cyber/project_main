"""Learning-rate schedule helpers that do not depend on the training runtime."""

import math


def scheduler_step_counts(micro_batches_per_epoch, accum_steps, epochs, warmup_epochs):
    """Return total and warmup optimizer-update counts after gradient accumulation."""
    if micro_batches_per_epoch < 1 or accum_steps < 1 or epochs < 1:
        raise ValueError("micro-batches, accumulation steps, and epochs must be positive")
    if warmup_epochs < 0 or warmup_epochs > epochs:
        raise ValueError("warmup_epochs must be between zero and epochs")
    updates_per_epoch = math.ceil(micro_batches_per_epoch / accum_steps)
    return updates_per_epoch * epochs, updates_per_epoch * warmup_epochs
