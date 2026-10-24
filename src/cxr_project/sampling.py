"""Training-only, bounded Study sampling without duplicate-image counts."""
from __future__ import annotations
import math
import torch
from torch.utils.data import WeightedRandomSampler

def study_sampling_weights(studies, power=.5, max_weight=3.):
    """Use maximum rare-positive boost; all-negative Studies retain weight one."""
    if not 0 <= float(power) <= 1 or not math.isfinite(float(max_weight)) or max_weight < 1:
        raise ValueError('sampling_power must be in [0,1] and sampling_max_weight >= 1')
    if not studies or len({s.study_id for s in studies}) != len(studies):
        raise ValueError('sampling requires nonempty unique Study records')
    if any(s.labels is None for s in studies):
        raise ValueError('sampling requires training labels')
    targets = torch.tensor([s.labels for s in studies], dtype=torch.float64)
    positive = (targets == 1).sum(0)
    known = (targets >= 0).sum(0)
    boost = (known / positive.clamp(min=1)).pow(float(power)).clamp(min=1., max=float(max_weight))
    return torch.where(targets == 1, boost, 1.).amax(1)

def build_train_sampler(studies, config, seed):
    strategy = str(config.get('sampling_strategy', 'shuffle')).lower()
    if strategy == 'shuffle':
        return None
    if strategy != 'weighted':
        raise ValueError('sampling_strategy must be shuffle or weighted')
    if str(config.get('pos_weight_mode', 'ratio')).lower() != 'unit':
        raise ValueError('weighted sampling requires pos_weight_mode: unit to avoid double rebalancing')
    weights = study_sampling_weights(studies, config.get('sampling_power', .5), config.get('sampling_max_weight', 3.))
    generator = torch.Generator().manual_seed(int(seed))
    return WeightedRandomSampler(weights, len(studies), replacement=True, generator=generator)
