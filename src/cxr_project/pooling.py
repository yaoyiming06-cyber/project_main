"""Pure-image, label-specific pooling within a single variable-size Study."""
from __future__ import annotations

import torch
from torch import nn


class LabelAttentionPooling(nn.Module):
    """Return [labels, channels] from global or per-label image features."""

    def __init__(self, feature_dim, num_labels=10, hidden_dim=128):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_labels = int(num_labels)
        self.norm = nn.LayerNorm(self.feature_dim)
        self.projection = nn.Linear(self.feature_dim, int(hidden_dim))
        self.score = nn.Linear(int(hidden_dim), self.num_labels)

    def forward(self, features):
        if (features.ndim not in (2, 3) or features.shape[0] < 1
                or features.shape[-1] != self.feature_dim
                or (features.ndim == 3 and features.shape[1] != self.num_labels)):
            raise ValueError('label attention expects nonempty [images, channels] or [images, labels, channels]')
        hidden = torch.tanh(self.projection(self.norm(features)))
        if features.ndim == 2:
            scores = self.score(hidden)
            values = features[:, None, :]
        else:
            # Each label scores only its matching image representation.
            scores = torch.einsum('nlh,lh->nl', hidden, self.score.weight) + self.score.bias
            values = features
        # Normalize only over images inside this Study, never across Studies.
        weights = scores.float().softmax(dim=0).to(features.dtype)
        return (weights[..., None] * values).sum(dim=0)
