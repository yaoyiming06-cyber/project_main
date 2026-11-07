"""Lightweight label queries over ConvNeXt spatial maps; no extra weights."""
from __future__ import annotations

import math

import torch
from torch import nn


class LabelwiseClassifier(nn.Module):
    """A distinct binary classifier for each [Study, label, feature] vector."""

    def __init__(self, feature_dim, num_labels=10, dropout=.2):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.empty(num_labels, feature_dim))
        self.bias = nn.Parameter(torch.zeros(num_labels))
        nn.init.uniform_(self.weight, -1 / math.sqrt(feature_dim), 1 / math.sqrt(feature_dim))

    def forward(self, features):
        if features.ndim != 3 or features.shape[1:] != self.weight.shape:
            raise ValueError('labelwise classifier expects [studies, labels, channels]')
        return (self.dropout(self.norm(features)) * self.weight).sum(-1) + self.bias


class _CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention and FFN; no label self-attention."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim * 4, dim), nn.Dropout(dropout))

    def forward(self, queries, memory):
        attended, _ = self.attention(self.query_norm(queries), memory, memory, need_weights=False)
        queries = queries + self.dropout(attended)
        return queries + self.ffn(self.ffn_norm(queries))


class SpatialLabelQuery(nn.Module):
    """Extract a label representation from each image's NCHW spatial map.

    This small query-based experiment is not a full reproduction of either
    Query2Label or ML-Decoder. Its head is initialized from scratch.
    """

    def __init__(self, feature_dim, query_dim=256, num_heads=4, num_layers=1,
                 dropout=.2, num_labels=10):
        super().__init__()
        if any(type(x) is not int or x < 1 for x in (feature_dim, query_dim, num_heads, num_layers, num_labels)):
            raise ValueError('query dimensions, heads and layers must be positive integers')
        if query_dim % num_heads or not math.isfinite(float(dropout)) or not 0 <= float(dropout) < 1:
            raise ValueError('query dimension must be divisible by heads and dropout must be in [0,1)')
        self.feature_dim = feature_dim
        self.memory = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, query_dim))
        self.queries = nn.Parameter(torch.empty(num_labels, query_dim))
        nn.init.normal_(self.queries, std=.02)
        self.blocks = nn.ModuleList([_CrossAttentionBlock(query_dim, num_heads, dropout)
                                    for _ in range(num_layers)])
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, feature_map):
        if (feature_map.ndim != 4 or feature_map.shape[1] != self.feature_dim
                or min(feature_map.shape[0], feature_map.shape[2], feature_map.shape[3]) < 1):
            raise ValueError('spatial query requires a nonempty NCHW spatial feature map, not pooled vectors')
        memory = self.memory(feature_map.flatten(2).transpose(1, 2))
        queries = self.queries[None].expand(feature_map.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, memory)
        return self.norm(queries)
