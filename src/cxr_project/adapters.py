"""AdaptFormer-style zero-initialized parallel FFN adapters for timm ViT."""
import math
from torch import nn

class BottleneckAdapter(nn.Module):
    def __init__(self, dim, bottleneck=64, dropout=.1, scale=1.):
        super().__init__()
        if int(bottleneck) < 1 or not 0 <= float(dropout) < 1:
            raise ValueError('adapter_dim must be positive and adapter_dropout in [0, 1)')
        if not math.isfinite(float(scale)) or float(scale) <= 0:
            raise ValueError('adapter_scale must be finite and positive')
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, int(bottleneck))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout))
        self.up = nn.Linear(int(bottleneck), dim)
        self.scale = float(scale)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.dropout(self.activation(self.down(self.norm(x))))) * self.scale

class AdaptFormerMLP(nn.Module):
    def __init__(self, base_mlp, dim, bottleneck=64, dropout=.1, scale=1.):
        super().__init__()
        self.base_mlp = base_mlp
        self.adapter = BottleneckAdapter(dim, bottleneck, dropout, scale)

    def forward(self, x):
        return self.base_mlp(x) + self.adapter(x)

def install_adaptformer(backbone, bottleneck=64, dropout=.1, scale=1.):
    if not hasattr(backbone, 'blocks') or not hasattr(backbone, 'embed_dim'):
        raise ValueError('AdaptFormer currently requires a timm VisionTransformer backbone')
    for block in backbone.blocks:
        if not hasattr(block, 'mlp') or isinstance(block.mlp, AdaptFormerMLP):
            raise ValueError('Unsupported or already adapted Transformer block')
        block.mlp = AdaptFormerMLP(block.mlp, backbone.embed_dim, bottleneck, dropout, scale)
    return backbone
