import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# MHA
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.n_heads, self.head_dim
        )  # [B, N, 3, H, d]
        q, k, v = qkv.unbind(dim=2)

        #
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # [B, H, N, d]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, N, N]
        attn = scores.softmax(dim=-1)
        out = attn @ v  # [B, H, N, d]
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


# Transformer Encoder Layer
class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# transformer encoder
class TransformerEncoder(nn.Module):
    def __init__(self, dim=256, depth=6, n_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, n_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return self.norm(x)  # 最后归一化
