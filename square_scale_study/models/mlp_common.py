from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h


class DeepResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        ff_multiplier: float,
        dropout: float,
    ):
        super().__init__()
        ff_dim = int(math.ceil(hidden_dim * ff_multiplier))
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, ff_dim=ff_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.final_norm(h)
