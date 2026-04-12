from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
VENDOR_DIR = WORKSPACE_ROOT / ".vendor_torchpy311"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import prepend_vendor_dir

prepend_vendor_dir(VENDOR_DIR, required_version=(3, 11))

import torch
import torch.nn as nn

from models.mlp_common import DeepResidualMLP


class Modelo1MLP2Regressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_resistors: int,
        hidden_dim: int = 1536,
        num_blocks: int = 8,
        ff_multiplier: float = 2.0,
        dropout: float = 0.02,
        max_abs: float = 250.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_resistors = num_resistors
        self.max_abs = max_abs

        self.backbone = DeepResidualMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            ff_multiplier=ff_multiplier,
            dropout=dropout,
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_resistors),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.size(0), -1)
        hidden = self.backbone(flat)
        return torch.tanh(self.value_head(hidden)) * self.max_abs

