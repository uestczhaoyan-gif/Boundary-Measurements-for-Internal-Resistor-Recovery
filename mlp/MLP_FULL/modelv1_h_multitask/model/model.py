import torch
import torch.nn as nn


class HMultiTaskMLP(nn.Module):
    """Shared trunk + dual heads (count via CORAL, delta regression)."""

    def __init__(self, in_dim=896, out_dim=112, dropout=0.1, max_abs=300.0):
        super().__init__()
        self.max_abs = max_abs

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.cls_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

        self.reg_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        feat = self.trunk(x)
        logits = self.cls_head(feat)
        delta = torch.tanh(self.reg_head(feat)) * self.max_abs
        return logits, delta

