import torch
import torch.nn as nn


class MLPReg2Prob(nn.Module):
    def __init__(self, in_dim=896, out_dim=112, dropout=0.1, max_abs=300.0):
        super().__init__()
        self.max_abs = max_abs
        self.backbone = nn.Sequential(
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
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(256, out_dim)

    def forward(self, x):
        y = self.out(self.backbone(x))
        return torch.tanh(y) * self.max_abs
