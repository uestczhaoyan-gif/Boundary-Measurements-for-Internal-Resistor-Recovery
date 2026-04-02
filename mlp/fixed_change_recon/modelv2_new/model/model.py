import torch
import torch.nn as nn


class Change3Regressor(nn.Module):
    def __init__(self, in_dim=896, out_dim=112, dropout=0.1, max_abs=310.0):
        super().__init__()
        self.max_abs = max_abs
        self.block1 = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Linear(1024, 896),
            nn.BatchNorm1d(896),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Linear(896, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.block4 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(256, out_dim)

    def forward(self, x):
        h1 = self.block1(x)
        h2 = self.block2(h1)
        h2 = h2 + x
        h3 = self.block3(h2)
        h4 = self.block4(h3)
        y = self.out(h4)
        return torch.tanh(y) * self.max_abs

