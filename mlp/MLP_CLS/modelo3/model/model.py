import torch
import torch.nn as nn


class MLPClassifierMultiHead(nn.Module):
    """Main CORAL head + auxiliary 2-vs-3 head."""

    def __init__(self, in_dim=896, out_dim=3, dropout=0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.main_head = nn.Linear(128, out_dim)
        self.aux_23_head = nn.Linear(128, 1)

    def forward(self, x):
        feat = self.backbone(x)
        main_logits = self.main_head(feat)
        aux_23_logit = self.aux_23_head(feat).squeeze(-1)
        return main_logits, aux_23_logit

