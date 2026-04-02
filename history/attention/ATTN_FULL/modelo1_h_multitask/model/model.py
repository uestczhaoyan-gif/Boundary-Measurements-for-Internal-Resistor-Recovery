import torch
import torch.nn as nn


class CNN2DHMultiTask(nn.Module):
    """Attention multitask baseline: shared Transformer + cls/reg heads."""

    def __init__(
        self,
        in_ch=97,
        out_dim=112,
        dropout=0.1,
        max_abs=300.0,
        d_model=128,
        nhead=8,
        depth=4,
        ff=256,
    ):
        super().__init__()
        self.max_abs = max_abs
        self.token_proj = nn.Linear(in_ch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 64, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)

        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        b = x.size(0)
        tokens = x.permute(0, 2, 3, 1).reshape(b, 64, x.size(1))
        h = self.token_proj(tokens) + self.pos
        h = self.encoder(h)
        h = self.norm(h)
        g = h.mean(dim=1)
        logits = self.cls_head(g)
        delta = torch.tanh(self.reg_head(g)) * self.max_abs
        return logits, delta

