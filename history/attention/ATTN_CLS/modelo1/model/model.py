import torch
import torch.nn as nn


class CNN2DClassifier(nn.Module):
    """Attention baseline: node-token Transformer for count classification."""

    def __init__(self, in_ch=97, out_dim=3, dropout=0.1, d_model=128, nhead=8, depth=4, ff=256):
        super().__init__()
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
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        # x: [B, 97, 8, 8] -> tokens [B, 64, 97]
        b = x.size(0)
        tokens = x.permute(0, 2, 3, 1).reshape(b, 64, x.size(1))
        h = self.token_proj(tokens) + self.pos
        h = self.encoder(h)
        h = self.norm(h)
        g = h.mean(dim=1)
        return self.head(g)

