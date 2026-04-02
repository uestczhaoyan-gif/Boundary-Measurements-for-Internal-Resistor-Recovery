import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedExcitationMLP(nn.Module):
    def __init__(self, in_dim=28, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        bsz, excitations, feat_dim = x.shape
        y = self.net(x.reshape(bsz * excitations, feat_dim))
        return y.reshape(bsz, excitations, -1)


class ExcitationMixer(nn.Module):
    def __init__(self, num_excitations=32, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(num_excitations),
            nn.Linear(num_excitations, num_excitations * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_excitations * 2, num_excitations),
            nn.Dropout(dropout),
        )
        self.channel_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        y = x.transpose(1, 2)
        y = self.token_mlp(y)
        x = x + y.transpose(1, 2)
        return x + self.channel_mlp(x)


class MLPMixerClassifier(nn.Module):
    def __init__(
        self,
        num_excitations=32,
        in_dim=28,
        hidden_dim=128,
        trunk_dim=256,
        proj_dim=128,
        aux_dim=128,
        out_dim=3,
        dropout=0.1,
    ):
        super().__init__()
        self.spatial_encoder = SharedExcitationMLP(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.mixer = ExcitationMixer(num_excitations=num_excitations, hidden_dim=hidden_dim, dropout=dropout)
        self.trunk = nn.Sequential(
            nn.LayerNorm(num_excitations * hidden_dim),
            nn.Linear(num_excitations * hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, trunk_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.contrast_proj = nn.Sequential(
            nn.Linear(trunk_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )
        self.coral_head = nn.Linear(proj_dim, out_dim)
        self.aux_head = nn.Sequential(
            nn.LayerNorm(trunk_dim),
            nn.Linear(trunk_dim, aux_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(aux_dim, 1),
        )

    def forward(self, x, return_aux=False):
        feat = self.spatial_encoder(x)
        feat = self.mixer(feat)
        trunk_feat = self.trunk(feat.reshape(feat.size(0), -1))
        contrast_feat = self.contrast_proj(trunk_feat)
        logits = self.coral_head(contrast_feat)
        aux_logit = self.aux_head(trunk_feat).squeeze(-1)
        if return_aux:
            return logits, {
                "contrast_feat": F.normalize(contrast_feat, dim=-1),
                "features": trunk_feat,
                "aux_logit": aux_logit,
            }
        return logits
