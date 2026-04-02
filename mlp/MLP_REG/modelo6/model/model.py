import torch
import torch.nn as nn


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
        # x: [B, 32, H]
        y = x.transpose(1, 2)
        y = self.token_mlp(y)
        x = x + y.transpose(1, 2)
        return x + self.channel_mlp(x)


class MLPMixerRegressor(nn.Module):
    def __init__(self, num_excitations=32, in_dim=28, hidden_dim=128, trunk_dim=256, out_dim=112, dropout=0.1, max_abs=300.0):
        super().__init__()
        self.max_abs = max_abs
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
        self.mask_head = nn.Linear(trunk_dim, out_dim)
        self.value_head = nn.Linear(trunk_dim, out_dim)

    def forward(self, x, return_aux=False):
        feat = self.spatial_encoder(x)
        feat = self.mixer(feat)
        trunk_feat = self.trunk(feat.reshape(feat.size(0), -1))

        mask_prob = torch.sigmoid(self.mask_head(trunk_feat))
        value = torch.tanh(self.value_head(trunk_feat)) * self.max_abs
        pred = mask_prob * value

        if return_aux:
            return pred, {"mask_prob": mask_prob, "value": value, "features": trunk_feat}
        return pred
