import torch
import torch.nn as nn


def build_adj(grid=8):
    n = grid * grid
    a = torch.zeros(n, n, dtype=torch.float32)
    for r in range(grid):
        for c in range(grid):
            u = r * grid + c
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < grid and 0 <= cc < grid:
                    v = rr * grid + cc
                    a[u, v] = 1.0
    a = a + torch.eye(n)
    deg = a.sum(dim=1).clamp(min=1.0)
    d_inv = torch.diag(1.0 / deg)
    return d_inv @ a


class GraphLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w_self = nn.Linear(in_dim, out_dim)
        self.w_nei = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, h, a_norm):
        # h: [B, N, F], a_norm: [N, N]
        h_nei = torch.einsum("ij,bjf->bif", a_norm, h)
        out = self.w_self(h) + self.w_nei(h_nei)
        return self.act(out)


class CNN2DClassifier(nn.Module):
    """GNN baseline on node tokens (from 8x8 grid features)."""

    def __init__(self, in_ch=97, out_dim=3, dropout=0.1, hidden=128, depth=5):
        super().__init__()
        self.in_proj = nn.Linear(in_ch, hidden)
        layers = []
        for _ in range(depth):
            layers.append(GraphLayer(hidden, hidden))
        self.layers = nn.ModuleList(layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )
        self.register_buffer("a_norm", build_adj(8))

    def forward(self, x):
        # x: [B, C, 8, 8] -> nodes [B, 64, C]
        b = x.size(0)
        h = x.permute(0, 2, 3, 1).reshape(b, 64, x.size(1))
        h = self.in_proj(h)
        for gnn in self.layers:
            h = gnn(h, self.a_norm)
            h = self.drop(h)
        g = h.mean(dim=1)
        return self.head(g)

