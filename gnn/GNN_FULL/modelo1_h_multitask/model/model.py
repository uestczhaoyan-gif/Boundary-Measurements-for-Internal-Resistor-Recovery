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


def build_resistor_edges(grid=8):
    edges = []
    for r in range(grid):
        for c in range(grid - 1):
            edges.append((r * grid + c, r * grid + c + 1))
        if r < grid - 1:
            for c in range(grid):
                edges.append((r * grid + c, (r + 1) * grid + c))
    return edges


class GraphLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w_self = nn.Linear(in_dim, out_dim)
        self.w_nei = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, h, a_norm):
        h_nei = torch.einsum("ij,bjf->bif", a_norm, h)
        out = self.w_self(h) + self.w_nei(h_nei)
        return self.act(out)


class CNN2DHMultiTask(nn.Module):
    """GNN multitask baseline: graph trunk + count head + edge reg head."""

    def __init__(self, in_ch=97, out_dim=112, dropout=0.1, max_abs=300.0, hidden=128, depth=5):
        super().__init__()
        self.max_abs = max_abs
        self.in_proj = nn.Linear(in_ch, hidden)
        self.layers = nn.ModuleList([GraphLayer(hidden, hidden) for _ in range(depth)])
        self.drop = nn.Dropout(dropout)

        self.cls_head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self.register_buffer("a_norm", build_adj(8))
        edges = build_resistor_edges(8)
        eu = torch.tensor([e[0] for e in edges], dtype=torch.long)
        ev = torch.tensor([e[1] for e in edges], dtype=torch.long)
        self.register_buffer("edge_u", eu)
        self.register_buffer("edge_v", ev)

    def forward(self, x):
        b = x.size(0)
        h = x.permute(0, 2, 3, 1).reshape(b, 64, x.size(1))
        h = self.in_proj(h)
        for gnn in self.layers:
            h = gnn(h, self.a_norm)
            h = self.drop(h)
        g = h.mean(dim=1)
        logits = self.cls_head(g)
        hu = h[:, self.edge_u, :]
        hv = h[:, self.edge_v, :]
        z = torch.cat([hu, hv], dim=-1)
        delta = torch.tanh(self.edge_mlp(z).squeeze(-1)) * self.max_abs
        return logits, delta

