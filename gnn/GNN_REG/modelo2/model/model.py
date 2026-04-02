import sys
from pathlib import Path

_VENDOR_DIR = Path(__file__).resolve().parents[4] / ".vendor_torchpy311"
if _VENDOR_DIR.exists() and str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


GRID = 8
NUM_NODES = GRID * GRID


def build_message_edges(grid=GRID):
    edges = []
    for r in range(grid):
        for c in range(grid):
            src = r * grid + c
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < grid and 0 <= cc < grid:
                    dst = rr * grid + cc
                    edges.append((src, dst))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_resistor_edges(grid=GRID):
    edges = []
    for r in range(grid):
        for c in range(grid - 1):
            edges.append((r * grid + c, r * grid + c + 1))
        if r < grid - 1:
            for c in range(grid):
                edges.append((r * grid + c, (r + 1) * grid + c))
    return edges


class GATv2ResidualLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        self.conv = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim,
            heads=heads,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, h, edge_index):
        out = self.conv(h, edge_index)
        out = self.dropout(out)
        out = out + self.res_proj(h)
        return self.norm(F.elu(out))


def repeat_edge_index(base_edge_index, num_graphs, num_nodes):
    offsets = torch.arange(num_graphs, device=base_edge_index.device, dtype=base_edge_index.dtype)
    offsets = offsets.view(-1, 1, 1) * num_nodes
    batched = base_edge_index.unsqueeze(0) + offsets
    return batched.permute(1, 0, 2).reshape(2, -1)


class CrossExcitationAttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        # h: [B, 32, 64, H]
        logits = self.score(h).squeeze(-1)  # [B, 32, 64]
        attn = torch.softmax(logits, dim=1)
        pooled = torch.sum(attn.unsqueeze(-1) * h, dim=1)
        return pooled, attn


class PhysicsInformedGNNRegressor(nn.Module):
    def __init__(
        self,
        in_dim=4,
        hidden_dim=128,
        edge_hidden=128,
        out_dim=112,
        heads=4,
        excitation_chunk_size=4,
        dropout=0.1,
        max_abs=300.0,
    ):
        super().__init__()
        self.excitation_chunk_size = excitation_chunk_size
        self.max_abs = max_abs
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                GATv2ResidualLayer(hidden_dim, hidden_dim, heads=heads, dropout=dropout),
                GATv2ResidualLayer(hidden_dim, hidden_dim, heads=heads, dropout=dropout),
                GATv2ResidualLayer(hidden_dim, hidden_dim, heads=heads, dropout=dropout),
            ]
        )
        self.cross_pool = CrossExcitationAttentionPool(hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, edge_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden, edge_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mask_head = nn.Linear(edge_hidden, 1)
        self.value_head = nn.Linear(edge_hidden, 1)

        resistor_edges = build_resistor_edges()
        self.register_buffer("base_edge_index", build_message_edges())
        self.register_buffer("edge_u", torch.tensor([u for u, _ in resistor_edges], dtype=torch.long))
        self.register_buffer("edge_v", torch.tensor([v for _, v in resistor_edges], dtype=torch.long))

    def encode_excitations(self, x):
        bsz, excitations, nodes, feat_dim = x.shape
        chunk_size = excitations if self.excitation_chunk_size <= 0 else min(self.excitation_chunk_size, excitations)
        encoded_chunks = []

        for start in range(0, excitations, chunk_size):
            end = min(excitations, start + chunk_size)
            x_chunk = x[:, start:end]
            num_graphs = bsz * (end - start)
            edge_index = repeat_edge_index(self.base_edge_index, num_graphs, nodes)
            h = x_chunk.reshape(num_graphs * nodes, feat_dim)
            h = self.input_proj(h)
            for layer in self.layers:
                h = layer(h, edge_index)
            encoded_chunks.append(h.reshape(bsz, end - start, nodes, -1))

        return torch.cat(encoded_chunks, dim=1)

    def forward(self, x, return_aux=False):
        h = self.encode_excitations(x)
        pooled_nodes, excitation_attn = self.cross_pool(h)

        hu = pooled_nodes[:, self.edge_u, :]
        hv = pooled_nodes[:, self.edge_v, :]
        edge_feat = torch.cat([hu, hv, torch.abs(hu - hv)], dim=-1)
        edge_hidden = self.edge_mlp(edge_feat)
        mask_prob = torch.sigmoid(self.mask_head(edge_hidden).squeeze(-1))
        value = torch.tanh(self.value_head(edge_hidden).squeeze(-1)) * self.max_abs
        pred = mask_prob * value

        if return_aux:
            return pred, {
                "mask_prob": mask_prob,
                "value": value,
                "pooled_nodes": pooled_nodes,
                "excitation_attn": excitation_attn,
            }
        return pred
