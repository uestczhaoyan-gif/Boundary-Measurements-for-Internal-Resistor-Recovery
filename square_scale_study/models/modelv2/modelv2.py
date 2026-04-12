from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
VENDOR_DIR = WORKSPACE_ROOT / ".vendor_torchpy311"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import prepend_vendor_dir

prepend_vendor_dir(VENDOR_DIR, required_version=(3, 11))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from project_common import SquareTopologySpec


class GATv2ResidualLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
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

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        out = self.conv(h, edge_index)
        out = self.dropout(out)
        out = out + self.res_proj(h)
        return self.norm(F.elu(out))


def repeat_edge_index(base_edge_index: torch.Tensor, num_graphs: int, num_nodes: int) -> torch.Tensor:
    offsets = torch.arange(num_graphs, device=base_edge_index.device, dtype=base_edge_index.dtype)
    offsets = offsets.view(-1, 1, 1) * num_nodes
    batched = base_edge_index.unsqueeze(0) + offsets
    return batched.permute(1, 0, 2).reshape(2, -1)


class CrossExcitationAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(h).squeeze(-1)
        attn = torch.softmax(logits, dim=1)
        pooled = torch.sum(attn.unsqueeze(-1) * h, dim=1)
        return pooled, attn


class ModelV2Regressor(nn.Module):
    def __init__(
        self,
        topology: SquareTopologySpec,
        in_dim: int = 4,
        hidden_dim: int = 128,
        edge_hidden: int = 128,
        heads: int = 4,
        excitation_chunk_size: int = 4,
        dropout: float = 0.1,
        max_abs: float = 250.0,
    ):
        super().__init__()
        self.topology = topology
        self.num_nodes = topology.num_nodes
        self.num_resistors = topology.num_resistors
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
        self.score_head = nn.Linear(edge_hidden, 1)
        self.value_head = nn.Linear(edge_hidden, 1)

        message_edge_index = torch.tensor(topology.message_edges, dtype=torch.long).t().contiguous()
        self.register_buffer("base_edge_index", message_edge_index)
        self.register_buffer("edge_u", torch.tensor([u for u, _ in topology.resistor_edges], dtype=torch.long))
        self.register_buffer("edge_v", torch.tensor([v for _, v in topology.resistor_edges], dtype=torch.long))

    def encode_excitations(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        h = self.encode_excitations(x)
        pooled_nodes, excitation_attn = self.cross_pool(h)
        hu = pooled_nodes[:, self.edge_u, :]
        hv = pooled_nodes[:, self.edge_v, :]
        edge_feat = torch.cat([hu, hv, torch.abs(hu - hv)], dim=-1)
        edge_hidden = self.edge_mlp(edge_feat)
        score_logits = self.score_head(edge_hidden).squeeze(-1)
        value_pred = torch.tanh(self.value_head(edge_hidden).squeeze(-1)) * self.max_abs
        if return_aux:
            return score_logits, value_pred, {"pooled_nodes": pooled_nodes, "excitation_attn": excitation_attn}
        return score_logits, value_pred
