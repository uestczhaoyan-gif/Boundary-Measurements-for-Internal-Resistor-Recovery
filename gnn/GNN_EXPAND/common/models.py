from __future__ import annotations

import math
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
VENDOR_DIR = WORKSPACE_ROOT / ".vendor_torchpy311"
if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from topologies import TopologySpec


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

    def forward(self, h: torch.Tensor):
        logits = self.score(h).squeeze(-1)
        attn = torch.softmax(logits, dim=1)
        pooled = torch.sum(attn.unsqueeze(-1) * h, dim=1)
        return pooled, attn


class PhysicsInformedGNNClassifier(nn.Module):
    def __init__(
        self,
        topology: TopologySpec,
        in_dim: int = 4,
        hidden_dim: int = 128,
        proj_dim: int = 128,
        out_dim: int = 3,
        heads: int = 4,
        excitation_chunk_size: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.topology = topology
        self.num_nodes = topology.num_nodes
        self.excitation_chunk_size = excitation_chunk_size
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                GATv2ResidualLayer(hidden_dim, hidden_dim, heads=heads, dropout=dropout),
                GATv2ResidualLayer(hidden_dim, hidden_dim, heads=heads, dropout=dropout),
                GATv2ResidualLayer(hidden_dim, hidden_dim, heads=heads, dropout=dropout),
            ]
        )
        self.cross_pool = CrossExcitationAttentionPool(hidden_dim)
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.contrast_proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )
        self.coral_head = nn.Linear(proj_dim, out_dim)
        edge_index = torch.tensor(topology.message_edges, dtype=torch.long).t().contiguous()
        self.register_buffer("base_edge_index", edge_index)

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

        mean_feat = pooled_nodes.mean(dim=1)
        max_feat = pooled_nodes.max(dim=1).values
        graph_feat = self.graph_head(torch.cat([mean_feat, max_feat], dim=-1))
        contrast_feat = self.contrast_proj(graph_feat)
        logits = self.coral_head(contrast_feat)
        if return_aux:
            return logits, {
                "contrast_feat": F.normalize(contrast_feat, dim=-1),
                "graph_feat": graph_feat,
                "pooled_nodes": pooled_nodes,
                "excitation_attn": excitation_attn,
            }
        return logits


class PhysicsInformedGNNRegressor(nn.Module):
    def __init__(
        self,
        topology: TopologySpec,
        in_dim: int = 4,
        hidden_dim: int = 128,
        edge_hidden: int = 128,
        heads: int = 4,
        excitation_chunk_size: int = 4,
        dropout: float = 0.1,
        max_abs: float = 300.0,
        mask_init_prob: float = 0.35,
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
        self.mask_head = nn.Linear(edge_hidden, 1)
        self.value_head = nn.Linear(edge_hidden, 1)
        safe_prob = min(max(float(mask_init_prob), 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.mask_head.bias, math.log(safe_prob / (1.0 - safe_prob)))

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
        mask_logits = self.mask_head(edge_hidden).squeeze(-1)
        mask_prob = torch.sigmoid(mask_logits)
        value = torch.tanh(self.value_head(edge_hidden).squeeze(-1)) * self.max_abs
        pred = mask_prob * value

        if return_aux:
            return pred, {
                "mask_logits": mask_logits,
                "mask_prob": mask_prob,
                "value": value,
                "pooled_nodes": pooled_nodes,
                "excitation_attn": excitation_attn,
            }
        return pred
