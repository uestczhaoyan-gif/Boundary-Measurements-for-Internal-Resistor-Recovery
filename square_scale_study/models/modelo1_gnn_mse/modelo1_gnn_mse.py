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

from models.modelo1_gnn.modelo1_gnn import (
    ExcitationAttentionPool,
    NodeEncoder,
    ResidualGraphBlock,
    repeat_edge_index,
)
from project_common import SquareTopologySpec


class Modelo1GNNMSERegressor(nn.Module):
    def __init__(
        self,
        topology: SquareTopologySpec,
        in_dim: int = 4,
        hidden_dim: int = 256,
        edge_hidden: int = 512,
        heads: int = 8,
        num_layers: int = 4,
        excitation_chunk_size: int = 8,
        dropout: float = 0.02,
        max_abs: float = 250.0,
    ):
        super().__init__()
        self.topology = topology
        self.num_nodes = topology.num_nodes
        self.num_resistors = topology.num_resistors
        self.excitation_chunk_size = excitation_chunk_size
        self.max_abs = max_abs

        self.node_encoder = NodeEncoder(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.layers = nn.ModuleList(
            [ResidualGraphBlock(hidden_dim=hidden_dim, heads=heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.jump_proj = nn.Sequential(
            nn.Linear(hidden_dim * (num_layers + 1), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.pool = ExcitationAttentionPool(hidden_dim=hidden_dim, dropout=dropout)

        fused_dim = hidden_dim * 3
        edge_feat_dim = fused_dim * 4
        self.edge_trunk = nn.Sequential(
            nn.Linear(edge_feat_dim, edge_hidden * 2),
            nn.LayerNorm(edge_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden * 2, edge_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden, edge_hidden),
            nn.GELU(),
        )
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
            h = self.node_encoder(h)
            features = [h]
            for layer in self.layers:
                h = layer(h, edge_index)
                features.append(h)
            h = torch.cat(features, dim=-1)
            h = self.jump_proj(h)
            encoded_chunks.append(h.reshape(bsz, end - start, nodes, -1))

        return torch.cat(encoded_chunks, dim=1)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        h = self.encode_excitations(x)
        pooled_nodes, aux = self.pool(h)
        hu = pooled_nodes[:, self.edge_u, :]
        hv = pooled_nodes[:, self.edge_v, :]
        edge_feat = torch.cat([hu, hv, torch.abs(hu - hv), hu * hv], dim=-1)
        edge_state = self.edge_trunk(edge_feat)
        value_pred = torch.tanh(self.value_head(edge_state).squeeze(-1)) * self.max_abs
        if return_aux:
            aux = dict(aux)
            aux["pooled_nodes"] = pooled_nodes
            return value_pred, aux
        return value_pred

