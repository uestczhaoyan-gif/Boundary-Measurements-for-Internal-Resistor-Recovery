import sys
from pathlib import Path

_VENDOR_DIR = Path(__file__).resolve().parents[4] / ".vendor_torchpy311"
if _VENDOR_DIR.exists() and str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F


GRID = 8
NUM_NODES = GRID * GRID
NUM_EXCITATIONS = 32


def build_resistor_edges(grid=GRID):
    edges = []
    for r in range(grid):
        for c in range(grid - 1):
            edges.append((r * grid + c, r * grid + c + 1))
        if r < grid - 1:
            for c in range(grid):
                edges.append((r * grid + c, (r + 1) * grid + c))
    return edges


def build_message_topology(grid=GRID):
    resistor_edges = build_resistor_edges(grid)
    directed_edges = []
    edge_ids = []
    for rid, (u, v) in enumerate(resistor_edges):
        directed_edges.append((u, v))
        edge_ids.append(rid)
        directed_edges.append((v, u))
        edge_ids.append(rid)
    edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    edge_id = torch.tensor(edge_ids, dtype=torch.long)
    return resistor_edges, edge_index, edge_id


def build_boundary_mask():
    mask = torch.zeros(NUM_NODES, 1, dtype=torch.float32)
    for r in range(GRID):
        for c in range(GRID):
            if r in (0, GRID - 1) or c in (0, GRID - 1):
                mask[r * GRID + c, 0] = 1.0
    return mask


def inverse_softplus(value):
    value_t = torch.as_tensor(value, dtype=torch.float32)
    return torch.log(torch.expm1(value_t))


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class EdgeDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mask_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, max_abs):
        h = self.backbone(x)
        mask_prob = torch.sigmoid(self.mask_head(h).squeeze(-1))
        value = torch.tanh(self.value_head(h).squeeze(-1)) * max_abs
        pred = mask_prob * value
        return pred, mask_prob, value


class PhysicalGNBlock(nn.Module):
    def __init__(self, global_dim, edge_hidden, node_hidden, dropout=0.1):
        super().__init__()
        self.edge_update = MLP(in_dim=6 + global_dim, hidden_dim=edge_hidden, out_dim=1, dropout=dropout)
        self.node_update = MLP(in_dim=2 + global_dim, hidden_dim=node_hidden, out_dim=1, dropout=dropout)
        self.global_update = MLP(in_dim=global_dim + 4, hidden_dim=edge_hidden, out_dim=global_dim, dropout=dropout)

    def forward(self, edge_feat, node_feat, global_feat):
        delta_edge = self.edge_update(torch.cat([edge_feat, global_feat], dim=-1)).squeeze(-1)
        delta_node = self.node_update(torch.cat([node_feat, global_feat], dim=-1)).squeeze(-1)
        delta_global = self.global_update(global_feat)
        return delta_edge, delta_node, delta_global


class PhysicalGNNRegressor(nn.Module):
    def __init__(
        self,
        num_excitations=NUM_EXCITATIONS,
        num_iters=6,
        node_hidden=64,
        edge_hidden=128,
        global_dim=16,
        dropout=0.1,
        max_abs=300.0,
        learn_alpha=True,
        alpha_init=0.1,
        alpha_max=0.25,
        conductance_init=1.0,
        edge_update_scale=0.15,
        voltage_update_scale=0.10,
        current_a=0.01,
    ):
        super().__init__()
        resistor_edges, edge_index, edge_id = build_message_topology()
        self.num_excitations = num_excitations
        self.num_iters = num_iters
        self.max_abs = max_abs
        self.learn_alpha = learn_alpha
        self.alpha_max = alpha_max
        self.edge_update_scale = edge_update_scale
        self.voltage_update_scale = voltage_update_scale

        self.register_buffer("base_edge_index", edge_index)
        self.register_buffer("base_edge_id", edge_id)
        self.register_buffer("edge_u", torch.tensor([u for u, _ in resistor_edges], dtype=torch.long))
        self.register_buffer("edge_v", torch.tensor([v for _, v in resistor_edges], dtype=torch.long))
        self.register_buffer("default_boundary_mask", build_boundary_mask())
        self.register_buffer("current_a_tensor", torch.tensor(float(current_a), dtype=torch.float32))

        init_raw = inverse_softplus(torch.full((len(resistor_edges),), conductance_init))
        self.base_conductance_raw = nn.Parameter(init_raw.clone())
        self.internal_init = nn.Parameter(torch.zeros(NUM_NODES, 1))

        if learn_alpha:
            alpha_ratio = min(max(alpha_init / alpha_max, 1e-4), 1 - 1e-4)
            self.alpha_raw = nn.Parameter(torch.logit(torch.tensor(alpha_ratio, dtype=torch.float32)))
        else:
            self.register_buffer("alpha_value", torch.tensor(float(alpha_init), dtype=torch.float32))

        self.global_init = nn.Sequential(
            nn.Linear(4, global_dim),
            nn.GELU(),
            nn.Linear(global_dim, global_dim),
        )
        self.block = PhysicalGNBlock(global_dim=global_dim, edge_hidden=edge_hidden, node_hidden=node_hidden, dropout=dropout)
        decoder_in = 12 + 2 + global_dim
        self.decoder = EdgeDecoder(decoder_in, edge_hidden, dropout=dropout)

    def conductance(self):
        return F.softplus(self.base_conductance_raw) + 1e-4

    def alpha(self):
        if self.learn_alpha:
            return self.alpha_max * torch.sigmoid(self.alpha_raw)
        return self.alpha_value

    def build_initial_global(self, boundary_input, boundary_mask):
        boundary_abs = (boundary_input * boundary_mask).abs()
        feat = torch.stack(
            [
                boundary_abs.mean(dim=(1, 2)),
                boundary_abs.amax(dim=(1, 2)),
                torch.full((boundary_input.size(0),), float(self.current_a_tensor), device=boundary_input.device),
                torch.full(
                    (boundary_input.size(0),),
                    float(self.current_a_tensor) * self.num_excitations,
                    device=boundary_input.device,
                ),
            ],
            dim=-1,
        )
        return self.global_init(feat)

    def build_initial_voltage(self, boundary_input, boundary_mask):
        internal = self.internal_init.view(1, NUM_NODES, 1)
        return boundary_mask * boundary_input + (1.0 - boundary_mask) * internal

    def aggregate_node_current(self, voltage, conductance):
        diff = voltage[:, self.edge_u, :] - voltage[:, self.edge_v, :]
        current = conductance.unsqueeze(-1) * diff
        bsz = voltage.size(0)
        agg = voltage.new_zeros(bsz, NUM_NODES, self.num_excitations)
        flat = agg.view(bsz * NUM_NODES, self.num_excitations)
        offsets = torch.arange(bsz, device=voltage.device).unsqueeze(1) * NUM_NODES
        flat_u = (self.edge_u.unsqueeze(0) + offsets).reshape(-1)
        flat_v = (self.edge_v.unsqueeze(0) + offsets).reshape(-1)
        flat_current = current.reshape(bsz * self.edge_u.numel(), self.num_excitations)
        flat.index_add_(0, flat_u, flat_current)
        flat.index_add_(0, flat_v, -flat_current)
        return agg, diff, current

    def edge_features(self, voltage, conductance):
        vu = voltage[:, self.edge_u, :]
        vv = voltage[:, self.edge_v, :]
        diff = vu - vv
        return torch.stack(
            [
                diff.abs().mean(dim=-1),
                diff.abs().amax(dim=-1),
                vu.mean(dim=-1),
                vv.mean(dim=-1),
                conductance,
                (vu * vv).mean(dim=-1),
            ],
            dim=-1,
        )

    def update_global(self, global_state, voltage, conductance, node_current):
        summary = torch.cat(
            [
                global_state,
                voltage.abs().mean(dim=(1, 2)).unsqueeze(-1),
                node_current.abs().mean(dim=(1, 2)).unsqueeze(-1),
                conductance.mean(dim=1).unsqueeze(-1),
                conductance.std(dim=1).unsqueeze(-1),
            ],
            dim=-1,
        )
        return global_state + self.block.global_update(summary)

    def physics_iterate(self, boundary_input, boundary_mask):
        bsz = boundary_input.size(0)
        global_state = self.build_initial_global(boundary_input, boundary_mask)
        voltage = self.build_initial_voltage(boundary_input, boundary_mask)
        edge_raw = self.base_conductance_raw.view(1, -1).expand(bsz, -1).clone()

        for _ in range(self.num_iters):
            conductance = F.softplus(edge_raw) + 1e-4
            edge_feat = self.edge_features(voltage, conductance)
            edge_global = global_state.unsqueeze(1).expand(-1, conductance.size(1), -1)
            delta_edge = self.block.edge_update(torch.cat([edge_feat, edge_global], dim=-1)).squeeze(-1)
            edge_raw = edge_raw + self.edge_update_scale * torch.tanh(delta_edge)
            conductance = F.softplus(edge_raw) + 1e-4

            node_current, _diff, _phys_current = self.aggregate_node_current(voltage, conductance)
            node_global = global_state.unsqueeze(1).unsqueeze(1).expand(-1, NUM_NODES, self.num_excitations, -1)
            node_feat = torch.cat([voltage.unsqueeze(-1), node_current.unsqueeze(-1), node_global], dim=-1)
            delta_voltage = self.block.node_update(node_feat).squeeze(-1)
            voltage = voltage - self.alpha() * node_current + self.voltage_update_scale * torch.tanh(delta_voltage)
            voltage = boundary_mask * boundary_input + (1.0 - boundary_mask) * voltage
            global_state = self.update_global(global_state, voltage, conductance, node_current)

        conductance = F.softplus(edge_raw) + 1e-4
        node_current, _diff, _phys_current = self.aggregate_node_current(voltage, conductance)
        kcl_residual = (((1.0 - boundary_mask) * node_current) ** 2).mean()
        return voltage, conductance, edge_raw, global_state, node_current, kcl_residual

    def decode_edges(self, voltage, conductance, edge_raw, global_state):
        vu = voltage[:, self.edge_u, :]
        vv = voltage[:, self.edge_v, :]
        diff = torch.abs(vu - vv)
        avg = 0.5 * (vu + vv)
        stacked = torch.stack([vu, vv, diff, avg], dim=-1)
        feat_mean = stacked.mean(dim=2)
        feat_max = stacked.amax(dim=2)
        feat_std = stacked.std(dim=2, unbiased=False)
        global_edge = global_state.unsqueeze(1).expand(-1, conductance.size(1), -1)
        feat = torch.cat(
            [
                feat_mean,
                feat_max,
                feat_std,
                conductance.unsqueeze(-1),
                edge_raw.unsqueeze(-1),
                global_edge,
            ],
            dim=-1,
        )
        return self.decoder(feat, self.max_abs)

    def forward(self, batch, return_aux=False):
        boundary_input = batch.x.float().view(batch.num_graphs, NUM_NODES, self.num_excitations)
        boundary_mask = getattr(batch, "boundary_mask", None)
        if boundary_mask is None:
            boundary_mask = self.default_boundary_mask.repeat(batch.num_graphs, 1).to(boundary_input.device)
        boundary_mask = boundary_mask.view(batch.num_graphs, NUM_NODES, 1).float()

        voltage, conductance, edge_raw, global_state, node_current, kcl_residual = self.physics_iterate(
            boundary_input, boundary_mask
        )
        pred, mask_prob, value = self.decode_edges(voltage, conductance, edge_raw, global_state)

        if return_aux:
            return pred, {
                "mask_prob": mask_prob,
                "value": value,
                "node_voltage": voltage,
                "node_current": node_current,
                "edge_conductance": conductance,
                "edge_raw": edge_raw,
                "global_state": global_state,
                "kcl_residual": kcl_residual,
                "alpha": self.alpha().detach(),
                "conductance": conductance.detach(),
            }
        return pred
