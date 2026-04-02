from __future__ import annotations

import csv
import json
import math
import random
import re
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
VENDOR_DIR = WORKSPACE_ROOT / ".vendor_torchpy311"
if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from topologies import TopologySpec, make_grid_topology


NUM_CLASSES = 4
DEFAULT_EXCITATIONS = 32
BASE_R = 1000.0
DEFAULT_SOURCE_GRID_SIZE = 8


try:
    from scipy.special import expit
except Exception:
    def expit(x):
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x, dtype=np.float64)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[~pos])
        out[~pos] = exp_x / (1.0 + exp_x)
        return out


class ClsDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class RegDataset(Dataset):
    def __init__(self, x, y_change, y_delta):
        self.x = torch.from_numpy(x).float()
        self.y_change = torch.from_numpy(y_change).float()
        self.y_delta = torch.from_numpy(y_delta).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_change[idx], self.y_delta[idx]


def sanitize_dataset_tag(raw_tag: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(raw_tag).strip())
    safe = safe.strip("._-")
    return safe or "dataset"


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(n: int, seed: int):
    rng = random.Random(seed)
    ids = list(range(n))
    rng.shuffle(ids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def resolve_input_data_path(raw_path: str, script_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    bases = [Path.cwd(), script_dir, script_dir.parent, script_dir.parents[1], script_dir.parents[2], script_dir.parents[3]]
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (script_dir / path).resolve()


def resolve_runtime_path(raw_path: str, script_dir: Path) -> str:
    if not raw_path:
        return ""
    path = Path(raw_path)
    if path.is_absolute():
        return str(path.resolve())
    bases = [Path.cwd(), script_dir, script_dir.parent, script_dir.parents[1], script_dir.parents[2], script_dir.parents[3]]
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists() or candidate.parent.exists():
            return str(candidate)
    return str((script_dir / path).resolve())


def resolve_dataset_runtime_paths(args, script_dir: Path, default_cache_name: str):
    data_path = resolve_input_data_path(args.data_path, script_dir)
    dataset_tag = sanitize_dataset_tag(args.dataset_tag or data_path.stem)
    cache_base = script_dir / "cache" if args.cache_path == default_cache_name else Path(args.cache_path)
    cache_path = (cache_base / dataset_tag / default_cache_name) if args.dataset_subdir else cache_base
    out_base = script_dir / "outputs" if args.out_dir == "./outputs" else Path(args.out_dir)
    out_dir = out_base / dataset_tag if args.dataset_subdir else out_base
    args.data_path = str(data_path)
    args.dataset_tag = dataset_tag
    args.cache_path = str(cache_path)
    args.out_dir = str(out_dir)
    if hasattr(args, "pretrained_model_path"):
        args.pretrained_model_path = resolve_runtime_path(args.pretrained_model_path, script_dir)


def resolve_inference_runtime_paths(args, script_dir: Path, default_cache_name: str):
    data_path = resolve_input_data_path(args.data_path, script_dir)
    dataset_tag = sanitize_dataset_tag(args.dataset_tag or data_path.stem)
    cache_base = script_dir / "cache" if args.cache_path == default_cache_name else Path(args.cache_path)
    cache_path = (cache_base / dataset_tag / default_cache_name) if args.dataset_subdir else cache_base
    outputs_root = script_dir / "outputs"
    outputs_dir = outputs_root / dataset_tag if args.dataset_subdir else outputs_root
    model_path = outputs_dir / "model_last.pt" if args.model_path == "./outputs/model_last.pt" else Path(args.model_path)
    metrics_path = outputs_dir / "metrics.json" if args.metrics_path == "./outputs/metrics.json" else Path(args.metrics_path)
    standardization = outputs_dir / "standardization.npz" if args.standardization == "./outputs/standardization.npz" else Path(args.standardization)
    args.data_path = str(data_path)
    args.dataset_tag = dataset_tag
    args.cache_path = str(cache_path)
    args.model_path = str(model_path)
    args.metrics_path = str(metrics_path)
    args.standardization = str(standardization)


def parse_voltage_columns(fieldnames):
    v_cols = [c for c in fieldnames if c.startswith("v_node")]
    ext_nodes = [int(c.replace("v_node", "")) for c in v_cols]
    return v_cols, np.array(ext_nodes, dtype=np.int64)


def load_dataset_meta(csv_path: Path):
    meta_path = csv_path.with_name(f"{csv_path.stem}_meta.json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def infer_num_excitations(meta: dict):
    excitations = meta.get("excitations", [])
    if excitations:
        return int(len(excitations))
    return DEFAULT_EXCITATIONS


def build_source_topology_from_meta(meta: dict):
    if meta.get("num_nodes") and meta.get("resistor_edges") and meta.get("external_nodes_clockwise") and meta.get("node_coords"):
        resistor_edges = tuple(tuple(int(v) for v in edge) for edge in meta["resistor_edges"])
        boundary_nodes = tuple(int(v) for v in meta["external_nodes_clockwise"])
        node_coords = tuple((float(x), float(y)) for x, y in meta["node_coords"])
        message_edges = tuple(
            edge
            for u, v in resistor_edges
            for edge in ((u, v), (v, u))
        )
        return TopologySpec(
            key=str(meta.get("topology_key", "source_topology")),
            title=str(meta.get("topology_title", meta.get("topology_key", "Source Topology"))),
            num_nodes=int(meta["num_nodes"]),
            resistor_edges=resistor_edges,
            message_edges=message_edges,
            node_coords=node_coords,
            boundary_nodes_clockwise=boundary_nodes,
            notes="Topology restored from dataset meta for GNN_EXPAND.",
        )
    grid_size = int(meta.get("grid_size", DEFAULT_SOURCE_GRID_SIZE))
    return make_grid_topology(
        key=f"source_grid_{grid_size}x{grid_size}",
        title=f"Source Grid {grid_size}x{grid_size}",
        rows=grid_size,
        cols=grid_size,
        notes="Original clean 8x8 source topology used for GNN_EXPAND label remapping.",
    )


def build_boundary_node_mapping(raw_ext_nodes, topology: TopologySpec):
    raw_ext_nodes = np.asarray(raw_ext_nodes, dtype=np.int64)
    if len(raw_ext_nodes) == 0:
        raise RuntimeError("No external voltage columns found in CSV.")
    if len(raw_ext_nodes) != topology.num_boundary_nodes:
        raise RuntimeError(
            f"CSV exposes {len(raw_ext_nodes)} external nodes, but topology {topology.key} expects "
            f"{topology.num_boundary_nodes} boundary nodes."
        )
    return np.asarray(topology.boundary_nodes_clockwise, dtype=np.int64)


def remap_excitation_nodes(src_nodes_raw, gnd_nodes_raw, raw_ext_nodes, topology: TopologySpec):
    position_by_node = {int(node_id): idx for idx, node_id in enumerate(np.asarray(raw_ext_nodes, dtype=np.int64).tolist())}
    src_nodes = []
    gnd_nodes = []
    for src, gnd in zip(src_nodes_raw, gnd_nodes_raw):
        if int(src) not in position_by_node or int(gnd) not in position_by_node:
            raise RuntimeError("Found src/gnd node outside CSV boundary-node list.")
        src_nodes.append(int(topology.boundary_nodes_clockwise[position_by_node[int(src)]]))
        gnd_nodes.append(int(topology.boundary_nodes_clockwise[position_by_node[int(gnd)]]))
    return np.asarray(src_nodes, dtype=np.int64), np.asarray(gnd_nodes, dtype=np.int64)


def validate_topology_inputs(topology: TopologySpec, ext_nodes, src_nodes, gnd_nodes):
    if len(ext_nodes) == 0:
        raise RuntimeError("No external voltage columns found in CSV.")
    if np.any(ext_nodes < 0) or np.any(ext_nodes >= topology.num_nodes):
        raise RuntimeError(f"External node ids exceed topology range: max node={topology.num_nodes - 1}")
    if np.any(src_nodes < 0) or np.any(src_nodes >= topology.num_nodes):
        raise RuntimeError("src_node out of topology range.")
    if np.any(gnd_nodes < 0) or np.any(gnd_nodes >= topology.num_nodes):
        raise RuntimeError("gnd_node out of topology range.")


def to_graph_input(x_delta, ext_nodes, src_nodes, gnd_nodes, num_nodes: int):
    n = x_delta.shape[0]
    num_excitations = x_delta.shape[1]
    graphs = np.zeros((n, num_excitations, num_nodes, 4), dtype=np.float32)
    graphs[:, :, ext_nodes, 3] = 1.0
    graphs[:, :, ext_nodes, 2] = x_delta
    ex_ids = np.arange(num_excitations, dtype=np.int64)
    for i in range(n):
        graphs[i, ex_ids, src_nodes, 0] = 1.0
        graphs[i, ex_ids, gnd_nodes, 1] = 1.0
    return graphs


def edge_midpoint_and_angle(node_coords, edge):
    u, v = edge
    x1, y1 = node_coords[u]
    x2, y2 = node_coords[v]
    dx = x2 - x1
    dy = y2 - y1
    angle = abs(math.degrees(math.atan2(dy, dx)))
    if angle > 90.0:
        angle = 180.0 - angle
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5), angle


def build_resistor_id_map(source_topology: TopologySpec, target_topology: TopologySpec):
    if (
        source_topology.num_nodes == target_topology.num_nodes
        and tuple(source_topology.resistor_edges) == tuple(target_topology.resistor_edges)
    ):
        return np.arange(len(target_topology.resistor_edges), dtype=np.int64)
    source_stats = [edge_midpoint_and_angle(source_topology.node_coords, edge) for edge in source_topology.resistor_edges]
    target_stats = [edge_midpoint_and_angle(target_topology.node_coords, edge) for edge in target_topology.resistor_edges]
    mapped = []
    for source_mid, source_angle in source_stats:
        best_idx = 0
        best_cost = float("inf")
        for idx, (target_mid, target_angle) in enumerate(target_stats):
            dist = math.hypot(source_mid[0] - target_mid[0], source_mid[1] - target_mid[1])
            angle_gap = min(abs(source_angle - target_angle), abs((180.0 - source_angle) - target_angle)) / 90.0
            cost = dist + 0.08 * angle_gap
            if cost < best_cost:
                best_cost = cost
                best_idx = idx
        mapped.append(best_idx)
    return np.asarray(mapped, dtype=np.int64)


def build_cls_dataset(csv_path: Path, cache_path: Path, topology: TopologySpec):
    if cache_path.exists():
        d = np.load(cache_path)
        return d["x"], d["y"], d["ext_nodes"]

    raw_meta = load_dataset_meta(csv_path)
    num_excitations = infer_num_excitations(raw_meta)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        v_cols, raw_ext_nodes = parse_voltage_columns(reader.fieldnames)
        ext_nodes = build_boundary_node_mapping(raw_ext_nodes, topology)
        v_num = len(v_cols)
        sums = np.zeros((num_excitations, v_num), dtype=np.float64)
        cnts = np.zeros(num_excitations, dtype=np.int64)
        prev_combo = None
        ex_idx = 0
        for row in reader:
            cid = int(row["combo_id"])
            if cid != prev_combo:
                prev_combo = cid
                ex_idx = 0
            if ex_idx >= num_excitations:
                raise RuntimeError(f"Combo {cid} has more than {num_excitations} excitations in {csv_path}.")
            if int(row["change_count"]) == 0:
                v = np.array([float(row[c]) for c in v_cols], dtype=np.float64)
                sums[ex_idx] += v
                cnts[ex_idx] += 1
            ex_idx += 1
        if np.any(cnts == 0):
            raise RuntimeError("0-change samples are insufficient to compute base mean.")
        base_mean = (sums / cnts[:, None]).astype(np.float32)

    x_list = []
    y_list = []
    src_nodes = None
    gnd_nodes = None
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        v_cols, raw_ext_nodes = parse_voltage_columns(reader.fieldnames)
        prev_combo = None
        combo_rows = []
        combo_src_raw = []
        combo_gnd_raw = []
        label = 0
        for row in reader:
            cid = int(row["combo_id"])
            if cid != prev_combo:
                if prev_combo is not None:
                    arr = np.stack(combo_rows, axis=0).astype(np.float32)
                    x_list.append(arr - base_mean)
                    y_list.append(label)
                    if src_nodes is None:
                        src_nodes, gnd_nodes = remap_excitation_nodes(combo_src_raw, combo_gnd_raw, raw_ext_nodes, topology)
                prev_combo = cid
                combo_rows = []
                combo_src_raw = []
                combo_gnd_raw = []
                label = int(row["change_count"])
            combo_rows.append(np.array([float(row[c]) for c in v_cols], dtype=np.float32))
            combo_src_raw.append(int(row["src_node"]))
            combo_gnd_raw.append(int(row["gnd_node"]))
        if combo_rows:
            arr = np.stack(combo_rows, axis=0).astype(np.float32)
            x_list.append(arr - base_mean)
            y_list.append(label)
            if src_nodes is None:
                src_nodes, gnd_nodes = remap_excitation_nodes(combo_src_raw, combo_gnd_raw, raw_ext_nodes, topology)

    validate_topology_inputs(topology, ext_nodes, src_nodes, gnd_nodes)
    x_delta = np.stack(x_list, axis=0).astype(np.float32)
    x = to_graph_input(x_delta, ext_nodes, src_nodes, gnd_nodes, topology.num_nodes)
    y = np.array(y_list, dtype=np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=x, y=y, ext_nodes=ext_nodes)
    return x, y, ext_nodes


def build_reg_dataset(csv_path: Path, cache_path: Path, topology: TopologySpec):
    if cache_path.exists():
        d = np.load(cache_path)
        return d["x"], d["y_change"], d["y_delta"], d["ext_nodes"]

    raw_meta = load_dataset_meta(csv_path)
    source_topology = build_source_topology_from_meta(raw_meta)
    resistor_id_map = build_resistor_id_map(source_topology, topology)
    num_excitations = infer_num_excitations(raw_meta)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        v_cols, raw_ext_nodes = parse_voltage_columns(reader.fieldnames)
        ext_nodes = build_boundary_node_mapping(raw_ext_nodes, topology)
        v_num = len(v_cols)
        sums = np.zeros((num_excitations, v_num), dtype=np.float64)
        cnts = np.zeros(num_excitations, dtype=np.int64)
        prev_combo = None
        ex_idx = 0
        for row in reader:
            cid = int(row["combo_id"])
            if cid != prev_combo:
                prev_combo = cid
                ex_idx = 0
            if ex_idx >= num_excitations:
                raise RuntimeError(f"Combo {cid} has more than {num_excitations} excitations in {csv_path}.")
            if int(row["change_count"]) == 0:
                v = np.array([float(row[c]) for c in v_cols], dtype=np.float64)
                sums[ex_idx] += v
                cnts[ex_idx] += 1
            ex_idx += 1
        if np.any(cnts == 0):
            raise RuntimeError("0-change samples are insufficient to compute base mean.")
        base_mean = (sums / cnts[:, None]).astype(np.float32)

    x_list, yc_list, yd_list = [], [], []
    src_nodes = None
    gnd_nodes = None
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        v_cols, raw_ext_nodes = parse_voltage_columns(reader.fieldnames)
        prev_combo = None
        combo_rows = []
        combo_src_raw = []
        combo_gnd_raw = []
        y_change = None
        y_delta = None
        for row in reader:
            cid = int(row["combo_id"])
            if cid != prev_combo:
                if prev_combo is not None:
                    arr = np.stack(combo_rows, axis=0).astype(np.float32)
                    x_list.append(arr - base_mean)
                    yc_list.append(y_change)
                    yd_list.append(y_delta)
                    if src_nodes is None:
                        src_nodes, gnd_nodes = remap_excitation_nodes(combo_src_raw, combo_gnd_raw, raw_ext_nodes, topology)
                prev_combo = cid
                combo_rows = []
                combo_src_raw = []
                combo_gnd_raw = []
                y_change = np.zeros(topology.num_resistors, dtype=np.float32)
                y_delta = np.zeros(topology.num_resistors, dtype=np.float32)
                remapped_values: dict[int, list[float]] = {}
                for i in (1, 2, 3):
                    rid = int(row[f"r{i}_id"])
                    if rid >= 0:
                        if rid >= len(resistor_id_map):
                            raise RuntimeError(
                                f"Resistor id {rid} exceeds source resistor count {len(resistor_id_map)}."
                            )
                        val = float(row[f"r{i}_value"])
                        target_rid = int(resistor_id_map[rid])
                        remapped_values.setdefault(target_rid, []).append(val - BASE_R)
                for target_rid, values in remapped_values.items():
                    y_change[target_rid] = 1.0
                    y_delta[target_rid] = float(np.mean(values))
            combo_rows.append(np.array([float(row[c]) for c in v_cols], dtype=np.float32))
            combo_src_raw.append(int(row["src_node"]))
            combo_gnd_raw.append(int(row["gnd_node"]))
        if combo_rows:
            arr = np.stack(combo_rows, axis=0).astype(np.float32)
            x_list.append(arr - base_mean)
            yc_list.append(y_change)
            yd_list.append(y_delta)
            if src_nodes is None:
                src_nodes, gnd_nodes = remap_excitation_nodes(combo_src_raw, combo_gnd_raw, raw_ext_nodes, topology)

    validate_topology_inputs(topology, ext_nodes, src_nodes, gnd_nodes)
    x_delta = np.stack(x_list, axis=0).astype(np.float32)
    x = to_graph_input(x_delta, ext_nodes, src_nodes, gnd_nodes, topology.num_nodes)
    y_change = np.stack(yc_list, axis=0).astype(np.float32)
    y_delta = np.stack(yd_list, axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=x, y_change=y_change, y_delta=y_delta, ext_nodes=ext_nodes)
    return x, y_change, y_delta, ext_nodes


def standardize_graph_voltage(x, mean, std, ext_nodes):
    x_std = x.copy()
    x_std[:, :, ext_nodes, 2] = (x_std[:, :, ext_nodes, 2] - mean) / std
    return x_std.astype(np.float32)


def class_weights(y):
    cnt = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    total = cnt.sum()
    w = total / (NUM_CLASSES * np.maximum(cnt, 1.0))
    return torch.tensor(w, dtype=torch.float32)


def confusion(pred, true, num_classes=NUM_CLASSES):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true, pred):
        cm[int(t), int(p)] += 1
    return cm


def macro_f1(cm):
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1s.append(0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(f1s))


def class_recall(cm, c):
    row = cm[c, :].sum()
    if row == 0:
        return 0.0
    return float(cm[c, c] / row)


def weighted_score(cm, penalty_32=0.12, bonus_r3=0.06, bonus_r2=0.05):
    base = macro_f1(cm)
    row3 = max(int(cm[3, :].sum()), 1)
    err_32 = float(cm[3, 2] / row3)
    r2 = class_recall(cm, 2)
    r3 = class_recall(cm, 3)
    return base - penalty_32 * err_32 + bonus_r3 * r3 + bonus_r2 * r2


def search_thresholds_constrained_weighted(val_probs, val_true, step=0.01, penalty_32=0.12, bonus_r3=0.06, bonus_r2=0.05):
    grid = np.arange(0.05, 0.951, step)
    best_score = -1e9
    best_t = [0.5, 0.5, 0.5]
    best_f = -1.0
    for t1 in grid:
        m1 = val_probs[:, 0] > t1
        for t2 in grid[grid >= t1]:
            m2 = val_probs[:, 1] > t2
            for t3 in grid[grid >= t2]:
                pred = m1.astype(np.int64) + m2.astype(np.int64) + (val_probs[:, 2] > t3).astype(np.int64)
                cm = confusion(pred, val_true)
                score = weighted_score(cm, penalty_32=penalty_32, bonus_r3=bonus_r3, bonus_r2=bonus_r2)
                if score > best_score:
                    best_score = float(score)
                    best_t = [float(t1), float(t2), float(t3)]
                    best_f = macro_f1(cm)
    return best_t, best_f, best_score


def coral_targets(labels, num_classes=NUM_CLASSES):
    thr = torch.arange(num_classes - 1, device=labels.device).view(1, -1)
    return (labels.view(-1, 1) > thr).float()


def coral_loss(logits, labels, sample_w=None):
    tgt = coral_targets(labels)
    loss = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
    if sample_w is not None:
        loss = loss * sample_w.view(-1, 1)
    return loss.mean()


def supervised_contrastive_loss(features, labels, temperature=0.12):
    if features.size(0) < 2:
        return features.new_zeros(())
    features = F.normalize(features, dim=-1)
    labels = labels.view(-1)
    logits = features @ features.t() / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    valid_anchor = ((labels == 2) | (labels == 3))
    pos_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & (~eye)
    exp_logits = torch.exp(logits) * (~eye).float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    pos_count = pos_mask.sum(dim=1)
    anchor_mask = valid_anchor & (pos_count > 0)
    if not anchor_mask.any():
        return features.new_zeros(())
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_count.clamp_min(1)
    return -mean_log_prob_pos[anchor_mask].mean()


def mask_l1_weight_at_epoch(epoch, args):
    start = float(args.lambda_mask_l1_start)
    target = float(args.lambda_mask_l1)
    warmup_epochs = max(int(args.lambda_mask_warmup_epochs), 0)
    if warmup_epochs <= 1:
        return target
    if epoch >= warmup_epochs:
        return target
    progress = float(epoch - 1) / float(warmup_epochs - 1)
    return start + (target - start) * progress


def compute_o4a2_loss(pred, aux, ycb, ydb, mask_l1_weight, pos_weight, args):
    loss_reg = F.smooth_l1_loss(pred, ydb, beta=args.reg_beta)
    loss_mask = F.binary_cross_entropy_with_logits(aux["mask_logits"], ycb, pos_weight=pos_weight)
    loss_sparse = aux["mask_prob"].mean()
    return loss_reg + args.mask_bce_weight * loss_mask + mask_l1_weight * loss_sparse


def search_best_count_threshold(model, loader, device, t_min=40, t_max=80, t_step=1):
    all_pred = []
    all_true = []
    with torch.no_grad():
        for xb, ycb, _ in loader:
            pred, _aux = model(xb.to(device), return_aux=True)
            all_pred.append(np.abs(pred.cpu().numpy()))
            all_true.append((ycb.numpy() > 0.5).sum(axis=1))
    pred_abs = np.concatenate(all_pred, axis=0)
    true_k = np.concatenate(all_true, axis=0).astype(np.int64)
    best_t = float(t_min)
    best_f = -1.0
    for t in np.arange(t_min, t_max + 1e-9, t_step):
        pred_k = np.clip((pred_abs > t).sum(axis=1), 0, 3).astype(np.int64)
        cm = confusion(pred_k, true_k, 4)
        f = macro_f1(cm)
        if f > best_f:
            best_f = float(f)
            best_t = float(t)
    return best_t, best_f


def evaluate_reg_val(model, loader, device, args, mask_l1_weight, pos_weight):
    model.eval()
    total_loss = 0.0
    sum_all_abs = 0.0
    n_all = 0
    sum_changed_abs = 0.0
    n_changed = 0
    active_counts = []
    mask_means = []
    with torch.no_grad():
        for xb, ycb, ydb in loader:
            xb, ycb, ydb = xb.to(device), ycb.to(device), ydb.to(device)
            pred, aux = model(xb, return_aux=True)
            loss = compute_o4a2_loss(pred, aux, ycb, ydb, mask_l1_weight, pos_weight, args)
            total_loss += loss.item() * xb.size(0)
            sum_all_abs += torch.abs(pred - ydb).sum().item()
            n_all += ydb.numel()
            mask = ycb > 0.5
            if mask.any():
                sum_changed_abs += torch.abs(pred[mask] - ydb[mask]).sum().item()
                n_changed += int(mask.sum().item())
            active_counts.extend((pred.abs() > args.eval_sparse_threshold).sum(dim=1).cpu().tolist())
            mask_means.append(aux["mask_prob"].mean().item())
    val_loss = total_loss / max(len(loader.dataset), 1)
    val_mae_all = sum_all_abs / max(n_all, 1)
    val_mae_changed = sum_changed_abs / max(n_changed, 1)
    val_avg_gt = float(np.mean(active_counts)) if active_counts else 0.0
    val_mask_mean = float(np.mean(mask_means)) if mask_means else 0.0
    val_score = val_mae_changed + args.val_mae_all_alpha * val_mae_all + args.val_sparse_alpha * val_avg_gt
    return val_loss, val_mae_all, val_mae_changed, val_avg_gt, val_mask_mean, val_score


def load_partial_state_dict(model, model_path: str, device, label: str):
    if not model_path:
        return {"loaded": 0, "skipped": 0}
    path = Path(model_path)
    if not path.exists():
        print(f"[Warm Start] skip {label}: model path does not exist -> {model_path}")
        return {"loaded": 0, "skipped": 0}
    state_dict = torch.load(model_path, map_location=device)
    current = model.state_dict()
    matched = {}
    skipped = []
    for key, value in state_dict.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            matched[key] = value
        else:
            skipped.append(key)
    current.update(matched)
    model.load_state_dict(current, strict=False)
    print(f"[Warm Start] loaded {len(matched)} tensors for {label} from {model_path}; skipped {len(skipped)}")
    return {"loaded": len(matched), "skipped": len(skipped)}


def select_focus_indices(test_idx, true_counts, num_samples, seed, focus_high_change=True, min_true_change=2):
    rng = random.Random(seed)
    pool = list(test_idx)
    if not focus_high_change:
        return rng.sample(pool, k=min(num_samples, len(pool)))
    high = [idx for idx in pool if int(true_counts[idx]) >= min_true_change]
    low = [idx for idx in pool if int(true_counts[idx]) < min_true_change]
    rng.shuffle(high)
    rng.shuffle(low)
    selected = high[: min(num_samples, len(high))]
    if len(selected) < num_samples:
        selected.extend(low[: num_samples - len(selected)])
    return selected


def build_edge_adjacency(resistor_edges: tuple[tuple[int, int], ...]):
    adjacency = [set() for _ in resistor_edges]
    for i, (u1, v1) in enumerate(resistor_edges):
        nodes_i = {u1, v1}
        for j in range(i + 1, len(resistor_edges)):
            u2, v2 = resistor_edges[j]
            if nodes_i.intersection({u2, v2}):
                adjacency[i].add(j)
                adjacency[j].add(i)
    return adjacency


def predict_count_from_coral(probs, thresholds):
    thr = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    return (probs > thr).sum(axis=1).astype(np.int64)


def apply_near_miss(topk_ids, sorted_ids, reg_abs, reg_prob, adjacency, k, near_ratio=0.92):
    if k <= 0 or len(sorted_ids) <= k or len(topk_ids) < 2:
        return topk_ids
    selected = list(topk_ids)
    selected_set = set(selected)
    adjacent_candidates = []
    for edge_id in selected:
        if adjacency[edge_id].intersection(selected_set - {edge_id}):
            adjacent_candidates.append(edge_id)
    if not adjacent_candidates:
        return selected
    weakest = min(adjacent_candidates, key=lambda eid: (float(reg_prob[eid]), float(reg_abs[eid])))
    challenger = int(sorted_ids[k])
    if challenger in selected_set:
        return selected
    weak_prob = float(reg_prob[weakest])
    weak_score = float(reg_abs[weakest])
    chal_prob = float(reg_prob[challenger])
    chal_score = float(reg_abs[challenger])
    if chal_prob < near_ratio * weak_prob and chal_score < near_ratio * weak_score:
        return selected
    weak_adj = len(adjacency[weakest].intersection(selected_set - {weakest}))
    chal_adj = len(adjacency[challenger].intersection(selected_set - {weakest}))
    if chal_adj > weak_adj:
        return selected
    selected[selected.index(weakest)] = challenger
    selected = sorted(selected, key=lambda eid: -reg_abs[eid])
    return selected


def extract_edge_values(delta, ids):
    return [float(delta[int(eid)]) for eid in ids]


def dump_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
