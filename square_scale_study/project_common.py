from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from bootstrap import prepend_vendor_dir

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
VENDOR_DIR = WORKSPACE_ROOT / ".vendor_torchpy311"

prepend_vendor_dir(VENDOR_DIR, required_version=(3, 11))

import numpy as np


BASE_R = 1000.0
DEFAULT_CURRENT_A = 0.01
DEFAULT_CHANGE_LIMIT = 0.20
DEFAULT_ID_PASS_THRESHOLD = 0.98
DEFAULT_VALUE_PASS_THRESHOLD = 0.90


@dataclass(frozen=True)
class SquareTopologySpec:
    grid_size: int
    num_nodes: int
    resistor_edges: tuple[tuple[int, int], ...]
    message_edges: tuple[tuple[int, int], ...]
    boundary_nodes_clockwise: tuple[int, ...]
    node_coords: tuple[tuple[float, float], ...]

    @property
    def key(self) -> str:
        return f"N{self.grid_size}x{self.grid_size}"

    @property
    def num_resistors(self) -> int:
        return len(self.resistor_edges)

    @property
    def port_count(self) -> int:
        return len(self.boundary_nodes_clockwise)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_float(value: float, decimals: int = 6) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.{decimals}f}"


def build_square_topology(grid_size: int) -> SquareTopologySpec:
    num_nodes = grid_size * grid_size
    resistor_edges: list[tuple[int, int]] = []
    for r in range(grid_size):
        for c in range(grid_size - 1):
            resistor_edges.append((r * grid_size + c, r * grid_size + c + 1))
        if r < grid_size - 1:
            for c in range(grid_size):
                resistor_edges.append((r * grid_size + c, (r + 1) * grid_size + c))

    message_edges = tuple(edge for u, v in resistor_edges for edge in ((u, v), (v, u)))
    boundary_nodes = external_nodes_clockwise(grid_size)
    node_coords = normalize_grid_coords(grid_size)
    return SquareTopologySpec(
        grid_size=grid_size,
        num_nodes=num_nodes,
        resistor_edges=tuple(resistor_edges),
        message_edges=message_edges,
        boundary_nodes_clockwise=tuple(boundary_nodes),
        node_coords=node_coords,
    )


def normalize_grid_coords(grid_size: int) -> tuple[tuple[float, float], ...]:
    coords: list[tuple[float, float]] = []
    denom = max(grid_size - 1, 1)
    for r in range(grid_size):
        for c in range(grid_size):
            coords.append((float(c) / denom, float(grid_size - 1 - r) / denom))
    return tuple(coords)


def external_nodes_clockwise(grid_size: int) -> list[int]:
    top = list(range(grid_size))
    right = [r * grid_size + (grid_size - 1) for r in range(1, grid_size)]
    bottom = list(range(grid_size * grid_size - 2, grid_size * (grid_size - 1) - 1, -1))
    left = [r * grid_size for r in range(grid_size - 2, 0, -1)]
    return top + right + bottom + left


def select_evenly_spaced_indices(total: int, count: int) -> list[int]:
    if count >= total:
        return list(range(total))
    raw = np.floor(np.linspace(0, total, num=count, endpoint=False)).astype(int).tolist()
    seen = set()
    ordered: list[int] = []
    for idx in raw:
        if idx not in seen:
            ordered.append(idx)
            seen.add(idx)
    cursor = 0
    while len(ordered) < count:
        if cursor not in seen:
            ordered.append(cursor)
            seen.add(cursor)
        cursor += 1
    return ordered


def build_boundary_excitations(topology: SquareTopologySpec, excitation_count: int | None = None) -> list[tuple[int, int]]:
    boundary_nodes = list(topology.boundary_nodes_clockwise)
    full = [(boundary_nodes[i], boundary_nodes[(i + 1) % len(boundary_nodes)]) for i in range(len(boundary_nodes))]
    if excitation_count is None or excitation_count >= len(full):
        return full
    keep = select_evenly_spaced_indices(len(full), excitation_count)
    return [full[idx] for idx in keep]


def build_conductance(num_nodes: int, values: np.ndarray, edges: Iterable[tuple[int, int]]) -> np.ndarray:
    gmat = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for rid, (n1, n2) in enumerate(edges):
        conductance = 1.0 / float(values[rid])
        gmat[n1, n1] += conductance
        gmat[n2, n2] += conductance
        gmat[n1, n2] -= conductance
        gmat[n2, n1] -= conductance
    return gmat


def build_rhs_matrix(num_nodes: int, ref_node: int, excitations: list[tuple[int, int]], current_a: float) -> tuple[np.ndarray, np.ndarray]:
    keep_idx = np.array([i for i in range(num_nodes) if i != ref_node], dtype=np.int64)
    node_to_reduced = {node: idx for idx, node in enumerate(keep_idx.tolist())}
    rhs = np.zeros((num_nodes - 1, len(excitations)), dtype=np.float64)
    for col, (src, gnd) in enumerate(excitations):
        if src != ref_node:
            rhs[node_to_reduced[src], col] += current_a
        if gnd != ref_node:
            rhs[node_to_reduced[gnd], col] -= current_a
    return keep_idx, rhs


def solve_all_excitations(
    gmat: np.ndarray,
    keep_idx: np.ndarray,
    ref_node: int,
    rhs_matrix: np.ndarray,
    excitations: list[tuple[int, int]],
) -> np.ndarray:
    g_reduced = gmat[np.ix_(keep_idx, keep_idx)]
    solved = np.linalg.solve(g_reduced, rhs_matrix)
    voltages = np.zeros((gmat.shape[0], rhs_matrix.shape[1]), dtype=np.float64)
    voltages[keep_idx, :] = solved
    for col, (_src, gnd) in enumerate(excitations):
        voltages[:, col] -= voltages[gnd, col]
        voltages[gnd, col] = 0.0
        voltages[ref_node, col] = voltages[ref_node, col]
    return voltages


def sample_change_bundle(
    num_resistors: int,
    k: int,
    rng: random.Random,
    seen: set[tuple[tuple[int, float], ...]],
    max_ratio: float = DEFAULT_CHANGE_LIMIT,
) -> list[tuple[int, float]]:
    while True:
        resistor_ids = sorted(rng.sample(range(num_resistors), k))
        changes: list[tuple[int, float]] = []
        for rid in resistor_ids:
            ratio = 0.0
            while abs(ratio) < 1e-9:
                ratio = rng.uniform(-max_ratio, max_ratio)
            new_r = round(BASE_R * (1.0 + ratio), 6)
            changes.append((rid, new_r))
        key = tuple(changes)
        if key not in seen:
            seen.add(key)
            return changes


def dataset_stem(grid_size: int, k: int, excitation_count: int | None = None) -> str:
    stem = f"square_N{grid_size}x{grid_size}_K{k}"
    if excitation_count is not None:
        stem += f"_E{excitation_count}"
    return stem


def run_dir_name(grid_size: int, k: int, port_count: int, excitation_count: int | None) -> str:
    name = f"N{grid_size}x{grid_size}_K{k}"
    if excitation_count is not None and excitation_count < port_count:
        name += f"_E{excitation_count}"
    return name


def dataset_dir(root: Path, grid_size: int) -> Path:
    return root / f"N{grid_size}x{grid_size}"


def generate_fixedk_dataset_bundle(
    grid_size: int,
    k: int,
    output_root: Path,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    current_a: float = DEFAULT_CURRENT_A,
    change_limit: float = DEFAULT_CHANGE_LIMIT,
    float_decimals: int = 6,
    excitation_count: int | None = None,
) -> Path:
    topology = build_square_topology(grid_size)
    excitations = build_boundary_excitations(topology, excitation_count=excitation_count)
    ref_node = topology.boundary_nodes_clockwise[0]
    keep_idx, rhs_matrix = build_rhs_matrix(topology.num_nodes, ref_node=ref_node, excitations=excitations, current_a=current_a)
    base_values = np.full(topology.num_resistors, BASE_R, dtype=np.float64)
    base_gmat = build_conductance(topology.num_nodes, base_values, topology.resistor_edges)
    baseline_voltages = solve_all_excitations(base_gmat, keep_idx, ref_node, rhs_matrix, excitations)
    boundary_nodes = list(topology.boundary_nodes_clockwise)
    baseline_boundary = baseline_voltages[boundary_nodes, :].T.astype(np.float32)

    base_dir = dataset_dir(output_root, grid_size)
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = dataset_stem(grid_size, k, excitation_count=excitation_count)
    split_specs = [("train", train_size), ("val", val_size), ("test", test_size)]

    header = (
        ["row_id", "sample_id", "excitation_idx", "src_node", "gnd_node"]
        + [f"v_node{n}" for n in boundary_nodes]
        + ["change_count"]
        + [item for i in range(1, k + 1) for item in (f"r{i}_id", f"r{i}_value")]
    )

    rng = random.Random(seed)
    seen: set[tuple[tuple[int, float], ...]] = set()
    row_id = 0
    split_paths: dict[str, str] = {}

    for split_name, split_size in split_specs:
        csv_path = base_dir / f"{stem}_{split_name}.csv"
        split_paths[split_name] = csv_path.name
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for sample_id in range(split_size):
                changes = sample_change_bundle(topology.num_resistors, k, rng, seen, max_ratio=change_limit)
                resistor_values = np.full(topology.num_resistors, BASE_R, dtype=np.float64)
                for rid, new_r in changes:
                    resistor_values[rid] = new_r
                gmat = build_conductance(topology.num_nodes, resistor_values, topology.resistor_edges)
                voltages = solve_all_excitations(gmat, keep_idx, ref_node, rhs_matrix, excitations)
                for excitation_idx, (src, gnd) in enumerate(excitations):
                    row = [row_id, sample_id, excitation_idx, src, gnd]
                    row.extend(fmt_float(float(voltages[node, excitation_idx]), float_decimals) for node in boundary_nodes)
                    row.append(k)
                    for rid, value in changes:
                        row.append(rid)
                        row.append(fmt_float(value, float_decimals))
                    writer.writerow(row)
                    row_id += 1

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "topology": {
            "grid_size": grid_size,
            "num_nodes": topology.num_nodes,
            "num_resistors": topology.num_resistors,
            "port_count": topology.port_count,
            "boundary_nodes_clockwise": boundary_nodes,
            "resistor_edges": [list(edge) for edge in topology.resistor_edges],
            "message_edges": [list(edge) for edge in topology.message_edges],
            "node_coords": [[float(x), float(y)] for x, y in topology.node_coords],
        },
        "dataset_stem": stem,
        "k": k,
        "base_resistance_ohm": BASE_R,
        "change_limit_ratio": change_limit,
        "current_source_a": current_a,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "seed": seed,
        "float_decimals": float_decimals,
        "excitations": [list(pair) for pair in excitations],
        "baseline_boundary_voltages": baseline_boundary.tolist(),
        "files": split_paths,
        "note": "Fixed-K square-grid dataset generated by direct Kirchhoff solves. Baseline is the nominal all-R0 network.",
    }
    meta_path = base_dir / f"{stem}_meta.json"
    dump_json(meta_path, meta)
    return meta_path


def topology_from_meta(meta: dict) -> SquareTopologySpec:
    topo = meta["topology"]
    return SquareTopologySpec(
        grid_size=int(topo["grid_size"]),
        num_nodes=int(topo["num_nodes"]),
        resistor_edges=tuple(tuple(int(v) for v in edge) for edge in topo["resistor_edges"]),
        message_edges=tuple(tuple(int(v) for v in edge) for edge in topo["message_edges"]),
        boundary_nodes_clockwise=tuple(int(v) for v in topo["boundary_nodes_clockwise"]),
        node_coords=tuple((float(x), float(y)) for x, y in topo["node_coords"]),
    )


def load_split_from_meta(meta_path: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, SquareTopologySpec, dict]:
    meta = load_json(meta_path)
    topology = topology_from_meta(meta)
    csv_path = meta_path.parent / meta["files"][split]
    baseline = np.asarray(meta["baseline_boundary_voltages"], dtype=np.float32)
    boundary_nodes = np.asarray(topology.boundary_nodes_clockwise, dtype=np.int64)
    excitations = [(int(src), int(gnd)) for src, gnd in meta["excitations"]]
    num_excitations = len(excitations)
    num_boundary_nodes = len(boundary_nodes)
    k = int(meta["k"])
    num_resistors = topology.num_resistors

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    sample_ids: list[int] = []
    sample_rows: list[np.ndarray] = []
    current_sample_id: int | None = None
    current_target: np.ndarray | None = None
    voltage_columns = [f"v_node{node}" for node in boundary_nodes]

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = int(row["sample_id"])
            if current_sample_id is None:
                current_sample_id = sample_id
                current_target = np.zeros(num_resistors, dtype=np.float32)
                for idx in range(1, k + 1):
                    rid = int(row[f"r{idx}_id"])
                    value = float(row[f"r{idx}_value"])
                    current_target[rid] = value - BASE_R
            elif sample_id != current_sample_id:
                arr = np.stack(sample_rows, axis=0).astype(np.float32)
                if arr.shape != (num_excitations, num_boundary_nodes):
                    raise RuntimeError(f"Sample {current_sample_id} has unexpected excitation shape {arr.shape}")
                x_list.append(arr - baseline)
                y_list.append(current_target)
                sample_ids.append(current_sample_id)
                sample_rows = []
                current_sample_id = sample_id
                current_target = np.zeros(num_resistors, dtype=np.float32)
                for idx in range(1, k + 1):
                    rid = int(row[f"r{idx}_id"])
                    value = float(row[f"r{idx}_value"])
                    current_target[rid] = value - BASE_R
            sample_rows.append(np.asarray([float(row[col]) for col in voltage_columns], dtype=np.float32))

    if current_sample_id is not None and current_target is not None:
        arr = np.stack(sample_rows, axis=0).astype(np.float32)
        if arr.shape != (num_excitations, num_boundary_nodes):
            raise RuntimeError(f"Sample {current_sample_id} has unexpected excitation shape {arr.shape}")
        x_list.append(arr - baseline)
        y_list.append(current_target)
        sample_ids.append(current_sample_id)

    x_delta = np.stack(x_list, axis=0).astype(np.float32)
    y_delta = np.stack(y_list, axis=0).astype(np.float32)
    graphs = to_graph_input(x_delta, boundary_nodes, excitations, topology.num_nodes)
    return graphs, y_delta, np.asarray(sample_ids, dtype=np.int64), topology, meta


def to_graph_input(
    x_delta: np.ndarray,
    boundary_nodes: np.ndarray,
    excitations: list[tuple[int, int]],
    num_nodes: int,
) -> np.ndarray:
    num_samples, num_excitations, _ = x_delta.shape
    graphs = np.zeros((num_samples, num_excitations, num_nodes, 4), dtype=np.float32)
    graphs[:, :, boundary_nodes, 3] = 1.0
    graphs[:, :, boundary_nodes, 2] = x_delta

    excitation_ids = np.arange(num_excitations, dtype=np.int64)
    src_nodes = np.asarray([src for src, _ in excitations], dtype=np.int64)
    gnd_nodes = np.asarray([gnd for _, gnd in excitations], dtype=np.int64)
    for sample_idx in range(num_samples):
        graphs[sample_idx, excitation_ids, src_nodes, 0] = 1.0
        graphs[sample_idx, excitation_ids, gnd_nodes, 1] = 1.0
    return graphs


def compute_standardization(x_train: np.ndarray, boundary_nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train[:, :, boundary_nodes, 2].mean(axis=0, keepdims=True)
    std = x_train[:, :, boundary_nodes, 2].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardization(x: np.ndarray, boundary_nodes: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    standardized = x.copy()
    standardized[:, :, boundary_nodes, 2] = (standardized[:, :, boundary_nodes, 2] - mean) / std
    return standardized.astype(np.float32)


def topk_indices(values: np.ndarray, k: int, use_abs: bool = True) -> np.ndarray:
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    base = np.abs(values) if use_abs else values
    part = np.argpartition(base, -k)[-k:]
    return part[np.argsort(-base[part])].astype(np.int64)


def compute_fixedk_metrics(
    pred_delta: np.ndarray,
    true_delta: np.ndarray,
    k: int,
    ranking_scores: np.ndarray | None = None,
    id_pass_threshold: float = DEFAULT_ID_PASS_THRESHOLD,
    value_pass_threshold: float = DEFAULT_VALUE_PASS_THRESHOLD,
) -> dict:
    if pred_delta.shape != true_delta.shape:
        raise RuntimeError(f"Shape mismatch: pred={pred_delta.shape}, true={true_delta.shape}")
    if ranking_scores is not None and ranking_scores.shape != true_delta.shape:
        raise RuntimeError(f"Ranking shape mismatch: ranking={ranking_scores.shape}, true={true_delta.shape}")

    exact_flags: list[float] = []
    overlap_scores: list[float] = []
    sample_mae_changed: list[float] = []
    prediction_rows: list[dict] = []

    for sample_idx in range(pred_delta.shape[0]):
        pred = pred_delta[sample_idx]
        target = true_delta[sample_idx]
        rank_base = pred if ranking_scores is None else ranking_scores[sample_idx]
        use_abs = ranking_scores is None
        true_support = np.flatnonzero(np.abs(target) > 1e-9).astype(np.int64)
        pred_support = topk_indices(rank_base, k, use_abs=use_abs)
        exact = float(set(pred_support.tolist()) == set(true_support.tolist()))
        overlap = float(len(set(pred_support.tolist()) & set(true_support.tolist())) / max(k, 1))
        mae_changed = float(np.abs(pred[true_support] - target[true_support]).mean()) if len(true_support) else 0.0

        exact_flags.append(exact)
        overlap_scores.append(overlap)
        sample_mae_changed.append(mae_changed)
        prediction_rows.append(
            {
                "sample_index": sample_idx,
                "true_support": true_support.tolist(),
                "pred_support": pred_support.tolist(),
                "support_exact": bool(exact),
                "support_overlap": overlap,
                "mae_changed_sample": mae_changed,
                "ranking_scores_on_pred_support": [float(rank_base[idx]) for idx in pred_support.tolist()],
                "true_delta_on_true_support": [float(target[idx]) for idx in true_support.tolist()],
                "pred_delta_on_true_support": [float(pred[idx]) for idx in true_support.tolist()],
            }
        )

    id_exact_rate = float(np.mean(exact_flags)) if exact_flags else 0.0
    id_mean_overlap = float(np.mean(overlap_scores)) if overlap_scores else 0.0
    mae_changed = float(np.mean(sample_mae_changed)) if sample_mae_changed else 0.0
    value_accuracy = max(0.0, 1.0 - mae_changed / (DEFAULT_CHANGE_LIMIT * BASE_R))
    pass_flag = bool(id_exact_rate >= id_pass_threshold and value_accuracy >= value_pass_threshold)

    return {
        "sample_count": int(pred_delta.shape[0]),
        "k": int(k),
        "id_exact_rate": id_exact_rate,
        "id_mean_overlap": id_mean_overlap,
        "mae_changed": mae_changed,
        "value_accuracy": value_accuracy,
        "id_pass_threshold": float(id_pass_threshold),
        "value_pass_threshold": float(value_pass_threshold),
        "pass_flag": pass_flag,
        "per_sample": prediction_rows,
    }


def write_predictions_csv(path: Path, sample_ids: np.ndarray, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "sample_index",
        "support_exact",
        "support_overlap",
        "mae_changed_sample",
        "true_support",
        "pred_support",
        "ranking_scores_on_pred_support",
        "true_delta_on_true_support",
        "pred_delta_on_true_support",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, sample_id in zip(metrics["per_sample"], sample_ids.tolist()):
            writer.writerow(
                {
                    "sample_id": int(sample_id),
                    "sample_index": row["sample_index"],
                    "support_exact": int(bool(row["support_exact"])),
                    "support_overlap": fmt_float(float(row["support_overlap"]), 6),
                    "mae_changed_sample": fmt_float(float(row["mae_changed_sample"]), 6),
                    "true_support": ";".join(str(v) for v in row["true_support"]),
                    "pred_support": ";".join(str(v) for v in row["pred_support"]),
                    "ranking_scores_on_pred_support": ";".join(
                        fmt_float(float(v), 6) for v in row["ranking_scores_on_pred_support"]
                    ),
                    "true_delta_on_true_support": ";".join(fmt_float(float(v), 6) for v in row["true_delta_on_true_support"]),
                    "pred_delta_on_true_support": ";".join(fmt_float(float(v), 6) for v in row["pred_delta_on_true_support"]),
                }
            )


def build_examples(metrics: dict, sample_ids: np.ndarray, limit: int = 12) -> list[dict]:
    rows = list(metrics["per_sample"])
    rows.sort(key=lambda row: (row["support_exact"], -row["mae_changed_sample"]))
    selected = rows[:limit]
    output: list[dict] = []
    for row in selected:
        sample_id = int(sample_ids[row["sample_index"]])
        output.append(
            {
                "sample_id": sample_id,
                "support_exact": bool(row["support_exact"]),
                "support_overlap": float(row["support_overlap"]),
                "mae_changed_sample": float(row["mae_changed_sample"]),
                "true_support": row["true_support"],
                "pred_support": row["pred_support"],
                "ranking_scores_on_pred_support": row["ranking_scores_on_pred_support"],
                "true_delta_on_true_support": row["true_delta_on_true_support"],
                "pred_delta_on_true_support": row["pred_delta_on_true_support"],
            }
        )
    return output


def excitation_label(excitation_count: int | None, port_count: int) -> str:
    return "full" if excitation_count is None or excitation_count >= port_count else str(excitation_count)
