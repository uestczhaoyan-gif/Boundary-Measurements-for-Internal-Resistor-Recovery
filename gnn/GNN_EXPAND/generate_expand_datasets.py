from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = WORKSPACE_ROOT / ".vendor_torchpy311"
if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

import numpy as np

COMMON_DIR = Path(__file__).resolve().parent / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from topologies import TOPOLOGY_REGISTRY


BASE_R = 1000.0
DEFAULT_TOPOLOGIES = (
    "square_10x10",
    "rect_6x10",
    "honeycomb_63",
    "circlecut_69",
)
EXTRA_EXCITATION_FRACTIONS = (
    (0.0, 0.5),
    (0.25, 0.75),
    (3.0 / 28.0, 17.0 / 28.0),
    (10.0 / 28.0, 24.0 / 28.0),
)


def fmt(x: float, decimals: int) -> str:
    if abs(x) < 1e-12:
        x = 0.0
    return f"{x:.{decimals}f}"


def combo_counts(total_combos: int):
    if total_combos == 10000:
        return {0: 700, 1: 3100, 2: 3100, 3: 3100}
    c0 = int(round(total_combos * 0.07))
    c1 = int(round(total_combos * 0.31))
    c2 = int(round(total_combos * 0.31))
    c3 = total_combos - c0 - c1 - c2
    return {0: c0, 1: c1, 2: c2, 3: c3}


def sample_changes(num_resistors: int, k: int, rng: random.Random, seen, min_ratio: float, max_ratio: float):
    if k == 0:
        return []
    while True:
        rids = sorted(rng.sample(range(num_resistors), k))
        changes = []
        for rid in rids:
            ratio = rng.uniform(min_ratio, max_ratio)
            sign = -1.0 if rng.random() < 0.5 else 1.0
            new_r = BASE_R * (1.0 + sign * ratio)
            changes.append((rid, round(new_r, 6)))
        key = tuple(changes)
        if key not in seen:
            seen.add(key)
            return changes


def build_boundary_only_excitations(ext_nodes: list[int]):
    n = len(ext_nodes)
    if n < 4:
        raise RuntimeError("Need at least 4 external nodes to build excitations.")
    excitations = [(ext_nodes[i], ext_nodes[(i + 1) % n]) for i in range(n)]

    used = set(excitations)
    for frac_a, frac_b in EXTRA_EXCITATION_FRACTIONS:
        idx_a = int(round(frac_a * n)) % n
        idx_b = int(round(frac_b * n)) % n
        pair = (ext_nodes[idx_a], ext_nodes[idx_b])
        if pair[0] == pair[1] or pair in used:
            continue
        excitations.append(pair)
        used.add(pair)
    return excitations


def build_conductance(num_nodes: int, values: np.ndarray, edges: tuple[tuple[int, int], ...]):
    gmat = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for rid, (n1, n2) in enumerate(edges):
        g = 1.0 / values[rid]
        gmat[n1, n1] += g
        gmat[n2, n2] += g
        gmat[n1, n2] -= g
        gmat[n2, n1] -= g
    return gmat


def build_rhs_matrix(num_nodes: int, ref_node: int, excitations: list[tuple[int, int]], current_a: float):
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
):
    g_reduced = gmat[np.ix_(keep_idx, keep_idx)]
    solved = np.linalg.solve(g_reduced, rhs_matrix)
    voltages = np.zeros((gmat.shape[0], rhs_matrix.shape[1]), dtype=np.float64)
    voltages[keep_idx, :] = solved
    for col, (_src, gnd) in enumerate(excitations):
        voltages[:, col] -= voltages[gnd, col]
        voltages[gnd, col] = 0.0
        voltages[ref_node, col] = voltages[ref_node, col]
    return voltages


def generate_dataset(
    topology_key: str,
    output_dir: Path,
    seed: int,
    total_combos: int,
    current_a: float,
    min_ratio: float,
    max_ratio: float,
    float_decimals: int,
):
    topology = TOPOLOGY_REGISTRY[topology_key]
    out_path = output_dir / f"{topology.key}.csv"
    meta_path = output_dir / f"{topology.key}_meta.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    ext_nodes = list(topology.boundary_nodes_clockwise)
    excitations = build_boundary_only_excitations(ext_nodes)
    counts = combo_counts(total_combos)
    keep_idx, rhs_matrix = build_rhs_matrix(topology.num_nodes, ref_node=ext_nodes[0], excitations=excitations, current_a=current_a)

    header = (
        ["row_id", "combo_id", "src_node", "gnd_node"]
        + [f"v_node{n}" for n in ext_nodes]
        + ["change_count", "r1_id", "r1_value", "r2_id", "r2_value", "r3_id", "r3_value"]
    )

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "topology_key": topology.key,
        "topology_title": topology.title,
        "num_nodes": topology.num_nodes,
        "num_resistors": topology.num_resistors,
        "base_resistance_ohm": BASE_R,
        "change_ratio_range": [min_ratio, max_ratio],
        "change_count_ratio": {"0": 0.07, "1": 0.31, "2": 0.31, "3": 0.31},
        "combo_counts": counts,
        "current_source_a": current_a,
        "external_nodes_clockwise": ext_nodes,
        "excitations": excitations,
        "resistor_edges": [list(edge) for edge in topology.resistor_edges],
        "node_coords": [[float(x), float(y)] for x, y in topology.node_coords],
        "seed": seed,
        "float_decimals": float_decimals,
        "note": "Direct solve of Kirchhoff linear systems on target topology. Excitation and measurement both use boundary nodes only.",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    seen_changed = set()
    row_id = 0
    combo_id = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for k in [0, 1, 2, 3]:
            produced = 0
            while produced < counts[k]:
                changes = sample_changes(topology.num_resistors, k, rng, seen_changed, min_ratio, max_ratio)
                values = np.full(topology.num_resistors, BASE_R, dtype=np.float64)
                for rid, new_r in changes:
                    values[rid] = new_r

                gmat = build_conductance(topology.num_nodes, values, topology.resistor_edges)
                voltages = solve_all_excitations(gmat, keep_idx, ext_nodes[0], rhs_matrix, excitations)
                padded = changes + [(-1, 0.0)] * (3 - k)

                for ex_idx, (src, gnd) in enumerate(excitations):
                    row = [row_id, combo_id, src, gnd]
                    row.extend(fmt(voltages[n, ex_idx], float_decimals) for n in ext_nodes)
                    row.append(k)
                    for rid, val in padded[:3]:
                        row.append(rid)
                        row.append(fmt(val, float_decimals))
                    writer.writerow(row)
                    row_id += 1

                combo_id += 1
                produced += 1

    print(f"[{topology.key}] rows={row_id} combos={combo_id} -> {out_path}")
    return out_path, meta_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate clean topology-specific CSV datasets for GNN_EXPAND.")
    parser.add_argument(
        "--topology-keys",
        nargs="*",
        default=list(DEFAULT_TOPOLOGIES),
        help="Topology keys to generate. Defaults to all four stages.",
    )
    parser.add_argument("--output-dir", default="gnn/GNN_EXPAND/data")
    parser.add_argument("--seed", type=int, default=20260402)
    parser.add_argument("--total-combos", type=int, default=10000)
    parser.add_argument("--current-a", type=float, default=0.01)
    parser.add_argument("--min-ratio", type=float, default=0.05)
    parser.add_argument("--max-ratio", type=float, default=0.30)
    parser.add_argument("--float-decimals", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    for idx, topology_key in enumerate(args.topology_keys):
        if topology_key not in TOPOLOGY_REGISTRY:
            raise KeyError(f"Unknown topology key: {topology_key}")
        generate_dataset(
            topology_key=topology_key,
            output_dir=output_dir,
            seed=args.seed + idx,
            total_combos=args.total_combos,
            current_a=args.current_a,
            min_ratio=args.min_ratio,
            max_ratio=args.max_ratio,
            float_decimals=args.float_decimals,
        )


if __name__ == "__main__":
    main()
