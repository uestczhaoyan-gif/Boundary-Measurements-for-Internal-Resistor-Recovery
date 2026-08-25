from __future__ import annotations

import argparse
import csv
import random
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_common import (
    BASE_R,
    DEFAULT_CHANGE_LIMIT,
    DEFAULT_CURRENT_A,
    build_boundary_excitations_for_nodes,
    build_conductance,
    build_rhs_matrix,
    build_square_topology,
    dump_json,
    fmt_float,
    select_active_boundary_nodes,
    solve_all_excitations,
)

import numpy as np


def dataset_stem(grid_size: int, p_active: int, k: int) -> str:
    return f"square_activeport_N{grid_size}x{grid_size}_Pactive{p_active}_K{k}"


def sample_change_bundle_full(
    num_resistors: int,
    k: int,
    rng: random.Random,
    seen: set[tuple[tuple[int, float], ...]],
    max_ratio: float,
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


def generate_activeport_dataset_bundle(
    grid_size: int,
    p_active: int,
    k: int,
    output_root: Path,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    current_a: float = DEFAULT_CURRENT_A,
    change_limit: float = DEFAULT_CHANGE_LIMIT,
    float_decimals: int = 6,
) -> Path:
    topology = build_square_topology(grid_size)
    full_boundary_nodes = list(topology.boundary_nodes_clockwise)
    if p_active > len(full_boundary_nodes):
        raise ValueError(f"p_active={p_active} exceeds total port count {len(full_boundary_nodes)}")

    active_boundary_nodes = select_active_boundary_nodes(full_boundary_nodes, p_active)
    excitations = build_boundary_excitations_for_nodes(active_boundary_nodes)
    ref_node = active_boundary_nodes[0]
    keep_idx, rhs_matrix = build_rhs_matrix(
        topology.num_nodes,
        ref_node=ref_node,
        excitations=excitations,
        current_a=current_a,
    )
    base_values = np.full(topology.num_resistors, BASE_R, dtype=np.float64)
    base_gmat = build_conductance(topology.num_nodes, base_values, topology.resistor_edges)
    baseline_voltages = solve_all_excitations(base_gmat, keep_idx, ref_node, rhs_matrix, excitations)
    baseline_boundary = baseline_voltages[active_boundary_nodes, :].T.astype(np.float32)

    base_dir = output_root / f"N{grid_size}x{grid_size}"
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = dataset_stem(grid_size, p_active, k)
    split_specs = [("train", train_size), ("val", val_size), ("test", test_size)]
    header = (
        ["row_id", "sample_id", "excitation_idx", "src_node", "gnd_node"]
        + [f"v_node{n}" for n in active_boundary_nodes]
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
                changes = sample_change_bundle_full(topology.num_resistors, k, rng, seen, change_limit)
                resistor_values = np.full(topology.num_resistors, BASE_R, dtype=np.float64)
                for rid, new_r in changes:
                    resistor_values[rid] = new_r
                gmat = build_conductance(topology.num_nodes, resistor_values, topology.resistor_edges)
                voltages = solve_all_excitations(gmat, keep_idx, ref_node, rhs_matrix, excitations)
                for excitation_idx, (src, gnd) in enumerate(excitations):
                    row = [row_id, sample_id, excitation_idx, src, gnd]
                    row.extend(fmt_float(float(voltages[node, excitation_idx]), float_decimals) for node in active_boundary_nodes)
                    row.append(k)
                    for rid, value in changes:
                        row.append(rid)
                        row.append(fmt_float(value, float_decimals))
                    writer.writerow(row)
                    row_id += 1

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "study_protocol": "variable_active_ports",
        "topology": {
            "grid_size": grid_size,
            "num_nodes": topology.num_nodes,
            "num_resistors": topology.num_resistors,
            "port_count": topology.port_count,
            "boundary_nodes_clockwise": full_boundary_nodes,
            "resistor_edges": [list(edge) for edge in topology.resistor_edges],
            "message_edges": [list(edge) for edge in topology.message_edges],
            "node_coords": [[float(x), float(y)] for x, y in topology.node_coords],
        },
        "dataset_stem": stem,
        "k": k,
        "candidate_edge_count": topology.num_resistors,
        "candidate_edge_ids": list(range(topology.num_resistors)),
        "candidate_edge_mask": [1] * topology.num_resistors,
        "active_port_count": p_active,
        "active_boundary_nodes": active_boundary_nodes,
        "measurement_boundary_nodes": active_boundary_nodes,
        "excitation_boundary_nodes": active_boundary_nodes,
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
        "note": "Subproject-3 dataset: full square topology is preserved, all resistors may change, but only active_boundary_nodes participate in both excitation and measurement. Forward responses are generated by direct Kirchhoff solves.",
    }
    meta_path = base_dir / f"{stem}_meta.json"
    dump_json(meta_path, meta)
    return meta_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate subproject-3 variable-active-port datasets.")
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--pactive-list", default="4,6,8,10,12")
    parser.add_argument("--k-list", default="1,2,3,4")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "data_subproj3_activeport"))
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260417)
    parser.add_argument("--seed-stride", type=int, default=97)
    parser.add_argument("--current-a", type=float, default=DEFAULT_CURRENT_A)
    parser.add_argument("--change-limit", type=float, default=DEFAULT_CHANGE_LIMIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    pactive_list = [int(item) for item in args.pactive_list.split(",") if item.strip()]
    k_list = [int(item) for item in args.k_list.split(",") if item.strip()]

    counter = 0
    for p_active in pactive_list:
        for k in k_list:
            seed = args.seed + counter * args.seed_stride
            meta_path = generate_activeport_dataset_bundle(
                grid_size=args.grid_size,
                p_active=p_active,
                k=k,
                output_root=output_root,
                train_size=args.train_size,
                val_size=args.val_size,
                test_size=args.test_size,
                seed=seed,
                current_a=args.current_a,
                change_limit=args.change_limit,
            )
            print(f"Generated dataset meta: {meta_path}")
            counter += 1


if __name__ == "__main__":
    main()
