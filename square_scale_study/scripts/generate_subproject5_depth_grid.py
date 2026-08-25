from __future__ import annotations

import argparse
import csv
import random
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_common import (
    BASE_R,
    DEFAULT_CHANGE_LIMIT,
    DEFAULT_CURRENT_A,
    build_boundary_excitations,
    build_conductance,
    build_rhs_matrix,
    build_square_topology,
    dump_json,
    edge_depths,
    edge_ids_by_depth,
    fmt_float,
    solve_all_excitations,
)

import numpy as np


def depth_label(depth_shell: int) -> str:
    return "outer" if depth_shell == 0 else f"inner_d{depth_shell}"


def dataset_stem(grid_size: int, k: int, sampling_mode: str) -> str:
    if sampling_mode == "shell":
        return f"square_depth_N{grid_size}x{grid_size}_K{k}"
    if sampling_mode == "edge":
        return f"square_depth_edgebal_N{grid_size}x{grid_size}_K{k}"
    raise ValueError(f"Unsupported sampling_mode={sampling_mode}")


def balanced_shell_schedule(split_size: int, shells: list[int], rng: random.Random) -> list[int]:
    schedule = [shells[idx % len(shells)] for idx in range(split_size)]
    rng.shuffle(schedule)
    return schedule


def balanced_edge_schedule(split_size: int, edge_ids: list[int], rng: random.Random) -> list[int]:
    schedule = [edge_ids[idx % len(edge_ids)] for idx in range(split_size)]
    rng.shuffle(schedule)
    return schedule


def depth_pair_label(pair: tuple[int, ...]) -> str:
    return "+".join(depth_label(shell) for shell in pair)


def all_depth_pair_groups(edge_depth: np.ndarray, k: int) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
    groups: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    for edge_tuple in combinations(range(len(edge_depth)), k):
        depth_tuple = tuple(sorted(int(edge_depth[rid]) for rid in edge_tuple))
        groups.setdefault(depth_tuple, []).append(tuple(int(rid) for rid in edge_tuple))
    return groups


def balanced_category_schedule(
    split_size: int,
    categories: list[tuple[int, ...]],
    rng: random.Random,
) -> list[tuple[int, ...]]:
    schedule = [categories[idx % len(categories)] for idx in range(split_size)]
    rng.shuffle(schedule)
    return schedule


def sample_single_change_for_shell(
    shell_edge_ids: list[int],
    rng: random.Random,
    seen: set[tuple[tuple[int, float], ...]],
    max_ratio: float,
) -> tuple[int, float]:
    while True:
        rid = int(rng.choice(shell_edge_ids))
        ratio = 0.0
        while abs(ratio) < 1e-9:
            ratio = rng.uniform(-max_ratio, max_ratio)
        new_r = round(BASE_R * (1.0 + ratio), 6)
        key = ((rid, new_r),)
        if key not in seen:
            seen.add(key)
            return rid, new_r


def sample_single_change_for_edge(
    rid: int,
    rng: random.Random,
    seen: set[tuple[tuple[int, float], ...]],
    max_ratio: float,
) -> tuple[int, float]:
    while True:
        ratio = 0.0
        while abs(ratio) < 1e-9:
            ratio = rng.uniform(-max_ratio, max_ratio)
        new_r = round(BASE_R * (1.0 + ratio), 6)
        key = ((int(rid), new_r),)
        if key not in seen:
            seen.add(key)
            return int(rid), new_r


def sample_change_values_for_edges(
    edge_ids: tuple[int, ...],
    rng: random.Random,
    seen: set[tuple[tuple[int, float], ...]],
    max_ratio: float,
) -> tuple[list[int], list[float]]:
    ordered_edges = tuple(sorted(int(rid) for rid in edge_ids))
    while True:
        values: list[float] = []
        for _rid in ordered_edges:
            ratio = 0.0
            while abs(ratio) < 1e-9:
                ratio = rng.uniform(-max_ratio, max_ratio)
            values.append(round(BASE_R * (1.0 + ratio), 6))
        key = tuple((int(rid), float(value)) for rid, value in zip(ordered_edges, values))
        if key not in seen:
            seen.add(key)
            return list(ordered_edges), values


def generate_depth_dataset_bundle(
    grid_size: int,
    k: int,
    output_root: Path,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    sampling_mode: str = "shell",
    current_a: float = DEFAULT_CURRENT_A,
    change_limit: float = DEFAULT_CHANGE_LIMIT,
    float_decimals: int = 6,
) -> Path:
    if k not in {1, 2}:
        raise ValueError("Subproject-5 currently supports K=1 and K=2.")

    topology = build_square_topology(grid_size)
    edge_depth = edge_depths(topology)
    depth_groups = edge_ids_by_depth(topology)
    valid_shells = sorted(int(shell) for shell, ids in depth_groups.items() if ids)
    if not valid_shells:
        raise ValueError(f"No valid depth shells found for N={grid_size}")
    all_edge_ids = list(range(topology.num_resistors))
    depth_pair_groups = all_depth_pair_groups(edge_depth, k)
    depth_pair_categories = sorted(depth_pair_groups)

    excitations = build_boundary_excitations(topology, excitation_count=None)
    ref_node = topology.boundary_nodes_clockwise[0]
    keep_idx, rhs_matrix = build_rhs_matrix(topology.num_nodes, ref_node=ref_node, excitations=excitations, current_a=current_a)
    base_values = np.full(topology.num_resistors, BASE_R, dtype=np.float64)
    base_gmat = build_conductance(topology.num_nodes, base_values, topology.resistor_edges)
    baseline_voltages = solve_all_excitations(base_gmat, keep_idx, ref_node, rhs_matrix, excitations)
    boundary_nodes = list(topology.boundary_nodes_clockwise)
    baseline_boundary = baseline_voltages[boundary_nodes, :].T.astype(np.float32)

    base_dir = output_root / f"N{grid_size}x{grid_size}"
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = dataset_stem(grid_size, k, sampling_mode=sampling_mode)
    split_specs = [("train", train_size), ("val", val_size), ("test", test_size)]
    header = (
        ["row_id", "sample_id", "excitation_idx", "src_node", "gnd_node"]
        + [f"v_node{n}" for n in boundary_nodes]
        + [
            "depth_shell",
            "depth_label",
            "balanced_edge_id",
            "depth_pair",
            "depth_pair_label",
            "balanced_edge_ids",
            "change_count",
        ]
        + [item for i in range(1, k + 1) for item in (f"r{i}_id", f"r{i}_value")]
    )

    rng = random.Random(seed)
    seen: set[tuple[tuple[int, float], ...]] = set()
    row_id = 0
    split_paths: dict[str, str] = {}
    split_shell_hist: dict[str, dict[int, int]] = {}
    split_edge_hist: dict[str, dict[int, int]] = {}
    split_depth_pair_hist: dict[str, dict[str, int]] = {}

    for split_name, split_size in split_specs:
        csv_path = base_dir / f"{stem}_{split_name}.csv"
        split_paths[split_name] = csv_path.name
        if k == 1 and sampling_mode == "shell":
            schedule = balanced_shell_schedule(split_size, valid_shells, rng)
        elif k == 1 and sampling_mode == "edge":
            schedule = balanced_edge_schedule(split_size, all_edge_ids, rng)
        elif k == 2:
            schedule = balanced_category_schedule(split_size, depth_pair_categories, rng)
        else:
            raise ValueError(f"Unsupported sampling_mode={sampling_mode}")
        shell_hist = {shell: 0 for shell in valid_shells}
        edge_hist = {edge_id: 0 for edge_id in all_edge_ids}
        pair_hist = {depth_pair_label(category): 0 for category in depth_pair_categories}
        pair_cycle: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
        pair_cursor: dict[tuple[int, ...], int] = {}

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for sample_id, scheduled_item in enumerate(schedule):
                if k == 1 and sampling_mode == "shell":
                    depth_shell = int(scheduled_item)
                    shell_edge_ids = depth_groups[int(depth_shell)]
                    rid, new_r = sample_single_change_for_shell(shell_edge_ids, rng, seen, change_limit)
                    edge_ids = [int(rid)]
                    new_values = [float(new_r)]
                    pair_key = (int(depth_shell),)
                elif k == 1 and sampling_mode == "edge":
                    rid, new_r = sample_single_change_for_edge(int(scheduled_item), rng, seen, change_limit)
                    depth_shell = int(edge_depth[rid])
                    edge_ids = [int(rid)]
                    new_values = [float(new_r)]
                    pair_key = (int(depth_shell),)
                else:
                    pair_key = tuple(int(item) for item in scheduled_item)
                    if pair_key not in pair_cycle or pair_cursor.get(pair_key, 0) >= len(pair_cycle[pair_key]):
                        pair_cycle[pair_key] = list(depth_pair_groups[pair_key])
                        rng.shuffle(pair_cycle[pair_key])
                        pair_cursor[pair_key] = 0
                    edge_tuple = pair_cycle[pair_key][pair_cursor[pair_key]]
                    pair_cursor[pair_key] += 1
                    edge_ids, new_values = sample_change_values_for_edges(edge_tuple, rng, seen, change_limit)
                    depth_shell = -1

                for rid in edge_ids:
                    shell_hist[int(edge_depth[rid])] += 1
                    edge_hist[int(rid)] += 1
                pair_hist[depth_pair_label(pair_key)] += 1

                resistor_values = np.full(topology.num_resistors, BASE_R, dtype=np.float64)
                for rid, new_r in zip(edge_ids, new_values):
                    resistor_values[int(rid)] = float(new_r)
                gmat = build_conductance(topology.num_nodes, resistor_values, topology.resistor_edges)
                voltages = solve_all_excitations(gmat, keep_idx, ref_node, rhs_matrix, excitations)

                for excitation_idx, (src, gnd) in enumerate(excitations):
                    row = [row_id, sample_id, excitation_idx, src, gnd]
                    row.extend(fmt_float(float(voltages[node, excitation_idx]), float_decimals) for node in boundary_nodes)
                    row.append(int(depth_shell))
                    row.append(depth_label(int(depth_shell)) if depth_shell >= 0 else depth_pair_label(pair_key))
                    row.append(int(edge_ids[0]) if len(edge_ids) == 1 else "|".join(str(int(rid)) for rid in edge_ids))
                    row.append("|".join(str(int(shell)) for shell in pair_key))
                    row.append(depth_pair_label(pair_key))
                    row.append("|".join(str(int(rid)) for rid in edge_ids))
                    row.append(k)
                    for rid, new_r in zip(edge_ids, new_values):
                        row.append(int(rid))
                        row.append(fmt_float(float(new_r), float_decimals))
                    writer.writerow(row)
                    row_id += 1

        split_shell_hist[split_name] = shell_hist
        split_edge_hist[split_name] = edge_hist
        split_depth_pair_hist[split_name] = pair_hist

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "study_protocol": "depth_balanced" if sampling_mode == "shell" else "depth_edge_balanced",
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
        "depth_shells": valid_shells,
        "depth_labels": {str(shell): depth_label(shell) for shell in valid_shells},
        "edge_depths": edge_depth.astype(int).tolist(),
        "edge_ids_by_depth": {str(shell): [int(idx) for idx in depth_groups[shell]] for shell in valid_shells},
        "depth_pair_categories": {
            "|".join(str(int(shell)) for shell in category): depth_pair_label(category)
            for category in depth_pair_categories
        },
        "depth_pair_counts": {
            depth_pair_label(category): len(pairs)
            for category, pairs in depth_pair_groups.items()
        },
        "depth_sampling_mode": (
            "balanced_mixed_single_change"
            if k == 1 and sampling_mode == "shell"
            else "balanced_per_edge_single_change"
            if k == 1
            else "balanced_depth_pair_categories_with_pair_cycle"
        ),
        "split_depth_histograms": {
            split: {str(shell): int(count) for shell, count in hist.items()}
            for split, hist in split_shell_hist.items()
        },
        "split_edge_histograms": {
            split: {str(edge_id): int(count) for edge_id, count in hist.items()}
            for split, hist in split_edge_hist.items()
        },
        "split_depth_pair_histograms": split_depth_pair_hist,
        "measurement_boundary_nodes": boundary_nodes,
        "active_boundary_nodes": boundary_nodes,
        "note": (
            "Subproject-5 dataset: fixed square topology with boundary-only excitation and measurement. "
            f"Each sample contains exactly {k} changed resistor(s). "
            + ("Train/val/test are balanced across depth-pair categories for K=2." if k == 2 else "")
            + ("Train/val/test are balanced across depth shells." if k == 1 and sampling_mode == "shell" else "")
            + ("Train/val/test are balanced across concrete resistor edges so each edge receives the same exposure." if k == 1 and sampling_mode == "edge" else "")
        ),
    }
    meta_path = base_dir / f"{stem}_meta.json"
    dump_json(meta_path, meta)
    return meta_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate subproject-5 depth-balanced datasets.")
    parser.add_argument("--grid-list", default="4,5")
    parser.add_argument("--k-list", default="1")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "data_subproj5_depth"))
    parser.add_argument("--sampling-mode", choices=["shell", "edge"], default="shell")
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260419)
    parser.add_argument("--seed-stride", type=int, default=131)
    parser.add_argument("--current-a", type=float, default=DEFAULT_CURRENT_A)
    parser.add_argument("--change-limit", type=float, default=DEFAULT_CHANGE_LIMIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    grid_list = [int(item) for item in args.grid_list.split(",") if item.strip()]
    k_list = [int(item) for item in args.k_list.split(",") if item.strip()]

    counter = 0
    for grid_size in grid_list:
        for k in k_list:
            seed = args.seed + counter * args.seed_stride
            meta_path = generate_depth_dataset_bundle(
                grid_size=grid_size,
                k=k,
                output_root=output_root,
                train_size=args.train_size,
                val_size=args.val_size,
                test_size=args.test_size,
                seed=seed,
                sampling_mode=args.sampling_mode,
                current_a=args.current_a,
                change_limit=args.change_limit,
            )
            print(f"generated_meta={meta_path}")
            counter += 1


if __name__ == "__main__":
    main()
