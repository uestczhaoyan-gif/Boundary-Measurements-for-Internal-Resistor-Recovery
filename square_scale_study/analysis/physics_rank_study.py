from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
VENDOR_DIR = WORKSPACE_ROOT / ".vendor_torchpy311"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import prepend_vendor_dir

prepend_vendor_dir(VENDOR_DIR, required_version=(3, 11))

import numpy as np

from project_common import (
    BASE_R,
    DEFAULT_CHANGE_LIMIT,
    DEFAULT_CURRENT_A,
    build_boundary_excitations,
    build_conductance,
    build_rhs_matrix,
    build_square_topology,
    solve_all_excitations,
)


def effective_rank(singular_values: np.ndarray) -> float:
    if singular_values.size == 0 or singular_values[0] <= 0:
        return 0.0
    probs = singular_values / max(singular_values.sum(), 1e-12)
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
    return float(math.exp(entropy))


def build_response_matrix(grid_size: int, excitation_count: int | None, delta_ratio: float, current_a: float) -> tuple[np.ndarray, dict]:
    topology = build_square_topology(grid_size)
    excitations = build_boundary_excitations(topology, excitation_count=excitation_count)
    ref_node = topology.boundary_nodes_clockwise[0]
    keep_idx, rhs_matrix = build_rhs_matrix(topology.num_nodes, ref_node, excitations, current_a)
    base_values = np.full(topology.num_resistors, BASE_R, dtype=np.float64)
    base_gmat = build_conductance(topology.num_nodes, base_values, topology.resistor_edges)
    base_voltages = solve_all_excitations(base_gmat, keep_idx, ref_node, rhs_matrix, excitations)
    boundary_nodes = list(topology.boundary_nodes_clockwise)
    base_boundary = base_voltages[boundary_nodes, :].T.reshape(-1)

    columns = []
    for rid in range(topology.num_resistors):
        changed = base_values.copy()
        changed[rid] = BASE_R * (1.0 + delta_ratio)
        gmat = build_conductance(topology.num_nodes, changed, topology.resistor_edges)
        voltages = solve_all_excitations(gmat, keep_idx, ref_node, rhs_matrix, excitations)
        delta = voltages[boundary_nodes, :].T.reshape(-1) - base_boundary
        columns.append(delta)
    matrix = np.stack(columns, axis=1)
    info = {
        "grid_size": grid_size,
        "port_count": topology.port_count,
        "num_resistors": topology.num_resistors,
        "excitation_count": len(excitations),
    }
    return matrix.astype(np.float64), info


def analyze_excitation_counts(
    grid_size: int,
    excitation_counts: list[int | None],
    delta_ratio: float = DEFAULT_CHANGE_LIMIT,
    current_a: float = DEFAULT_CURRENT_A,
) -> list[dict]:
    rows: list[dict] = []
    for excitation_count in excitation_counts:
        matrix, info = build_response_matrix(grid_size, excitation_count, delta_ratio=delta_ratio, current_a=current_a)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        rank = int(np.linalg.matrix_rank(matrix))
        cond = float("inf")
        if singular_values.size > 1 and singular_values[-1] > 1e-12:
            cond = float(singular_values[0] / singular_values[-1])
        norms = np.linalg.norm(matrix, axis=0)
        normalized = matrix / np.clip(norms, 1e-12, None)
        coherence = float(np.max(np.abs(normalized.T @ normalized - np.eye(normalized.shape[1]))))
        rows.append(
            {
                "N": grid_size,
                "P": info["port_count"],
                "M": info["num_resistors"],
                "excitation_count": info["excitation_count"],
                "rank": rank,
                "effective_rank": effective_rank(singular_values),
                "condition_number": cond,
                "max_coherence": coherence,
                "top_singular_value": float(singular_values[0]) if singular_values.size else 0.0,
                "tail_singular_value": float(singular_values[-1]) if singular_values.size else 0.0,
            }
        )
    return rows


def parse_excitation_counts(raw_values: list[str]) -> list[int | None]:
    counts: list[int | None] = []
    for raw in raw_values:
        if raw.lower() == "full":
            counts.append(None)
        else:
            counts.append(int(raw))
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study whether more excitations increase effective information channels.")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--excitation-counts", nargs="+", default=["1", "2", "4", "8", "full"])
    parser.add_argument("--delta-ratio", type=float, default=DEFAULT_CHANGE_LIMIT)
    parser.add_argument("--current-a", type=float, default=DEFAULT_CURRENT_A)
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = parse_excitation_counts(args.excitation_counts)
    rows = analyze_excitation_counts(args.n, counts, delta_ratio=args.delta_ratio, current_a=args.current_a)
    output_csv = Path(args.output_csv) if args.output_csv else PROJECT_ROOT / "outputs" / f"physics_rank_N{args.n}x{args.n}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "N",
                "P",
                "M",
                "excitation_count",
                "rank",
                "effective_rank",
                "condition_number",
                "max_coherence",
                "top_singular_value",
                "tail_singular_value",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved physics-rank summary to {output_csv}")


if __name__ == "__main__":
    main()
