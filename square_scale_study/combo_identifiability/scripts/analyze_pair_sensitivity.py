from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import prepend_vendor_dir

prepend_vendor_dir(PROJECT_ROOT / ".vendor_plot", required_version=(3, 11))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from project_common import (
    BASE_R,
    DEFAULT_CURRENT_A,
    build_boundary_excitations,
    build_conductance,
    build_rhs_matrix,
    build_square_topology,
    edge_depths,
    solve_all_excitations,
)
from scripts.generate_subproject2_varcand_grid import build_candidate_edge_order


SUBPROJECT_ROOT = PROJECT_ROOT / "combo_identifiability"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "SimHei",
                "Microsoft YaHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 10.5,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 9.5,
        }
    )


def boundary_response(topology, resistor_values: np.ndarray, excitations, current_a: float) -> np.ndarray:
    ref_node = topology.boundary_nodes_clockwise[0]
    keep_idx, rhs_matrix = build_rhs_matrix(topology.num_nodes, ref_node, excitations, current_a)
    gmat = build_conductance(topology.num_nodes, resistor_values, topology.resistor_edges)
    voltages = solve_all_excitations(gmat, keep_idx, ref_node, rhs_matrix, excitations)
    boundary_nodes = list(topology.boundary_nodes_clockwise)
    return voltages[boundary_nodes, :].T.astype(np.float64)


def compute_edge_sensitivities(grid_size: int, candidate_ids: list[int], perturb_ratio: float, current_a: float) -> tuple:
    topology = build_square_topology(grid_size)
    excitations = build_boundary_excitations(topology, excitation_count=None)
    base_values = np.full(topology.num_resistors, BASE_R, dtype=np.float64)
    base_response = boundary_response(topology, base_values, excitations, current_a)

    vectors: list[np.ndarray] = []
    for rid in candidate_ids:
        values = base_values.copy()
        values[rid] = BASE_R * (1.0 + perturb_ratio)
        response = boundary_response(topology, values, excitations, current_a)
        # Sensitivity to relative resistance change. Shape: all excitation-boundary voltage observations.
        vectors.append(((response - base_response) / perturb_ratio).reshape(-1))
    return topology, excitations, np.asarray(vectors, dtype=np.float64)


def normalized_dot(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < eps:
        return 0.0
    return float(np.dot(a, b) / denom)


def edge_midpoints(topology) -> np.ndarray:
    coords = np.asarray(topology.node_coords, dtype=np.float64)
    return np.asarray([0.5 * (coords[u] + coords[v]) for u, v in topology.resistor_edges], dtype=np.float64)


def analyze_pairs(grid_size: int, candidate_count: int | None, perturb_ratio: float, current_a: float) -> tuple:
    topology = build_square_topology(grid_size)
    full_order = build_candidate_edge_order(topology)
    if candidate_count is None:
        candidate_count = topology.num_resistors
    if candidate_count > topology.num_resistors:
        raise ValueError(f"candidate_count={candidate_count} exceeds M={topology.num_resistors}")

    candidate_ids = sorted(full_order[:candidate_count])
    topology, excitations, sensitivities = compute_edge_sensitivities(
        grid_size=grid_size,
        candidate_ids=candidate_ids,
        perturb_ratio=perturb_ratio,
        current_a=current_a,
    )
    norms = np.linalg.norm(sensitivities, axis=1)
    max_norm = float(np.max(norms)) if len(norms) else 1.0
    depths = edge_depths(topology)
    mids = edge_midpoints(topology)

    raw_rows: list[dict] = []
    worst_joint_values: list[float] = []
    mean_norm_values: list[float] = []
    for local_i, local_j in combinations(range(len(candidate_ids)), 2):
        edge_i = candidate_ids[local_i]
        edge_j = candidate_ids[local_j]
        s_i = sensitivities[local_i]
        s_j = sensitivities[local_j]
        signed_cos = normalized_dot(s_i, s_j)
        abs_cos = abs(signed_cos)
        same_joint = float(np.linalg.norm(s_i + s_j))
        opposite_joint = float(np.linalg.norm(s_i - s_j))
        worst_joint = min(same_joint, opposite_joint)
        mean_norm = 0.5 * (float(norms[local_i]) + float(norms[local_j]))
        distance = float(np.linalg.norm(mids[edge_i] - mids[edge_j]))
        row = {
            "grid_size": grid_size,
            "port_count": topology.port_count,
            "total_resistors": topology.num_resistors,
            "candidate_count": candidate_count,
            "R": 2,
            "edge_i": edge_i,
            "edge_j": edge_j,
            "edge_i_depth": int(depths[edge_i]),
            "edge_j_depth": int(depths[edge_j]),
            "pair_depth_min": int(min(depths[edge_i], depths[edge_j])),
            "edge_distance": distance,
            "sensitivity_norm_i": float(norms[local_i]),
            "sensitivity_norm_j": float(norms[local_j]),
            "mean_sensitivity_norm": mean_norm,
            "signed_cosine": signed_cos,
            "abs_cosine": abs_cos,
            "same_sign_joint_norm": same_joint,
            "opposite_sign_joint_norm": opposite_joint,
            "worst_case_joint_norm": worst_joint,
        }
        raw_rows.append(row)
        worst_joint_values.append(worst_joint)
        mean_norm_values.append(mean_norm)

    max_worst_joint = max(worst_joint_values) if worst_joint_values else 1.0
    max_mean_norm = max(mean_norm_values) if mean_norm_values else max_norm
    rows: list[dict] = []
    for row in raw_rows:
        joint_score = row["worst_case_joint_norm"] / max_worst_joint if max_worst_joint > 0 else 0.0
        norm_score = row["mean_sensitivity_norm"] / max_mean_norm if max_mean_norm > 0 else 0.0
        dissimilar_score = 1.0 - row["abs_cosine"]
        easy_score = 0.55 * joint_score + 0.25 * norm_score + 0.20 * dissimilar_score
        row["easy_score"] = float(easy_score)
        row["difficulty_score"] = float(1.0 - easy_score)
        rows.append(row)

    rows.sort(key=lambda item: item["easy_score"], reverse=True)
    n_pairs = len(rows)
    for rank, row in enumerate(rows, start=1):
        row["easy_rank"] = rank
        if rank <= max(1, int(0.2 * n_pairs)):
            row["tier"] = "easy"
        elif rank > n_pairs - max(1, int(0.2 * n_pairs)):
            row["tier"] = "hard"
        else:
            row["tier"] = "middle"
    return topology, candidate_ids, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_scatter(path: Path, rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    x = np.asarray([row["abs_cosine"] for row in rows], dtype=float)
    y = np.asarray([row["worst_case_joint_norm"] for row in rows], dtype=float)
    c = np.asarray([row["easy_score"] for row in rows], dtype=float)
    scatter = ax.scatter(x, y, c=c, cmap="viridis", s=22, alpha=0.82, edgecolors="none")
    ax.set_title("两电阻组合物理可辨识性初筛", fontsize=14.5, weight="bold")
    ax.set_xlabel("两条电阻响应相似度 |cos|")
    ax.set_ylabel("最弱组合响应强度")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("易识别评分")
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def draw_pair_examples(path: Path, topology, rows: list[dict], count: int = 4) -> None:
    easy = rows[:count]
    hard = rows[-count:]
    examples = [("易识别组合", easy), ("困难组合", list(reversed(hard)))]
    coords = np.asarray(topology.node_coords, dtype=float)
    segments = [[coords[u], coords[v]] for u, v in topology.resistor_edges]
    boundary = set(topology.boundary_nodes_clockwise)

    fig, axes = plt.subplots(2, count, figsize=(3.0 * count, 5.7), constrained_layout=True)
    for row_idx, (label, group) in enumerate(examples):
        for col_idx, row in enumerate(group):
            ax = axes[row_idx, col_idx]
            pair = {int(row["edge_i"]), int(row["edge_j"])}
            colors = ["#d62728" if rid in pair else "#c7c7c7" for rid in range(topology.num_resistors)]
            widths = [3.4 if rid in pair else 1.15 for rid in range(topology.num_resistors)]
            ax.add_collection(LineCollection(segments, colors=colors, linewidths=widths, capstyle="round"))
            for node, (x, y) in enumerate(coords):
                face = "#ffbf00" if node in boundary else "white"
                radius = 0.035 if node in boundary else 0.025
                ax.add_patch(Circle((x, y), radius, facecolor=face, edgecolor="#333333", linewidth=0.9, zorder=4))
            ax.set_aspect("equal")
            ax.set_xlim(-0.12, 1.12)
            ax.set_ylim(-0.12, 1.12)
            ax.axis("off")
            ax.set_title(f"{label}\n({row['edge_i']}, {row['edge_j']})", fontsize=10.5)
    fig.suptitle("Top 易识别 / 困难两电阻组合示例", fontsize=14.5, weight="bold", y=1.04)
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze pair-level resistor-combination identifiability.")
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--candidate-count", type=int, default=None)
    parser.add_argument("--perturb-ratio", type=float, default=0.01)
    parser.add_argument("--current-a", type=float, default=DEFAULT_CURRENT_A)
    parser.add_argument("--output-root", default=str(SUBPROJECT_ROOT / "outputs"))
    parser.add_argument("--figure-root", default=str(SUBPROJECT_ROOT / "Figure"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    topology, candidate_ids, rows = analyze_pairs(
        grid_size=args.grid_size,
        candidate_count=args.candidate_count,
        perturb_ratio=args.perturb_ratio,
        current_a=args.current_a,
    )
    c_count = len(candidate_ids)
    stem = f"N{args.grid_size}x{args.grid_size}_C{c_count}_R2"
    output_root = Path(args.output_root)
    figure_root = Path(args.figure_root)
    output_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / f"{stem}_pair_sensitivity.csv"
    write_csv(csv_path, rows)
    plot_scatter(figure_root / f"{stem}_pair_sensitivity_scatter.png", rows)
    draw_pair_examples(figure_root / f"{stem}_easy_hard_pair_examples.png", topology, rows)

    print(f"candidate_edges={candidate_ids}")
    print(f"pairs={len(rows)}")
    print(f"csv={csv_path.resolve()}")
    print(f"scatter={(figure_root / f'{stem}_pair_sensitivity_scatter.png').resolve()}")
    print(f"examples={(figure_root / f'{stem}_easy_hard_pair_examples.png').resolve()}")


if __name__ == "__main__":
    main()
