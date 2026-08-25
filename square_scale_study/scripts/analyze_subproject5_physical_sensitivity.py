from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_PLOT = PROJECT_ROOT / ".vendor_plot"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import prepend_vendor_dir

prepend_vendor_dir(VENDOR_PLOT, required_version=(3, 11))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from project_common import (
    BASE_R,
    DEFAULT_CURRENT_A,
    build_boundary_excitations,
    build_conductance,
    build_rhs_matrix,
    build_square_topology,
    edge_depths,
    fmt_float,
    solve_all_excitations,
)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.labelsize": 10.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def node_xy(topology, node_id: int) -> tuple[float, float]:
    x, y = topology.node_coords[node_id]
    return float(x), float(y)


def solve_stacked_boundary_response(topology, resistor_values: np.ndarray) -> np.ndarray:
    excitations = build_boundary_excitations(topology, excitation_count=None)
    ref_node = topology.boundary_nodes_clockwise[0]
    keep_idx, rhs = build_rhs_matrix(
        topology.num_nodes,
        ref_node=ref_node,
        excitations=excitations,
        current_a=DEFAULT_CURRENT_A,
    )
    gmat = build_conductance(topology.num_nodes, resistor_values, topology.resistor_edges)
    voltages = solve_all_excitations(gmat, keep_idx, ref_node, rhs, excitations)
    boundary_nodes = list(topology.boundary_nodes_clockwise)
    return voltages[boundary_nodes, :].T.reshape(-1)


def single_edge_sensitivity_matrix(grid_size: int, delta_r: float) -> tuple[np.ndarray, np.ndarray]:
    topology = build_square_topology(grid_size)
    base_values = np.full(topology.num_resistors, BASE_R, dtype=np.float64)
    base_stack = solve_stacked_boundary_response(topology, base_values)
    sens = np.zeros((topology.num_resistors, base_stack.shape[0]), dtype=np.float64)

    for rid in range(topology.num_resistors):
        values = base_values.copy()
        values[rid] += delta_r
        stack = solve_stacked_boundary_response(topology, values)
        sens[rid] = (stack - base_stack) / delta_r
    return sens, edge_depths(topology)


def compute_edge_metrics(grid_size: int, delta_r: float) -> list[dict]:
    topology = build_square_topology(grid_size)
    sens, depth = single_edge_sensitivity_matrix(grid_size, delta_r)
    l2_norm = np.linalg.norm(sens, axis=1)
    linf_norm = np.max(np.abs(sens), axis=1)

    normalized = sens / np.maximum(np.linalg.norm(sens, axis=1, keepdims=True), 1e-12)
    corr = np.abs(normalized @ normalized.T)
    np.fill_diagonal(corr, -np.inf)
    nearest_corr = np.max(corr, axis=1)
    nearest_idx = np.argmax(corr, axis=1)

    dist = np.linalg.norm(sens[:, None, :] - sens[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    nearest_l2 = np.min(dist, axis=1)

    rows: list[dict] = []
    for rid, (u, v) in enumerate(topology.resistor_edges):
        rows.append(
            {
                "N": grid_size,
                "P": topology.port_count,
                "M": topology.num_resistors,
                "edge_id": rid,
                "u": u,
                "v": v,
                "depth_shell": int(depth[rid]),
                "sensitivity_l2": float(l2_norm[rid]),
                "sensitivity_linf": float(linf_norm[rid]),
                "nearest_abs_corr": float(nearest_corr[rid]),
                "uniqueness_score": float(1.0 - nearest_corr[rid]),
                "nearest_l2_distance": float(nearest_l2[rid]),
                "nearest_edge_id": int(nearest_idx[rid]),
            }
        )
    return rows


def draw_edge_metric_map(ax, grid_size: int, values: np.ndarray, title: str, cmap: str) -> LineCollection:
    topology = build_square_topology(grid_size)
    segments = []
    widths = []
    vmax = max(float(np.max(values)), 1e-12)
    for rid, (u, v) in enumerate(topology.resistor_edges):
        x1, y1 = node_xy(topology, u)
        x2, y2 = node_xy(topology, v)
        segments.append([(x1, y1), (x2, y2)])
        widths.append(1.2 + 4.2 * float(values[rid] / vmax))

    lc = LineCollection(
        segments,
        cmap=cmap,
        linewidths=widths,
        zorder=2.2,
    )
    lc.set_array(np.asarray(values, dtype=np.float64))
    ax.add_collection(lc)

    boundary_set = set(topology.boundary_nodes_clockwise)
    for node_id in range(topology.num_nodes):
        x, y = node_xy(topology, node_id)
        if node_id in boundary_set:
            ax.scatter(x, y, s=45, facecolor="white", edgecolor="#1f77b4", linewidth=1.3, zorder=4)
        else:
            ax.scatter(x, y, s=28, color="#404040", zorder=4)

    ax.set_aspect("equal")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    return lc


def plot_topology_maps(rows: list[dict], figure_dir: Path) -> None:
    if not rows:
        return
    apply_style()
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["N"]), []).append(row)

    ns = sorted(grouped)
    fig, axes = plt.subplots(2, len(ns), figsize=(4.2 * len(ns), 7.2), constrained_layout=True)
    if len(ns) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    sens_handle = None
    uniq_handle = None
    for col, n in enumerate(ns):
        group = sorted(grouped[n], key=lambda item: item["edge_id"])
        sens = np.asarray([float(row["sensitivity_l2"]) for row in group], dtype=np.float64)
        uniq = np.asarray([float(row["uniqueness_score"]) for row in group], dtype=np.float64)
        sens_handle = draw_edge_metric_map(axes[0, col], n, sens, f"N={n}: sensitivity norm", cmap="YlOrRd")
        uniq_handle = draw_edge_metric_map(axes[1, col], n, uniq, f"N={n}: uniqueness score", cmap="viridis")

    if sens_handle is not None:
        cbar = fig.colorbar(sens_handle, ax=axes[0, :].tolist(), fraction=0.03, pad=0.02)
        cbar.set_label("||sensitivity||_2")
    if uniq_handle is not None:
        cbar = fig.colorbar(uniq_handle, ax=axes[1, :].tolist(), fraction=0.03, pad=0.02)
        cbar.set_label("1 - max |corr|")

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "subproject5_physics_topology_maps.png", dpi=240)
    plt.close(fig)


def plot_depth_summary(rows: list[dict], figure_dir: Path) -> None:
    if not rows:
        return
    apply_style()
    grouped_by_n: dict[int, list[dict]] = {}
    for row in rows:
        grouped_by_n.setdefault(int(row["N"]), []).append(row)

    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)

    for idx, n in enumerate(sorted(grouped_by_n)):
        group = grouped_by_n[n]
        depth_levels = sorted({int(row["depth_shell"]) for row in group})
        sens_means = []
        uniq_means = []
        for d in depth_levels:
            d_rows = [row for row in group if int(row["depth_shell"]) == d]
            sens_means.append(float(np.mean([float(row["sensitivity_l2"]) for row in d_rows])))
            uniq_means.append(float(np.mean([float(row["uniqueness_score"]) for row in d_rows])))
        color = colors[idx % len(colors)]
        axes[0].plot(depth_levels, sens_means, marker="o", linewidth=2.0, color=color, label=f"N={n}")
        axes[1].plot(depth_levels, uniq_means, marker="s", linewidth=2.0, color=color, label=f"N={n}")

    axes[0].set_xlabel("Depth shell")
    axes[0].set_ylabel("Mean sensitivity norm")
    axes[0].set_title("Physical sensitivity by depth")
    axes[1].set_xlabel("Depth shell")
    axes[1].set_ylabel("Mean uniqueness score")
    axes[1].set_title("Response distinctness by depth")
    axes[0].legend(frameon=False, loc="best")
    axes[1].legend(frameon=False, loc="best")

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "subproject5_physics_depth_summary.png", dpi=240)
    plt.close(fig)


def render_metric_table(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    apply_style()
    summary_rows = []
    for n in sorted({int(row["N"]) for row in rows}):
        for depth_shell in sorted({int(row["depth_shell"]) for row in rows if int(row["N"]) == n}):
            d_rows = [row for row in rows if int(row["N"]) == n and int(row["depth_shell"]) == depth_shell]
            summary_rows.append(
                [
                    f"{n}x{n}",
                    str(depth_shell),
                    str(len(d_rows)),
                    fmt_float(float(np.mean([float(row["sensitivity_l2"]) for row in d_rows])), 6),
                    fmt_float(float(np.mean([float(row["uniqueness_score"]) for row in d_rows])), 6),
                ]
            )

    fig_h = max(2.6, 0.55 * len(summary_rows) + 1.3)
    fig, ax = plt.subplots(figsize=(8.4, fig_h), constrained_layout=True)
    ax.axis("off")
    table = ax.table(
        cellText=summary_rows,
        colLabels=["Topology", "Depth", "Edges", "Mean ||sens||_2", "Mean uniqueness"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze single-edge physical sensitivity for subproject-5.")
    parser.add_argument("--grid-list", default="4,5,6")
    parser.add_argument("--delta-r", type=float, default=10.0)
    parser.add_argument("--outputs-root", default=str(PROJECT_ROOT / "outputs_subproj5_physics"))
    parser.add_argument("--figure-dir", default=str(PROJECT_ROOT / "Figure" / "subproject5_physics"))
    args = parser.parse_args()

    grid_list = [int(item) for item in args.grid_list.split(",") if item.strip()]
    outputs_root = Path(args.outputs_root).resolve()
    figure_dir = Path(args.figure_dir).resolve()

    all_rows: list[dict] = []
    for grid_size in grid_list:
        all_rows.extend(compute_edge_metrics(grid_size, args.delta_r))

    write_csv(
        outputs_root / "subproject5_single_edge_physical_metrics.csv",
        all_rows,
        [
            "N",
            "P",
            "M",
            "edge_id",
            "u",
            "v",
            "depth_shell",
            "sensitivity_l2",
            "sensitivity_linf",
            "nearest_abs_corr",
            "uniqueness_score",
            "nearest_l2_distance",
            "nearest_edge_id",
        ],
    )
    plot_topology_maps(all_rows, figure_dir)
    plot_depth_summary(all_rows, figure_dir)
    render_metric_table(all_rows, figure_dir / "subproject5_physics_metric_table.png")

    print(f"wrote_csv={outputs_root / 'subproject5_single_edge_physical_metrics.csv'}")
    print(f"wrote_figure={figure_dir / 'subproject5_physics_topology_maps.png'}")
    print(f"wrote_figure={figure_dir / 'subproject5_physics_depth_summary.png'}")
    print(f"wrote_figure={figure_dir / 'subproject5_physics_metric_table.png'}")


if __name__ == "__main__":
    main()
