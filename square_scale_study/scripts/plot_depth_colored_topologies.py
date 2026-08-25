from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_PLOT = PROJECT_ROOT / ".vendor_plot"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import prepend_vendor_dir
from project_common import build_square_topology, edge_depths

prepend_vendor_dir(VENDOR_PLOT, required_version=(3, 11))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot square-grid resistor edges colored by depth shell.")
    parser.add_argument("--grid-list", default="5,6,7,8")
    parser.add_argument(
        "--figure-path",
        default=str(PROJECT_ROOT / "Figure" / "subproject5_depth_extended" / "topology_depth_colored_N5_to_N8.png"),
    )
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def depth_name(depth: int) -> str:
    return "outer" if int(depth) == 0 else f"inner_d{int(depth)}"


def plot_one(ax, grid_size: int, colors: dict[int, str]) -> set[int]:
    topology = build_square_topology(grid_size)
    depths = edge_depths(topology)
    used_depths = set(int(item) for item in depths.tolist())

    for edge_id, (u, v) in enumerate(topology.resistor_edges):
        x1, y1 = topology.node_coords[u]
        x2, y2 = topology.node_coords[v]
        depth = int(depths[edge_id])
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=colors[depth],
            linewidth=3.0,
            solid_capstyle="round",
            zorder=2,
        )

    boundary = set(topology.boundary_nodes_clockwise)
    for node_id, (x, y) in enumerate(topology.node_coords):
        if node_id in boundary:
            ax.scatter(x, y, s=32, facecolor="white", edgecolor="#222222", linewidth=1.0, zorder=4)
        else:
            ax.scatter(x, y, s=22, facecolor="#222222", edgecolor="#222222", linewidth=0.8, zorder=4)

    ax.set_aspect("equal")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"N={grid_size}x{grid_size}\nP={topology.port_count}, M={topology.num_resistors}", pad=8)
    return used_depths


def main() -> None:
    args = parse_args()
    grid_list = [int(item) for item in args.grid_list.split(",") if item.strip()]
    figure_path = Path(args.figure_path).resolve()
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    apply_style()
    colors = {
        0: "#1f77b4",
        1: "#ff7f0e",
        2: "#2ca02c",
        3: "#d62728",
        4: "#9467bd",
    }

    fig, axes = plt.subplots(1, len(grid_list), figsize=(4.2 * len(grid_list), 4.4), constrained_layout=True)
    if len(grid_list) == 1:
        axes = [axes]

    used_depths: set[int] = set()
    for ax, grid_size in zip(axes, grid_list):
        used_depths |= plot_one(ax, grid_size, colors)

    handles = [
        Line2D([0], [0], color=colors[depth], linewidth=3.0, label=depth_name(depth))
        for depth in sorted(used_depths)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(figure_path, dpi=240, bbox_inches="tight")
    print(f"wrote_figure={figure_path}")


if __name__ == "__main__":
    main()
