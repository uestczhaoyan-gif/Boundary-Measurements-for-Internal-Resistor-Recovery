from __future__ import annotations

import sys
from pathlib import Path


EXPAND_ROOT = Path(__file__).resolve().parent
FIGURE_DIR = EXPAND_ROOT / "Figure"
VENDOR_PLOT = EXPAND_ROOT / ".vendor_plot"
COMMON_DIR = EXPAND_ROOT / "common"

if VENDOR_PLOT.exists() and str(VENDOR_PLOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_PLOT))
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from topologies import TOPOLOGY_REGISTRY


PLOT_ORDER = [
    "square_10x10",
    "rect_6x10",
    "honeycomb_63",
    "circlecut_69",
]

STAGE_LABELS = {
    "square_10x10": "Stage1 Square 10x10",
    "rect_6x10": "Stage2 Rect 6x10",
    "honeycomb_63": "Stage3 Honeycomb 63",
    "circlecut_69": "Stage4 Circle-Cut 69",
}

EDGE_COLOR = "#3A3A3A"
NODE_FACE = "#F7F7F7"
NODE_EDGE = "#303030"
BOUNDARY_FACE = "#D62828"
BOUNDARY_EDGE = "#7F0000"


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
        }
    )


def draw_single_topology(topology_key: str):
    spec = TOPOLOGY_REGISTRY[topology_key]
    coords = np.asarray(spec.node_coords, dtype=np.float64)
    boundary_ids = np.asarray(spec.boundary_nodes_clockwise, dtype=np.int64)
    boundary_mask = np.zeros(spec.num_nodes, dtype=bool)
    boundary_mask[boundary_ids] = True

    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for u, v in spec.resistor_edges:
        ax.plot(
            [coords[u, 0], coords[v, 0]],
            [coords[u, 1], coords[v, 1]],
            color=EDGE_COLOR,
            linewidth=1.1,
            alpha=0.95,
            zorder=1,
        )

    inner_coords = coords[~boundary_mask]
    boundary_coords = coords[boundary_mask]
    if len(inner_coords):
        ax.scatter(
            inner_coords[:, 0],
            inner_coords[:, 1],
            s=28,
            facecolors=NODE_FACE,
            edgecolors=NODE_EDGE,
            linewidths=0.8,
            zorder=2,
        )
    if len(boundary_coords):
        ax.scatter(
            boundary_coords[:, 0],
            boundary_coords[:, 1],
            s=38,
            facecolors=BOUNDARY_FACE,
            edgecolors=BOUNDARY_EDGE,
            linewidths=0.9,
            zorder=3,
        )

    ax.set_title(
        f"{STAGE_LABELS[topology_key]}\n"
        f"{spec.num_nodes} nodes, {spec.num_resistors} resistors, {spec.num_boundary_nodes} boundary nodes",
        pad=10,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    x_pad = 0.06
    y_pad = 0.06
    ax.set_xlim(coords[:, 0].min() - x_pad, coords[:, 0].max() + x_pad)
    ax.set_ylim(coords[:, 1].min() - y_pad, coords[:, 1].max() + y_pad)

    png_path = FIGURE_DIR / f"topology_{topology_key}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved topology figure to {png_path}")


def main():
    configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for topology_key in PLOT_ORDER:
        draw_single_topology(topology_key)


if __name__ == "__main__":
    main()
