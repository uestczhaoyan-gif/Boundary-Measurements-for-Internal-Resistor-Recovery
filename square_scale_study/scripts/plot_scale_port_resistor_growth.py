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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot resistor/port growth for square grids.")
    parser.add_argument("--n-min", type=int, default=3)
    parser.add_argument("--n-max", type=int, default=8)
    parser.add_argument(
        "--figure-dir",
        default=str(PROJECT_ROOT / "Figure" / "sensitivity_svd"),
    )
    parser.add_argument(
        "--summary-csv",
        default=str(PROJECT_ROOT / "outputs_sensitivity_svd" / "scale_port_resistor_growth.csv"),
    )
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "legend.fontsize": 9.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["N", "node_count", "P", "M", "M_per_P"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = []
    for n in range(args.n_min, args.n_max + 1):
        p = 4 * n - 4
        m = 2 * n * (n - 1)
        rows.append(
            {
                "N": n,
                "node_count": n * n,
                "P": p,
                "M": m,
                "M_per_P": m / p,
            }
        )

    figure_dir = Path(args.figure_dir).resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)
    write_csv(Path(args.summary_csv).resolve(), rows)
    apply_style()

    ns = [row["N"] for row in rows]
    p_vals = [row["P"] for row in rows]
    m_vals = [row["M"] for row in rows]
    ratio_vals = [row["M_per_P"] for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(ns, m_vals, marker="o", linewidth=2.2, color="#1f77b4", label="Resistors M")
    ax.plot(ns, p_vals, marker="s", linewidth=2.2, color="#d62728", label="Ports P")
    ax.set_xlabel("Grid size N")
    ax.set_ylabel("Count")
    ax.set_title("Growth of resistor count and port count")
    ax.set_xticks(ns)
    ax.legend(frameon=False)
    fig.savefig(figure_dir / "scale_resistor_vs_port_growth.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(p_vals, ratio_vals, marker="o", linewidth=2.2, color="#2ca02c")
    ax.set_xlabel("Port count P")
    ax.set_ylabel("M / P")
    ax.set_title("Candidate resistors per port vs port count")
    ax.set_xticks(p_vals)
    for p, ratio in zip(p_vals, ratio_vals):
        ax.annotate(f"{ratio:.1f}", (p, ratio), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)
    fig.savefig(figure_dir / "scale_resistor_per_port_vs_port.png", dpi=220)
    plt.close(fig)

    print(f"wrote_csv={Path(args.summary_csv).resolve()}")
    print(f"wrote_figure={figure_dir / 'scale_resistor_vs_port_growth.png'}")
    print(f"wrote_figure={figure_dir / 'scale_resistor_per_port_vs_port.png'}")


if __name__ == "__main__":
    main()
