from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_PLOT = PROJECT_ROOT / ".vendor_plot"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import prepend_vendor_dir
from project_common import BASE_R, DEFAULT_CHANGE_LIMIT, build_square_topology, edge_depths

prepend_vendor_dir(VENDOR_PLOT, required_version=(3, 11))

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize K=2 depth-composition experiments.")
    parser.add_argument(
        "--outputs-root",
        default=str(PROJECT_ROOT / "outputs_subproj5_depth_k2_modelg2"),
    )
    parser.add_argument(
        "--summary-dir",
        default=str(PROJECT_ROOT / "outputs_subproj5_depth_k2_summary"),
    )
    parser.add_argument(
        "--figure-dir",
        default=str(PROJECT_ROOT / "Figure" / "subproject5_depth_k2"),
    )
    return parser.parse_args()


def apply_style() -> None:
    if plt is None:
        return
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
        }
    )


def depth_label(shell: int) -> str:
    return "outer" if int(shell) == 0 else f"inner_d{int(shell)}"


def depth_pair_label(shells: tuple[int, ...]) -> str:
    return "+".join(depth_label(shell) for shell in shells)


def parse_support_list(value: str) -> list[int]:
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, int):
            return [int(parsed)]
        return [int(item) for item in parsed]
    except Exception:
        return [int(item) for item in re.findall(r"-?\d+", text)]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def category_pair_counts(n: int, k: int) -> dict[str, int]:
    topology = build_square_topology(n)
    depths = edge_depths(topology)
    counts: dict[str, int] = defaultdict(int)
    for i in range(len(depths)):
        for j in range(i + 1, len(depths)):
            label = depth_pair_label(tuple(sorted((int(depths[i]), int(depths[j])))))
            counts[label] += 1
    return dict(counts)


def collect_rows(outputs_root: Path) -> list[dict]:
    rows: list[dict] = []
    for metrics_path in sorted(outputs_root.rglob("inference_metrics.json")):
        pred_path = metrics_path.parent / "predictions.csv"
        if not pred_path.exists():
            continue
        metrics = read_json(metrics_path)
        if int(metrics.get("k", 0)) != 2:
            continue
        if str(metrics.get("study_protocol", "")) != "depth_edge_balanced":
            continue

        n = int(metrics["grid_size"])
        topology = build_square_topology(n)
        depths = edge_depths(topology)
        groups: dict[str, list[dict[str, float]]] = defaultdict(list)

        with pred_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for pred in reader:
                support = parse_support_list(pred["true_support"])
                shells = tuple(sorted(int(depths[rid]) for rid in support))
                label = depth_pair_label(shells)
                groups[label].append(
                    {
                        "support_exact": float(pred["support_exact"]),
                        "support_overlap": float(pred["support_overlap"]),
                        "mae_changed_sample": float(pred["mae_changed_sample"]),
                    }
                )

        pair_counts = category_pair_counts(n, 2)
        for label, items in sorted(groups.items()):
            sample_count = len(items)
            mae = float(np.mean([item["mae_changed_sample"] for item in items])) if items else 0.0
            rows.append(
                {
                    "run_dir": metrics_path.parent.name,
                    "N": n,
                    "P": int(metrics["port_count"]),
                    "M": int(metrics["num_resistors"]),
                    "depth_pair_label": label,
                    "candidate_pair_count": int(pair_counts.get(label, 0)),
                    "sample_count": sample_count,
                    "id_exact_rate": float(np.mean([item["support_exact"] for item in items])) if items else 0.0,
                    "id_mean_overlap": float(np.mean([item["support_overlap"] for item in items])) if items else 0.0,
                    "value_accuracy": max(0.0, 1.0 - mae / (BASE_R * DEFAULT_CHANGE_LIMIT)),
                    "mae_changed": mae,
                }
            )
    rows.sort(key=lambda item: (int(item["N"]), str(item["depth_pair_label"])))
    return rows


def plot_k2_category_bars(rows: list[dict], figure_dir: Path) -> None:
    if plt is None or not rows:
        return
    apply_style()
    ns = sorted({int(row["N"]) for row in rows})
    fig, axes = plt.subplots(1, len(ns), figsize=(5.8 * len(ns), 4.8), constrained_layout=True, sharey=True)
    if len(ns) == 1:
        axes = [axes]
    width = 0.28
    for ax, n in zip(axes, ns):
        group = [row for row in rows if int(row["N"]) == n]
        labels = [str(row["depth_pair_label"]) for row in group]
        x = np.arange(len(group))
        ax.bar(x - width, [float(row["id_exact_rate"]) for row in group], width=width, color="#1f77b4", label="Exact")
        ax.bar(x, [float(row["id_mean_overlap"]) for row in group], width=width, color="#2ca02c", label="Overlap")
        ax.bar(x + width, [float(row["value_accuracy"]) for row in group], width=width, color="#ff7f0e", label="Value")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylim(0.0, 1.03)
        ax.set_xlabel("Depth pair")
        ax.set_title(f"N={n}, K=2")
    axes[0].set_ylabel("Metric")
    axes[0].legend(frameon=False, loc="lower left")
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "subproject5_k2_depth_pair_metrics.png", dpi=220)
    plt.close(fig)


def plot_k2_pair_load(rows: list[dict], figure_dir: Path) -> None:
    if plt is None or not rows:
        return
    apply_style()
    ns = sorted({int(row["N"]) for row in rows})
    fig, axes = plt.subplots(1, len(ns), figsize=(5.8 * len(ns), 4.4), constrained_layout=True)
    if len(ns) == 1:
        axes = [axes]
    for ax, n in zip(axes, ns):
        group = [row for row in rows if int(row["N"]) == n]
        labels = [str(row["depth_pair_label"]) for row in group]
        x = np.arange(len(group))
        ax.bar(x, [int(row["candidate_pair_count"]) for row in group], color="#7f7f7f")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel("Candidate pair count")
        ax.set_xlabel("Depth pair")
        ax.set_title(f"N={n}: pair-space size")
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "subproject5_k2_depth_pair_candidate_count.png", dpi=220)
    plt.close(fig)


def render_table(rows: list[dict], output_path: Path) -> None:
    if plt is None or not rows:
        return
    table_rows = [
        [
            f"{int(row['N'])}x{int(row['N'])}",
            str(int(row["P"])),
            str(int(row["M"])),
            str(row["depth_pair_label"]),
            str(int(row["candidate_pair_count"])),
            str(int(row["sample_count"])),
            f"{float(row['id_exact_rate']):.3f}",
            f"{float(row['id_mean_overlap']):.3f}",
            f"{float(row['value_accuracy']):.4f}",
            f"{float(row['mae_changed']):.3f}",
        ]
        for row in rows
    ]
    headers = ["Topology", "P", "M", "Depth pair", "Pairs", "Samples", "Exact", "Overlap", "Value", "MAE"]
    fig_h = max(3.0, 0.5 * len(table_rows) + 1.4)
    fig, ax = plt.subplots(figsize=(12.5, fig_h), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1.0, 1.24)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root).resolve()
    summary_dir = Path(args.summary_dir).resolve()
    figure_dir = Path(args.figure_dir).resolve()
    rows = collect_rows(outputs_root)
    if not rows:
        raise RuntimeError(f"No K=2 depth results found under {outputs_root}")
    write_csv(
        summary_dir / "subproject5_k2_depth_pair_summary.csv",
        rows,
        [
            "run_dir",
            "N",
            "P",
            "M",
            "depth_pair_label",
            "candidate_pair_count",
            "sample_count",
            "id_exact_rate",
            "id_mean_overlap",
            "value_accuracy",
            "mae_changed",
        ],
    )
    plot_k2_category_bars(rows, figure_dir)
    plot_k2_pair_load(rows, figure_dir)
    render_table(rows, figure_dir / "subproject5_k2_depth_pair_metric_table.png")
    print(f"wrote_csv={summary_dir / 'subproject5_k2_depth_pair_summary.csv'}")
    print(f"wrote_figure={figure_dir / 'subproject5_k2_depth_pair_metrics.png'}")
    print(f"wrote_figure={figure_dir / 'subproject5_k2_depth_pair_candidate_count.png'}")
    print(f"wrote_figure={figure_dir / 'subproject5_k2_depth_pair_metric_table.png'}")


if __name__ == "__main__":
    main()
