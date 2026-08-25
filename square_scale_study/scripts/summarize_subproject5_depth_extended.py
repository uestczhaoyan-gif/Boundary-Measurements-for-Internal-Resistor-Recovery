from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Summarize extended subproject-5 depth experiments.")
    parser.add_argument(
        "--multiseed-root",
        default=str(PROJECT_ROOT / "outputs_subproj5_depth_edgebal_multiseed_modelg2"),
    )
    parser.add_argument(
        "--n78-root",
        default=str(PROJECT_ROOT / "outputs_subproj5_depth_edgebal_n78_modelg2"),
    )
    parser.add_argument(
        "--summary-dir",
        default=str(PROJECT_ROOT / "outputs_subproj5_depth_extended"),
    )
    parser.add_argument(
        "--figure-dir",
        default=str(PROJECT_ROOT / "Figure" / "subproject5_depth_extended"),
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


def parse_support(value: str) -> int:
    match = re.search(r"-?\d+", str(value))
    if not match:
        raise ValueError(f"Cannot parse support id from {value!r}")
    return int(match.group(0))


def seed_from_run_name(name: str) -> str:
    match = re.search(r"seed(\d+)", name)
    return match.group(1) if match else "single"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_rows(root: Path, source: str) -> list[dict]:
    rows: list[dict] = []
    for metrics_path in sorted(root.rglob("inference_metrics.json")):
        pred_path = metrics_path.parent / "predictions.csv"
        if not pred_path.exists():
            continue
        metrics = read_json(metrics_path)
        if int(metrics.get("k", 0)) != 1:
            continue
        if str(metrics.get("study_protocol", "")) != "depth_edge_balanced":
            continue

        n = int(metrics["grid_size"])
        topology = build_square_topology(n)
        depths = edge_depths(topology)
        groups: dict[int, list[dict[str, float]]] = defaultdict(list)
        with pred_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rid = parse_support(row["true_support"])
                shell = int(depths[rid])
                groups[shell].append(
                    {
                        "support_exact": float(row["support_exact"]),
                        "mae_changed_sample": float(row["mae_changed_sample"]),
                    }
                )

        for shell, items in sorted(groups.items()):
            sample_count = len(items)
            id_exact = float(np.mean([item["support_exact"] for item in items])) if items else 0.0
            mae = float(np.mean([item["mae_changed_sample"] for item in items])) if items else 0.0
            value = max(0.0, 1.0 - mae / (BASE_R * DEFAULT_CHANGE_LIMIT))
            rows.append(
                {
                    "source": source,
                    "run_dir": metrics_path.parent.name,
                    "seed": seed_from_run_name(metrics_path.parent.name),
                    "N": n,
                    "P": int(metrics["port_count"]),
                    "M": int(metrics["num_resistors"]),
                    "depth_shell": int(shell),
                    "depth_label": depth_label(shell),
                    "sample_count": sample_count,
                    "id_exact_rate": id_exact,
                    "value_accuracy": value,
                    "mae_changed": mae,
                }
            )
    rows.sort(key=lambda item: (int(item["N"]), str(item["seed"]), int(item["depth_shell"])))
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["N"]), int(row["depth_shell"]))].append(row)
    output: list[dict] = []
    for (n, shell), items in sorted(grouped.items()):
        id_vals = np.array([float(item["id_exact_rate"]) for item in items], dtype=float)
        value_vals = np.array([float(item["value_accuracy"]) for item in items], dtype=float)
        mae_vals = np.array([float(item["mae_changed"]) for item in items], dtype=float)
        output.append(
            {
                "N": n,
                "P": int(items[0]["P"]),
                "M": int(items[0]["M"]),
                "depth_shell": shell,
                "depth_label": depth_label(shell),
                "num_runs": len(items),
                "sample_count_mean": float(np.mean([float(item["sample_count"]) for item in items])),
                "id_mean": float(np.mean(id_vals)),
                "id_std": float(np.std(id_vals, ddof=1)) if len(id_vals) > 1 else 0.0,
                "value_mean": float(np.mean(value_vals)),
                "value_std": float(np.std(value_vals, ddof=1)) if len(value_vals) > 1 else 0.0,
                "mae_mean": float(np.mean(mae_vals)),
                "mae_std": float(np.std(mae_vals, ddof=1)) if len(mae_vals) > 1 else 0.0,
            }
        )
    return output


def plot_n6_multiseed(rows: list[dict], figure_dir: Path) -> None:
    if plt is None:
        return
    rows = [row for row in rows if int(row["N"]) == 6]
    if not rows:
        return
    apply_style()
    seeds = sorted({str(row["seed"]) for row in rows})
    shells = sorted({int(row["depth_shell"]) for row in rows})
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), constrained_layout=True)
    x = np.arange(len(seeds))
    for idx, shell in enumerate(shells):
        by_seed = {str(row["seed"]): row for row in rows if int(row["depth_shell"]) == shell}
        id_vals = [float(by_seed[seed]["id_exact_rate"]) for seed in seeds if seed in by_seed]
        value_vals = [float(by_seed[seed]["value_accuracy"]) for seed in seeds if seed in by_seed]
        xs = np.arange(len(id_vals))
        label = depth_label(shell)
        axes[0].plot(xs, id_vals, marker="o", linewidth=2.0, color=colors[idx % len(colors)], label=label)
        axes[1].plot(xs, value_vals, marker="s", linewidth=2.0, color=colors[idx % len(colors)], label=label)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([seed[-2:] for seed in seeds])
        ax.set_xlabel("Training seed suffix")
        ax.legend(frameon=False)
    axes[0].set_title("N=6 K=1: ID stability")
    axes[0].set_ylabel("ID exact rate")
    axes[0].set_ylim(0.90, 1.01)
    axes[1].set_title("N=6 K=1: Value stability")
    axes[1].set_ylabel("Value accuracy")
    axes[1].set_ylim(0.98, 1.002)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "subproject5_n6_multiseed_stability.png", dpi=220)
    plt.close(fig)


def plot_depth_metrics(agg_rows: list[dict], figure_dir: Path, ns: list[int], file_name: str) -> None:
    if plt is None:
        return
    rows = [row for row in agg_rows if int(row["N"]) in ns]
    if not rows:
        return
    apply_style()
    fig, axes = plt.subplots(1, len(ns), figsize=(5.2 * len(ns), 4.6), constrained_layout=True, sharey=True)
    if len(ns) == 1:
        axes = [axes]
    width = 0.36
    for ax, n in zip(axes, ns):
        group = sorted([row for row in rows if int(row["N"]) == n], key=lambda item: int(item["depth_shell"]))
        x = np.arange(len(group))
        id_mean = [float(row["id_mean"]) for row in group]
        value_mean = [float(row["value_mean"]) for row in group]
        id_std = [float(row["id_std"]) for row in group]
        value_std = [float(row["value_std"]) for row in group]
        ax.bar(x - width / 2, id_mean, width=width, yerr=id_std, capsize=3, color="#1f77b4", label="ID exact")
        ax.bar(x + width / 2, value_mean, width=width, yerr=value_std, capsize=3, color="#ff7f0e", label="Value")
        ax.set_xticks(x)
        ax.set_xticklabels([str(row["depth_label"]) for row in group])
        ax.set_ylim(0.0, 1.03)
        ax.set_xlabel("Depth shell")
        ax.set_title(f"N={n}")
    axes[0].set_ylabel("Metric")
    axes[0].legend(frameon=False, loc="lower left")
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / file_name, dpi=220)
    plt.close(fig)


def render_metric_table(agg_rows: list[dict], output_path: Path) -> None:
    if plt is None:
        return
    rows = sorted(agg_rows, key=lambda item: (int(item["N"]), int(item["depth_shell"])))
    table_rows = []
    for row in rows:
        id_text = f"{float(row['id_mean']):.3f}"
        value_text = f"{float(row['value_mean']):.4f}"
        if int(row["num_runs"]) > 1:
            id_text += f" +/- {float(row['id_std']):.3f}"
            value_text += f" +/- {float(row['value_std']):.4f}"
        table_rows.append(
            [
                f"{int(row['N'])}x{int(row['N'])}",
                str(int(row["P"])),
                str(int(row["M"])),
                str(row["depth_label"]),
                str(int(row["num_runs"])),
                str(int(round(float(row["sample_count_mean"])))),
                id_text,
                value_text,
                f"{float(row['mae_mean']):.3f}",
            ]
        )
    headers = ["Topology", "P", "M", "Depth", "Runs", "Samples", "ID exact", "Value", "MAE"]
    fig_h = max(3.5, 0.46 * len(table_rows) + 1.4)
    fig, ax = plt.subplots(figsize=(11.6, fig_h), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1.0, 1.22)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = collect_rows(Path(args.multiseed_root).resolve(), "multiseed")
    rows.extend(collect_rows(Path(args.n78_root).resolve(), "n78_single"))
    agg_rows = aggregate(rows)
    summary_dir = Path(args.summary_dir).resolve()
    figure_dir = Path(args.figure_dir).resolve()

    write_csv(
        summary_dir / "subproject5_depth_extended_per_run_rows.csv",
        rows,
        [
            "source",
            "run_dir",
            "seed",
            "N",
            "P",
            "M",
            "depth_shell",
            "depth_label",
            "sample_count",
            "id_exact_rate",
            "value_accuracy",
            "mae_changed",
        ],
    )
    write_csv(
        summary_dir / "subproject5_depth_extended_aggregate_rows.csv",
        agg_rows,
        [
            "N",
            "P",
            "M",
            "depth_shell",
            "depth_label",
            "num_runs",
            "sample_count_mean",
            "id_mean",
            "id_std",
            "value_mean",
            "value_std",
            "mae_mean",
            "mae_std",
        ],
    )

    plot_n6_multiseed(rows, figure_dir)
    plot_depth_metrics(agg_rows, figure_dir, ns=[6], file_name="subproject5_n6_depth_summary.png")
    plot_depth_metrics(agg_rows, figure_dir, ns=[7, 8], file_name="subproject5_n78_depth_summary.png")
    plot_depth_metrics(agg_rows, figure_dir, ns=[4, 5, 6, 7, 8], file_name="subproject5_n4_to_n8_depth_summary.png")
    render_metric_table(agg_rows, figure_dir / "subproject5_depth_extended_metric_table.png")

    print(f"wrote_csv={summary_dir / 'subproject5_depth_extended_per_run_rows.csv'}")
    print(f"wrote_csv={summary_dir / 'subproject5_depth_extended_aggregate_rows.csv'}")
    print(f"wrote_figure={figure_dir / 'subproject5_n6_multiseed_stability.png'}")
    print(f"wrote_figure={figure_dir / 'subproject5_n6_depth_summary.png'}")
    print(f"wrote_figure={figure_dir / 'subproject5_n78_depth_summary.png'}")
    print(f"wrote_figure={figure_dir / 'subproject5_n4_to_n8_depth_summary.png'}")
    print(f"wrote_figure={figure_dir / 'subproject5_depth_extended_metric_table.png'}")


if __name__ == "__main__":
    main()
