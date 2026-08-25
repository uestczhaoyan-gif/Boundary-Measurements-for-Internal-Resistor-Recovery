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
    parser = argparse.ArgumentParser(description="Summarize subproject-5 follow-up depth experiments.")
    parser.add_argument(
        "--multiseed-root",
        default=str(PROJECT_ROOT / "outputs_subproj5_depth_edgebal_multiseed_modelg2"),
    )
    parser.add_argument(
        "--single-root",
        default=str(PROJECT_ROOT / "outputs_subproj5_depth_edgebal_modelg2"),
    )
    parser.add_argument(
        "--physics-csv",
        default=str(PROJECT_ROOT / "outputs_subproj5_physics" / "subproject5_single_edge_physical_metrics.csv"),
    )
    parser.add_argument(
        "--summary-dir",
        default=str(PROJECT_ROOT / "outputs_subproj5_depth_followup"),
    )
    parser.add_argument(
        "--figure-dir",
        default=str(PROJECT_ROOT / "Figure" / "subproject5_depth_followup"),
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
    text = str(value).strip()
    if not text:
        raise ValueError("Empty support field.")
    match = re.search(r"-?\d+", text)
    if not match:
        raise ValueError(f"Cannot parse support id from {value!r}")
    return int(match.group(0))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_from_run_name(name: str) -> str:
    match = re.search(r"seed(\d+)", name)
    return match.group(1) if match else "single"


def collect_run_rows(root: Path, source_label: str) -> list[dict]:
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
        edge_depth = edge_depths(topology)
        groups: dict[int, list[dict[str, float]]] = defaultdict(list)

        with pred_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for pred in reader:
                rid = parse_support(pred["true_support"])
                shell = int(edge_depth[rid])
                groups[shell].append(
                    {
                        "support_exact": float(pred["support_exact"]),
                        "mae_changed_sample": float(pred["mae_changed_sample"]),
                    }
                )

        for shell, items in sorted(groups.items()):
            sample_count = len(items)
            id_exact = float(np.mean([item["support_exact"] for item in items])) if items else 0.0
            mae = float(np.mean([item["mae_changed_sample"] for item in items])) if items else 0.0
            value = max(0.0, 1.0 - mae / (BASE_R * DEFAULT_CHANGE_LIMIT))
            rows.append(
                {
                    "source": source_label,
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
    rows.sort(key=lambda item: (item["source"], item["N"], item["seed"], item["depth_shell"]))
    return rows


def aggregate_run_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source"]), int(row["N"]), int(row["depth_shell"]))].append(row)

    output: list[dict] = []
    for (source, n, shell), items in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0], item[0][2])):
        id_vals = np.array([float(item["id_exact_rate"]) for item in items], dtype=float)
        value_vals = np.array([float(item["value_accuracy"]) for item in items], dtype=float)
        mae_vals = np.array([float(item["mae_changed"]) for item in items], dtype=float)
        output.append(
            {
                "source": source,
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


def collect_physics_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    grouped: dict[tuple[int, int], list[dict[str, float]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped[(int(row["N"]), int(row["depth_shell"]))].append(
                {
                    "sensitivity_l2": float(row["sensitivity_l2"]),
                    "sensitivity_linf": float(row["sensitivity_linf"]),
                    "nearest_abs_corr": float(row["nearest_abs_corr"]),
                    "uniqueness_score": float(row["uniqueness_score"]),
                    "nearest_l2_distance": float(row["nearest_l2_distance"]),
                }
            )
    rows: list[dict] = []
    for (n, shell), items in sorted(grouped.items()):
        rows.append(
            {
                "N": n,
                "depth_shell": shell,
                "depth_label": depth_label(shell),
                "edge_count": len(items),
                "sensitivity_l2_mean": float(np.mean([item["sensitivity_l2"] for item in items])),
                "sensitivity_l2_std": float(np.std([item["sensitivity_l2"] for item in items], ddof=1)) if len(items) > 1 else 0.0,
                "uniqueness_mean": float(np.mean([item["uniqueness_score"] for item in items])),
                "uniqueness_std": float(np.std([item["uniqueness_score"] for item in items], ddof=1)) if len(items) > 1 else 0.0,
                "nearest_corr_mean": float(np.mean([item["nearest_abs_corr"] for item in items])),
                "nearest_l2_distance_mean": float(np.mean([item["nearest_l2_distance"] for item in items])),
            }
        )
    return rows


def plot_multiseed_stability(rows: list[dict], figure_dir: Path) -> None:
    if plt is None:
        return
    rows = [row for row in rows if row["source"] == "multiseed"]
    if not rows:
        return
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True, sharex=False)
    colors = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c"}

    for col, n in enumerate([4, 5]):
        n_rows = [row for row in rows if int(row["N"]) == n]
        shells = sorted({int(row["depth_shell"]) for row in n_rows})
        seeds = sorted({str(row["seed"]) for row in n_rows})
        x = np.arange(len(seeds))
        for shell in shells:
            shell_rows = {str(row["seed"]): row for row in n_rows if int(row["depth_shell"]) == shell}
            id_vals = [float(shell_rows[seed]["id_exact_rate"]) for seed in seeds if seed in shell_rows]
            value_vals = [float(shell_rows[seed]["value_accuracy"]) for seed in seeds if seed in shell_rows]
            xs = np.arange(len(id_vals))
            label = depth_label(shell)
            axes[0, col].plot(xs, id_vals, marker="o", linewidth=2.0, color=colors.get(shell, "#9467bd"), label=label)
            axes[1, col].plot(xs, value_vals, marker="s", linewidth=2.0, color=colors.get(shell, "#9467bd"), label=label)

        axes[0, col].set_title(f"N={n}: ID stability")
        axes[1, col].set_title(f"N={n}: Value stability")
        axes[1, col].set_xticks(x)
        axes[1, col].set_xticklabels([seed[-2:] for seed in seeds])
        axes[0, col].set_ylim(0.85, 1.01)
        axes[1, col].set_ylim(0.98, 1.002)
        axes[1, col].set_xlabel("Training seed suffix")
        axes[0, col].legend(frameon=False, loc="lower left")

    axes[0, 0].set_ylabel("ID exact rate")
    axes[1, 0].set_ylabel("Value accuracy")
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "subproject5_multiseed_stability.png", dpi=220)
    plt.close(fig)


def plot_depth_summary(agg_rows: list[dict], figure_dir: Path) -> None:
    if plt is None:
        return
    rows = [row for row in agg_rows if row["source"] in {"multiseed", "single_n6"}]
    if not rows:
        return
    apply_style()
    ns = sorted({int(row["N"]) for row in rows})
    fig, axes = plt.subplots(1, len(ns), figsize=(5.0 * len(ns), 4.6), constrained_layout=True, sharey=True)
    if len(ns) == 1:
        axes = [axes]
    width = 0.36
    for ax, n in zip(axes, ns):
        group = sorted([row for row in rows if int(row["N"]) == n], key=lambda item: int(item["depth_shell"]))
        x = np.arange(len(group))
        id_mean = [float(row["id_mean"]) for row in group]
        id_std = [float(row["id_std"]) for row in group]
        value_mean = [float(row["value_mean"]) for row in group]
        value_std = [float(row["value_std"]) for row in group]
        ax.bar(x - width / 2, id_mean, width=width, yerr=id_std, capsize=3, color="#1f77b4", label="ID exact")
        ax.bar(x + width / 2, value_mean, width=width, yerr=value_std, capsize=3, color="#ff7f0e", label="Value")
        ax.set_xticks(x)
        ax.set_xticklabels([str(row["depth_label"]) for row in group])
        ax.set_ylim(0.0, 1.03)
        ax.set_title(f"N={n}")
        ax.set_xlabel("Depth shell")
    axes[0].set_ylabel("Metric")
    axes[0].legend(frameon=False, loc="lower left")
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "subproject5_depth_metric_summary.png", dpi=220)
    plt.close(fig)


def plot_physics_depth_summary(physics_rows: list[dict], figure_dir: Path) -> None:
    if plt is None or not physics_rows:
        return
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for idx, n in enumerate(sorted({int(row["N"]) for row in physics_rows})):
        group = sorted([row for row in physics_rows if int(row["N"]) == n], key=lambda item: int(item["depth_shell"]))
        x = [int(row["depth_shell"]) for row in group]
        sensitivity = [float(row["sensitivity_l2_mean"]) for row in group]
        uniqueness = [float(row["uniqueness_mean"]) for row in group]
        color = colors[idx % len(colors)]
        axes[0].plot(x, sensitivity, marker="o", linewidth=2.0, color=color, label=f"N={n}")
        axes[1].plot(x, uniqueness, marker="s", linewidth=2.0, color=color, label=f"N={n}")
    axes[0].set_title("Single-edge response amplitude")
    axes[0].set_xlabel("Depth shell")
    axes[0].set_ylabel("Mean sensitivity L2")
    axes[1].set_title("Single-edge response uniqueness")
    axes[1].set_xlabel("Depth shell")
    axes[1].set_ylabel("Mean uniqueness score")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "subproject5_physics_depth_followup.png", dpi=220)
    plt.close(fig)


def render_experiment_table(agg_rows: list[dict], output_path: Path) -> None:
    if plt is None:
        return
    rows = [row for row in agg_rows if row["source"] in {"multiseed", "single_n6"}]
    if not rows:
        return
    sorted_rows = sorted(rows, key=lambda item: (int(item["N"]), int(item["depth_shell"])))
    table_rows = []
    for row in sorted_rows:
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
                id_text,
                value_text,
                f"{float(row['mae_mean']):.3f}",
            ]
        )
    headers = ["Topology", "P", "M", "Depth", "Runs", "ID exact", "Value", "MAE"]
    fig_h = max(3.2, 0.5 * len(table_rows) + 1.3)
    fig, ax = plt.subplots(figsize=(10.5, fig_h), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    table.scale(1.0, 1.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_physics_table(physics_rows: list[dict], output_path: Path) -> None:
    if plt is None or not physics_rows:
        return
    table_rows = [
        [
            f"{int(row['N'])}x{int(row['N'])}",
            str(row["depth_label"]),
            str(int(row["edge_count"])),
            f"{float(row['sensitivity_l2_mean']):.6f}",
            f"{float(row['uniqueness_mean']):.3f}",
            f"{float(row['nearest_corr_mean']):.3f}",
        ]
        for row in sorted(physics_rows, key=lambda item: (int(item["N"]), int(item["depth_shell"])))
    ]
    headers = ["Topology", "Depth", "Edges", "Mean L2", "Uniqueness", "Nearest corr."]
    fig_h = max(3.2, 0.48 * len(table_rows) + 1.3)
    fig, ax = plt.subplots(figsize=(9.2, fig_h), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    table.scale(1.0, 1.22)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    multiseed_root = Path(args.multiseed_root).resolve()
    single_root = Path(args.single_root).resolve()
    summary_dir = Path(args.summary_dir).resolve()
    figure_dir = Path(args.figure_dir).resolve()

    multiseed_rows = collect_run_rows(multiseed_root, "multiseed")
    single_rows = [row for row in collect_run_rows(single_root, "single_n6") if int(row["N"]) == 6]
    all_rows = multiseed_rows + single_rows
    agg_rows = aggregate_run_rows(all_rows)
    physics_rows = collect_physics_rows(Path(args.physics_csv).resolve())

    write_csv(
        summary_dir / "subproject5_depth_per_run_rows.csv",
        all_rows,
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
        summary_dir / "subproject5_depth_aggregate_rows.csv",
        agg_rows,
        [
            "source",
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
    write_csv(
        summary_dir / "subproject5_physics_depth_rows.csv",
        physics_rows,
        [
            "N",
            "depth_shell",
            "depth_label",
            "edge_count",
            "sensitivity_l2_mean",
            "sensitivity_l2_std",
            "uniqueness_mean",
            "uniqueness_std",
            "nearest_corr_mean",
            "nearest_l2_distance_mean",
        ],
    )

    plot_multiseed_stability(all_rows, figure_dir)
    plot_depth_summary(agg_rows, figure_dir)
    plot_physics_depth_summary(physics_rows, figure_dir)
    render_experiment_table(agg_rows, figure_dir / "subproject5_depth_followup_metric_table.png")
    render_physics_table(physics_rows, figure_dir / "subproject5_physics_followup_metric_table.png")

    print(f"wrote_csv={summary_dir / 'subproject5_depth_per_run_rows.csv'}")
    print(f"wrote_csv={summary_dir / 'subproject5_depth_aggregate_rows.csv'}")
    print(f"wrote_csv={summary_dir / 'subproject5_physics_depth_rows.csv'}")
    print(f"wrote_figure={figure_dir / 'subproject5_multiseed_stability.png'}")
    print(f"wrote_figure={figure_dir / 'subproject5_depth_metric_summary.png'}")
    print(f"wrote_figure={figure_dir / 'subproject5_physics_depth_followup.png'}")
    print(f"wrote_figure={figure_dir / 'subproject5_depth_followup_metric_table.png'}")
    print(f"wrote_figure={figure_dir / 'subproject5_physics_followup_metric_table.png'}")


if __name__ == "__main__":
    main()
