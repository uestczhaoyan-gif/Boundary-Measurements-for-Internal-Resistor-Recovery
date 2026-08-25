from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_PLOT = PROJECT_ROOT / ".vendor_plot"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import prepend_vendor_dir

prepend_vendor_dir(VENDOR_PLOT, required_version=(3, 11))

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


THRESHOLDS = [0.98, 0.95, 0.90, 0.85, 0.80, 0.75]


def load_rows(outputs_root: Path) -> list[dict]:
    rows: list[dict] = []
    for train_path in sorted(outputs_root.rglob("train_metrics.json")):
        infer_path = train_path.parent / "inference_metrics.json"
        if not infer_path.exists():
            continue

        train_payload = json.loads(train_path.read_text(encoding="utf-8"))
        infer_payload = json.loads(infer_path.read_text(encoding="utf-8"))
        best_train = train_payload.get("best_train_metrics") or {}
        best_val = train_payload.get("best_val_metrics") or {}

        rows.append(
            {
                "run_dir": train_path.parent.name,
                "N": int(train_payload["grid_size"]),
                "P": int(train_payload["port_count"]),
                "M": int(train_payload["num_resistors"]),
                "K": int(train_payload["k"]),
                "best_epoch": int(train_payload.get("best_epoch") or 0),
                "history_len": int(len(train_payload.get("history") or [])),
                "train_id_exact_rate": float(best_train.get("id_exact_rate") or 0.0),
                "train_value_accuracy": float(best_train.get("value_accuracy") or 0.0),
                "val_id_exact_rate": float(best_val.get("id_exact_rate") or 0.0),
                "val_value_accuracy": float(best_val.get("value_accuracy") or 0.0),
                "test_id_exact_rate": float(infer_payload.get("id_exact_rate") or 0.0),
                "test_value_accuracy": float(infer_payload.get("value_accuracy") or 0.0),
                "test_mae_changed": float(infer_payload.get("mae_changed") or 0.0),
                "test_pass_flag_098": int(bool(infer_payload.get("pass_flag"))),
            }
        )

    rows.sort(key=lambda item: (item["N"], item["K"]))
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_threshold_rows(rows: list[dict], thresholds: list[float]) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["N"], []).append(row)

    threshold_rows: list[dict] = []
    for threshold in thresholds:
        for n, group in sorted(grouped.items()):
            candidates = [
                row
                for row in group
                if row["test_id_exact_rate"] >= threshold and row["test_value_accuracy"] >= 0.90
            ]
            if candidates:
                best = max(candidates, key=lambda item: item["K"])
                threshold_rows.append(
                    {
                        "id_threshold": threshold,
                        "N": n,
                        "P": best["P"],
                        "M": best["M"],
                        "K_max": best["K"],
                        "test_id_exact_rate": best["test_id_exact_rate"],
                        "test_value_accuracy": best["test_value_accuracy"],
                        "run_dir": best["run_dir"],
                    }
                )
            else:
                threshold_rows.append(
                    {
                        "id_threshold": threshold,
                        "N": n,
                        "P": group[0]["P"],
                        "M": group[0]["M"],
                        "K_max": 0,
                        "test_id_exact_rate": "",
                        "test_value_accuracy": "",
                        "run_dir": "",
                    }
                )
    return threshold_rows


def _apply_plot_style() -> None:
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


def plot_metric_dropoff_per_scale(rows: list[dict], figure_dir: Path, name_prefix: str) -> None:
    if plt is None or not rows:
        return

    _apply_plot_style()
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["N"], []).append(row)

    colors = ("#1f77b4", "#ff7f0e")
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6), constrained_layout=True)
    axes = axes.flatten()
    legend_handles = None

    for ax, (n, group) in zip(axes, sorted(grouped.items())):
        ks = [row["K"] for row in group]
        id_vals = [row["test_id_exact_rate"] for row in group]
        value_vals = [row["test_value_accuracy"] for row in group]
        width = 0.36
        x = list(range(len(ks)))

        bars_id = ax.bar([v - width / 2 for v in x], id_vals, width=width, color=colors[0], label="Test ID")
        bars_value = ax.bar([v + width / 2 for v in x], value_vals, width=width, color=colors[1], label="Test Value")
        ax.axhline(0.98, color=colors[0], linestyle=":", linewidth=1.0)
        ax.axhline(0.90, color=colors[1], linestyle=":", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_ylim(0.0, 1.03)
        ax.set_xlabel("K")
        ax.set_ylabel("Metric")
        ax.set_title(f"N={n}, P={group[0]['P']}")
        if legend_handles is None:
            legend_handles = (bars_id[0], bars_value[0])

        single_fig, single_ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
        single_ax.bar([v - width / 2 for v in x], id_vals, width=width, color=colors[0], label="Test ID")
        single_ax.bar([v + width / 2 for v in x], value_vals, width=width, color=colors[1], label="Test Value")
        single_ax.axhline(0.98, color=colors[0], linestyle=":", linewidth=1.0)
        single_ax.axhline(0.90, color=colors[1], linestyle=":", linewidth=1.0)
        single_ax.set_xticks(x)
        single_ax.set_xticklabels([str(k) for k in ks])
        single_ax.set_ylim(0.0, 1.03)
        single_ax.set_xlabel("Changed resistor count K")
        single_ax.set_ylabel("Metric")
        single_ax.set_title(f"{name_prefix} test metrics, N={n}, P={group[0]['P']}")
        single_ax.legend(frameon=False, loc="lower left")
        single_fig.savefig(figure_dir / f"N{n}_P{group[0]['P']}_test_metric_dropoff.png", dpi=220)
        plt.close(single_fig)

    if legend_handles is not None:
        fig.legend(legend_handles, ["Test ID", "Test Value"], frameon=False, loc="upper center", ncol=2)
    fig.savefig(figure_dir / f"{name_prefix}_test_metric_dropoff_overview.png", dpi=220)
    plt.close(fig)


def plot_threshold_compare(threshold_rows: list[dict], output_path: Path) -> None:
    if plt is None or not threshold_rows:
        return

    _apply_plot_style()
    grouped: dict[float, list[dict]] = {}
    for row in threshold_rows:
        grouped.setdefault(float(row["id_threshold"]), []).append(row)

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.3), constrained_layout=True)
    axes = axes.flatten()
    color = "#1f77b4"

    for ax, threshold in zip(axes, THRESHOLDS):
        group = sorted(grouped.get(float(threshold), []), key=lambda item: item["P"])
        ports = [row["P"] for row in group]
        kmax = [row["K_max"] for row in group]
        ax.plot(ports, kmax, marker="o", linewidth=2.0, color=color)
        ax.set_xticks(ports)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("Port count P")
        ax.set_ylabel("K_max")
        ax.set_title(f"ID threshold = {int(round(threshold * 100))}%")

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def summarize(outputs_root: Path, figure_dir: Path, name_prefix: str = "modelg1_subproj1") -> tuple[Path, Path]:
    rows = load_rows(outputs_root)
    if not rows:
        raise SystemExit(f"No paired train/inference metrics found under {outputs_root}")

    summary_csv = outputs_root / f"{name_prefix}_test_summary.csv"
    threshold_csv = outputs_root / f"{name_prefix}_threshold_kmax_summary.csv"

    write_csv(
        summary_csv,
        rows,
        [
            "run_dir",
            "N",
            "P",
            "M",
            "K",
            "best_epoch",
            "history_len",
            "train_id_exact_rate",
            "train_value_accuracy",
            "val_id_exact_rate",
            "val_value_accuracy",
            "test_id_exact_rate",
            "test_value_accuracy",
            "test_mae_changed",
            "test_pass_flag_098",
        ],
    )

    threshold_rows = build_threshold_rows(rows, THRESHOLDS)
    write_csv(
        threshold_csv,
        threshold_rows,
        [
            "id_threshold",
            "N",
            "P",
            "M",
            "K_max",
            "test_id_exact_rate",
            "test_value_accuracy",
            "run_dir",
        ],
    )

    plot_metric_dropoff_per_scale(rows, figure_dir, name_prefix=name_prefix)
    plot_threshold_compare(threshold_rows, figure_dir / f"{name_prefix}_threshold_kmax_compare.png")
    return summary_csv, threshold_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize modelg1 subproject-1 results.")
    parser.add_argument("--outputs-root", default=str(PROJECT_ROOT / "outputs_modelg1"))
    parser.add_argument("--figure-dir", default=str(PROJECT_ROOT / "Figure" / "modelg1_subproj1"))
    parser.add_argument("--name-prefix", default="modelg1_subproj1")
    args = parser.parse_args()

    summary_csv, threshold_csv = summarize(
        Path(args.outputs_root).resolve(),
        Path(args.figure_dir).resolve(),
        name_prefix=str(args.name_prefix),
    )
    print(f"summary_csv={summary_csv}")
    print(f"threshold_csv={threshold_csv}")


if __name__ == "__main__":
    main()
