from __future__ import annotations

import argparse
import csv
import json
import re
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


THRESHOLDS = [0.95, 0.90, 0.85, 0.80]


def parse_excitation_count(train_payload: dict, infer_payload: dict, run_dir: str) -> int:
    for payload in (infer_payload, train_payload):
        value = payload.get("excitation_count")
        if value not in (None, "", 0, "0"):
            return int(value)

    candidates = [
        str(infer_payload.get("dataset_stem", "")),
        str(train_payload.get("dataset_stem", "")),
        str(infer_payload.get("meta_path", "")),
        str(train_payload.get("meta_path", "")),
        str(run_dir),
    ]
    for text in candidates:
        match = re.search(r"_E(\d+)_K", text)
        if match:
            return int(match.group(1))
        match = re.search(r"\bE(\d+)_K", text)
        if match:
            return int(match.group(1))
    return 0


def load_rows(outputs_root: Path) -> list[dict]:
    rows: list[dict] = []
    for train_path in sorted(outputs_root.rglob("train_metrics.json")):
        infer_path = train_path.parent / "inference_metrics.json"
        if not infer_path.exists():
            continue
        train_payload = json.loads(train_path.read_text(encoding="utf-8"))
        infer_payload = json.loads(infer_path.read_text(encoding="utf-8"))
        if str(infer_payload.get("study_protocol", "")) != "excitation_ablation":
            continue

        best_train = train_payload.get("best_train_metrics") or {}
        best_val = train_payload.get("best_val_metrics") or {}
        excitation_count = parse_excitation_count(train_payload, infer_payload, train_path.parent.name)

        rows.append(
            {
                "run_dir": train_path.parent.name,
                "N": int(train_payload["grid_size"]),
                "P": int(train_payload["port_count"]),
                "M": int(train_payload["num_resistors"]),
                "E": excitation_count,
                "K": int(train_payload["k"]),
                "best_epoch": int(train_payload.get("best_epoch") or 0),
                "train_id_exact_rate": float(best_train.get("id_exact_rate") or 0.0),
                "train_value_accuracy": float(best_train.get("value_accuracy") or 0.0),
                "val_id_exact_rate": float(best_val.get("id_exact_rate") or 0.0),
                "val_value_accuracy": float(best_val.get("value_accuracy") or 0.0),
                "test_id_exact_rate": float(infer_payload.get("id_exact_rate") or 0.0),
                "test_value_accuracy": float(infer_payload.get("value_accuracy") or 0.0),
                "test_mae_changed": float(infer_payload.get("mae_changed") or 0.0),
                "test_pass_flag_090": int(
                    float(infer_payload.get("id_exact_rate") or 0.0) >= 0.90
                    and float(infer_payload.get("value_accuracy") or 0.0) >= 0.90
                ),
            }
        )
    rows.sort(key=lambda item: (item["E"], item["K"]))
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_threshold_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["E"], []).append(row)

    threshold_rows: list[dict] = []
    for threshold in THRESHOLDS:
        for excitation_count, group in sorted(grouped.items()):
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
                        "N": best["N"],
                        "P": best["P"],
                        "M": best["M"],
                        "E": excitation_count,
                        "K_max": best["K"],
                        "test_id_exact_rate": best["test_id_exact_rate"],
                        "test_value_accuracy": best["test_value_accuracy"],
                        "run_dir": best["run_dir"],
                    }
                )
            else:
                template = group[0]
                threshold_rows.append(
                    {
                        "id_threshold": threshold,
                        "N": template["N"],
                        "P": template["P"],
                        "M": template["M"],
                        "E": excitation_count,
                        "K_max": 0,
                        "test_id_exact_rate": "",
                        "test_value_accuracy": "",
                        "run_dir": "",
                    }
                )
    return threshold_rows


def _apply_style() -> None:
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


def plot_metric_dropoff(rows: list[dict], figure_dir: Path) -> None:
    if plt is None or not rows:
        return
    _apply_style()
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["E"], []).append(row)

    items = sorted(grouped.items())
    fig, axes = plt.subplots(1, len(items), figsize=(4.2 * len(items), 3.8), constrained_layout=True)
    if len(items) == 1:
        axes = [axes]

    legend_handles = None
    for ax, (excitation_count, group) in zip(axes, items):
        ks = [row["K"] for row in group]
        id_vals = [row["test_id_exact_rate"] for row in group]
        value_vals = [row["test_value_accuracy"] for row in group]
        width = 0.36
        x = list(range(len(ks)))
        bars_id = ax.bar([v - width / 2 for v in x], id_vals, width=width, color="#1f77b4", label="Test ID")
        bars_value = ax.bar([v + width / 2 for v in x], value_vals, width=width, color="#ff7f0e", label="Test Value")
        if legend_handles is None:
            legend_handles = (bars_id[0], bars_value[0])
        ax.axhline(0.90, color="#7f7f7f", linestyle=":", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_ylim(0.0, 1.03)
        ax.set_xlabel("K")
        ax.set_ylabel("Metric")
        ax.set_title(f"E = {excitation_count}")

    if legend_handles is not None:
        fig.legend(legend_handles, ["Test ID", "Test Value"], frameon=False, loc="upper center", ncol=2)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "subproject4_excitation_test_metric_dropoff_overview.png", dpi=220)
    plt.close(fig)


def plot_curve_compare(rows: list[dict], figure_dir: Path) -> None:
    if plt is None or not rows:
        return
    _apply_style()
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["E"], []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (excitation_count, group) in enumerate(sorted(grouped.items())):
        ks = [row["K"] for row in group]
        id_vals = [row["test_id_exact_rate"] for row in group]
        value_vals = [row["test_value_accuracy"] for row in group]
        color = colors[idx % len(colors)]
        axes[0].plot(ks, id_vals, marker="o", linewidth=2.0, color=color, label=f"E={excitation_count}")
        axes[1].plot(ks, value_vals, marker="s", linewidth=2.0, color=color, label=f"E={excitation_count}")

    axes[0].axhline(0.90, color="#7f7f7f", linestyle=":", linewidth=1.0)
    axes[1].axhline(0.90, color="#7f7f7f", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("ID exact rate")
    axes[0].set_ylim(0.0, 1.01)
    axes[0].set_title("ID exact rate vs K")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Value accuracy")
    axes[1].set_ylim(0.0, 1.01)
    axes[1].set_title("Value accuracy vs K")
    axes[0].legend(frameon=False, loc="lower left")

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "subproject4_excitation_curve_compare.png", dpi=220)
    plt.close(fig)


def plot_metric_by_k(rows: list[dict], figure_dir: Path) -> None:
    if plt is None or not rows:
        return
    _apply_style()
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["K"], []).append(row)

    ks = sorted(grouped)
    ncols = 3
    nrows = (len(ks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.2, 3.7 * nrows), constrained_layout=True)
    try:
        axes = axes.flatten()
    except Exception:
        axes = [axes]

    for ax, k in zip(axes, ks):
        group = sorted(grouped[k], key=lambda item: item["E"])
        x = [row["E"] for row in group]
        id_vals = [row["test_id_exact_rate"] for row in group]
        value_vals = [row["test_value_accuracy"] for row in group]

        ax.plot(x, id_vals, marker="o", linewidth=2.0, color="#1f77b4", label="Test ID")
        ax.plot(x, value_vals, marker="s", linewidth=2.0, color="#ff7f0e", label="Test Value")
        ax.set_xticks(x)
        ax.set_ylim(0.0, 1.01)
        ax.set_xlabel("Excitation count E")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"K = {k}")

    for ax in axes[len(ks):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "subproject4_excitation_accuracy_by_k_overview.png", dpi=220)
    plt.close(fig)


def plot_threshold_compare(threshold_rows: list[dict], output_path: Path) -> None:
    if plt is None or not threshold_rows:
        return
    _apply_style()
    grouped: dict[float, list[dict]] = {}
    for row in threshold_rows:
        grouped.setdefault(float(row["id_threshold"]), []).append(row)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), constrained_layout=True)
    axes = axes.flatten()
    for ax, threshold in zip(axes, THRESHOLDS):
        group = sorted(grouped.get(float(threshold), []), key=lambda item: item["E"])
        x = [row["E"] for row in group]
        y = [row["K_max"] for row in group]
        ax.plot(x, y, marker="o", linewidth=2.0, color="#1f77b4")
        ax.set_xticks(x)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("Excitation count E")
        ax.set_ylabel("K_max")
        ax.set_title(f"ID threshold = {int(round(threshold * 100))}%")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_normalized_capacity(threshold_rows: list[dict], output_path: Path) -> None:
    if plt is None or not threshold_rows:
        return
    _apply_style()
    grouped: dict[float, list[dict]] = {}
    for row in threshold_rows:
        grouped.setdefault(float(row["id_threshold"]), []).append(row)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), constrained_layout=True)
    axes = axes.flatten()
    for ax, threshold in zip(axes, THRESHOLDS):
        group = sorted(grouped.get(float(threshold), []), key=lambda item: item["E"])
        x = [row["E"] for row in group]
        y = [
            (float(row["K_max"]) / float(row["E"])) if float(row["E"]) > 0 else 0.0
            for row in group
        ]
        ax.plot(x, y, marker="o", linewidth=2.0, color="#2ca02c")
        ax.set_xticks(x)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("Available excitation ports Pa")
        ax.set_ylabel("K_max / Pa")
        ax.set_title(f"ID threshold = {int(round(threshold * 100))}%")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_metric_table_rows(rows: list[dict]) -> tuple[list[dict], list[str]]:
    grouped: dict[int, dict[int, dict]] = {}
    ks = sorted({row["K"] for row in rows})
    for row in rows:
        grouped.setdefault(row["E"], {})[row["K"]] = row

    fieldnames = ["E"]
    for k in ks:
        fieldnames.extend([f"K{k}_ID", f"K{k}_Value"])

    table_rows: list[dict] = []
    for excitation_count in sorted(grouped):
        row_out: dict[str, object] = {"E": excitation_count}
        for k in ks:
            item = grouped[excitation_count].get(k)
            row_out[f"K{k}_ID"] = "" if item is None else f"{item['test_id_exact_rate']:.3f}"
            row_out[f"K{k}_Value"] = "" if item is None else f"{item['test_value_accuracy']:.3f}"
        table_rows.append(row_out)
    return table_rows, fieldnames


def render_metric_table(table_rows: list[dict], fieldnames: list[str], output_png: Path) -> None:
    if plt is None or not table_rows:
        return

    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(13.0, 2.3 + 0.52 * len(table_rows)), constrained_layout=True)
    ax.axis("off")

    cell_text = [[str(row.get(field, "")) for field in fieldnames] for row in table_rows]
    table = ax.table(
        cellText=cell_text,
        colLabels=fieldnames,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.3)
    ax.set_title("subproject4 test metric table", fontsize=12, pad=10)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize(outputs_root: Path, figure_dir: Path) -> tuple[Path, Path]:
    rows = load_rows(outputs_root)
    if not rows:
        raise SystemExit(f"No subproject4 train/inference pairs found under {outputs_root}")

    summary_csv = outputs_root / "subproject4_excitation_test_summary.csv"
    threshold_csv = outputs_root / "subproject4_excitation_threshold_kmax_summary.csv"
    write_csv(
        summary_csv,
        rows,
        [
            "run_dir",
            "N",
            "P",
            "M",
            "E",
            "K",
            "best_epoch",
            "train_id_exact_rate",
            "train_value_accuracy",
            "val_id_exact_rate",
            "val_value_accuracy",
            "test_id_exact_rate",
            "test_value_accuracy",
            "test_mae_changed",
            "test_pass_flag_090",
        ],
    )

    threshold_rows = build_threshold_rows(rows)
    write_csv(
        threshold_csv,
        threshold_rows,
        [
            "id_threshold",
            "N",
            "P",
            "M",
            "E",
            "K_max",
            "test_id_exact_rate",
            "test_value_accuracy",
            "run_dir",
        ],
    )

    plot_metric_by_k(rows, figure_dir)
    plot_threshold_compare(threshold_rows, figure_dir / "subproject4_excitation_threshold_kmax_compare.png")
    plot_normalized_capacity(threshold_rows, figure_dir / "subproject4_excitation_kmax_per_excitation.png")

    metric_table_rows, metric_table_fields = build_metric_table_rows(rows)
    write_csv(outputs_root / "subproject4_excitation_metric_table.csv", metric_table_rows, metric_table_fields)
    render_metric_table(
        metric_table_rows,
        metric_table_fields,
        figure_dir / "subproject4_excitation_metric_table.png",
    )
    return summary_csv, threshold_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize subproject-4 excitation-ablation results.")
    parser.add_argument("--outputs-root", default=str(PROJECT_ROOT / "outputs_subproj4_excitation_modelg2"))
    parser.add_argument("--figure-dir", default=str(PROJECT_ROOT / "Figure" / "subproject4_excitation"))
    args = parser.parse_args()

    summary_csv, threshold_csv = summarize(Path(args.outputs_root).resolve(), Path(args.figure_dir).resolve())
    print(f"summary_csv={summary_csv}")
    print(f"threshold_csv={threshold_csv}")


if __name__ == "__main__":
    main()
