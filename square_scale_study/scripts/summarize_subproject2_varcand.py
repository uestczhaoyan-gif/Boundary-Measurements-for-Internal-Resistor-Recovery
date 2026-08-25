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
        if str(infer_payload.get("study_protocol", "")) != "variable_candidate_pool":
            continue
        best_train = train_payload.get("best_train_metrics") or {}
        best_val = train_payload.get("best_val_metrics") or {}
        rows.append(
            {
                "run_dir": train_path.parent.name,
                "N": int(train_payload["grid_size"]),
                "P": int(train_payload["port_count"]),
                "M": int(train_payload["num_resistors"]),
                "M_var": int(infer_payload["candidate_edge_count"]),
                "K": int(train_payload["k"]),
                "best_epoch": int(train_payload.get("best_epoch") or 0),
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
    rows.sort(key=lambda item: (item["M_var"], item["K"]))
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
        grouped.setdefault(row["M_var"], []).append(row)

    threshold_rows: list[dict] = []
    for threshold in THRESHOLDS:
        for m_var, group in sorted(grouped.items()):
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
                        "M_var": m_var,
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
                        "M_var": m_var,
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
        grouped.setdefault(row["M_var"], []).append(row)

    items = sorted(grouped.items())
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.3), constrained_layout=True)
    axes = axes.flatten()
    legend_handles = None

    for ax, (m_var, group) in zip(axes, items):
        ks = [row["K"] for row in group]
        id_vals = [row["test_id_exact_rate"] for row in group]
        value_vals = [row["test_value_accuracy"] for row in group]
        width = 0.36
        x = list(range(len(ks)))
        bars_id = ax.bar([v - width / 2 for v in x], id_vals, width=width, color="#1f77b4", label="Test ID")
        bars_value = ax.bar([v + width / 2 for v in x], value_vals, width=width, color="#ff7f0e", label="Test Value")
        if legend_handles is None:
            legend_handles = (bars_id[0], bars_value[0])
        ax.axhline(0.98, color="#1f77b4", linestyle=":", linewidth=1.0)
        ax.axhline(0.90, color="#ff7f0e", linestyle=":", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_ylim(0.0, 1.03)
        ax.set_xlabel("K")
        ax.set_ylabel("Metric")
        ax.set_title(f"M_var={m_var}")

        single_fig, single_ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
        single_ax.bar([v - width / 2 for v in x], id_vals, width=width, color="#1f77b4", label="Test ID")
        single_ax.bar([v + width / 2 for v in x], value_vals, width=width, color="#ff7f0e", label="Test Value")
        single_ax.axhline(0.98, color="#1f77b4", linestyle=":", linewidth=1.0)
        single_ax.axhline(0.90, color="#ff7f0e", linestyle=":", linewidth=1.0)
        single_ax.set_xticks(x)
        single_ax.set_xticklabels([str(k) for k in ks])
        single_ax.set_ylim(0.0, 1.03)
        single_ax.set_xlabel("Changed resistor count K")
        single_ax.set_ylabel("Metric")
        single_ax.set_title(f"subproject2 test metrics, M_var={m_var}")
        single_ax.legend(frameon=False, loc="lower left")
        figure_dir.mkdir(parents=True, exist_ok=True)
        single_fig.savefig(figure_dir / f"Mvar{m_var}_test_metric_dropoff.png", dpi=220)
        plt.close(single_fig)

    for ax in axes[len(items):]:
        ax.axis("off")

    if legend_handles is not None:
        fig.legend(legend_handles, ["Test ID", "Test Value"], frameon=False, loc="upper center", ncol=2)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "subproject2_varcand_test_metric_dropoff_overview.png", dpi=220)
    plt.close(fig)


def plot_threshold_compare(threshold_rows: list[dict], output_path: Path) -> None:
    if plt is None or not threshold_rows:
        return
    _apply_style()
    grouped: dict[float, list[dict]] = {}
    for row in threshold_rows:
        grouped.setdefault(float(row["id_threshold"]), []).append(row)

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.3), constrained_layout=True)
    axes = axes.flatten()
    for ax, threshold in zip(axes, THRESHOLDS):
        group = sorted(grouped.get(float(threshold), []), key=lambda item: item["M_var"])
        x = [row["M_var"] for row in group]
        y = [row["K_max"] for row in group]
        ax.plot(x, y, marker="o", linewidth=2.0, color="#1f77b4")
        ax.set_xticks(x)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("Candidate pool size M_var")
        ax.set_ylabel("K_max")
        ax.set_title(f"ID threshold = {int(round(threshold * 100))}%")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_metric_by_k(rows: list[dict], figure_dir: Path) -> None:
    if plt is None or not rows:
        return
    _apply_style()
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["K"], []).append(row)

    ks = sorted(grouped)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6), constrained_layout=True)
    axes = axes.flatten()

    for ax, k in zip(axes, ks):
        group = sorted(grouped[k], key=lambda item: item["M_var"])
        x = [row["M_var"] for row in group]
        id_vals = [row["test_id_exact_rate"] for row in group]
        value_vals = [row["test_value_accuracy"] for row in group]

        ax.plot(x, id_vals, marker="o", linewidth=2.0, color="#1f77b4", label="Test ID")
        ax.plot(x, value_vals, marker="s", linewidth=2.0, color="#ff7f0e", label="Test Value")
        ax.set_xticks(x)
        ax.set_ylim(0.70, 1.01)
        ax.set_xlabel("Candidate pool size M_var")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"K = {k}")

        single_fig, single_ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
        single_ax.plot(x, id_vals, marker="o", linewidth=2.0, color="#1f77b4", label="Test ID")
        single_ax.plot(x, value_vals, marker="s", linewidth=2.0, color="#ff7f0e", label="Test Value")
        single_ax.set_xticks(x)
        single_ax.set_ylim(0.70, 1.01)
        single_ax.set_xlabel("Candidate pool size M_var")
        single_ax.set_ylabel("Accuracy")
        single_ax.set_title(f"subproject2 test accuracy trend, K={k}")
        single_ax.legend(frameon=False, loc="lower left")
        figure_dir.mkdir(parents=True, exist_ok=True)
        single_fig.savefig(figure_dir / f"K{k}_accuracy_vs_mvar.png", dpi=220)
        plt.close(single_fig)

    for ax in axes[len(ks):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "subproject2_varcand_accuracy_by_k_overview.png", dpi=220)
    plt.close(fig)


def build_metric_table_rows(rows: list[dict]) -> tuple[list[dict], list[str]]:
    grouped: dict[int, dict[int, dict]] = {}
    ks = sorted({row["K"] for row in rows})
    for row in rows:
        grouped.setdefault(row["M_var"], {})[row["K"]] = row

    fieldnames = ["M_var"]
    for k in ks:
        fieldnames.extend([f"K{k}_ID", f"K{k}_Value"])

    table_rows: list[dict] = []
    for m_var in sorted(grouped):
        row_out: dict[str, object] = {"M_var": m_var}
        for k in ks:
            item = grouped[m_var].get(k)
            row_out[f"K{k}_ID"] = "" if item is None else f"{item['test_id_exact_rate']:.3f}"
            row_out[f"K{k}_Value"] = "" if item is None else f"{item['test_value_accuracy']:.3f}"
        table_rows.append(row_out)
    return table_rows, fieldnames


def render_metric_table(table_rows: list[dict], fieldnames: list[str], output_png: Path) -> None:
    if plt is None or not table_rows:
        return

    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(10.8, 2.4 + 0.48 * len(table_rows)), constrained_layout=True)
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
    ax.set_title("subproject2 test metric table", fontsize=12, pad=10)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize(outputs_root: Path, figure_dir: Path) -> tuple[Path, Path]:
    rows = load_rows(outputs_root)
    if not rows:
        raise SystemExit(f"No subproject2 train/inference pairs found under {outputs_root}")

    summary_csv = outputs_root / "subproject2_varcand_test_summary.csv"
    threshold_csv = outputs_root / "subproject2_varcand_threshold_kmax_summary.csv"
    write_csv(
        summary_csv,
        rows,
        [
            "run_dir",
            "N",
            "P",
            "M",
            "M_var",
            "K",
            "best_epoch",
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

    threshold_rows = build_threshold_rows(rows)
    write_csv(
        threshold_csv,
        threshold_rows,
        [
            "id_threshold",
            "N",
            "P",
            "M",
            "M_var",
            "K_max",
            "test_id_exact_rate",
            "test_value_accuracy",
            "run_dir",
        ],
    )

    plot_metric_dropoff(rows, figure_dir)
    plot_metric_by_k(rows, figure_dir)
    plot_threshold_compare(threshold_rows, figure_dir / "subproject2_varcand_threshold_kmax_compare.png")

    metric_table_rows, metric_table_fields = build_metric_table_rows(rows)
    write_csv(outputs_root / "subproject2_varcand_metric_table.csv", metric_table_rows, metric_table_fields)
    render_metric_table(
        metric_table_rows,
        metric_table_fields,
        figure_dir / "subproject2_varcand_metric_table.png",
    )
    return summary_csv, threshold_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize subproject-2 variable-candidate-pool results.")
    parser.add_argument("--outputs-root", default=str(PROJECT_ROOT / "outputs_subproj2_varcand_modelg2"))
    parser.add_argument("--figure-dir", default=str(PROJECT_ROOT / "Figure" / "subproject2_varcand"))
    args = parser.parse_args()

    summary_csv, threshold_csv = summarize(Path(args.outputs_root).resolve(), Path(args.figure_dir).resolve())
    print(f"summary_csv={summary_csv}")
    print(f"threshold_csv={threshold_csv}")


if __name__ == "__main__":
    main()
