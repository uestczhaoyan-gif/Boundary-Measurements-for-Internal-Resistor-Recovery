from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


NOISE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = NOISE_ROOT.parents[1]
FIGURE_DIR = NOISE_ROOT / "Figure"
VENDOR_PLOT_DIRS = [
    NOISE_ROOT / ".vendor_plot",
    WORKSPACE_ROOT / "gnn" / "GNN_EXPAND" / ".vendor_plot",
]
for vendor_dir in VENDOR_PLOT_DIRS:
    if vendor_dir.exists() and str(vendor_dir) not in sys.path:
        sys.path.insert(0, str(vendor_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUN_TAG = "training_data64Nodes_2_noiseft_struct_boundary_v2_20260402"
LEVEL_SPECS = [
    {"label": "Clean", "suffix": "clean", "json_suffix": "clean", "noise_std": 0.0},
    {"label": "40 dB", "suffix": "40dB", "json_suffix": "40dB", "noise_std": 0.01},
    {"label": "30 dB", "suffix": "30dB", "json_suffix": "30dB", "noise_std": 0.0316227766},
    {"label": "20 dB", "suffix": "20dB", "json_suffix": "20dB", "noise_std": 0.1},
]
SUBSCORE_COLORS = {
    "S_num": "#F28E2B",
    "S_F1": "#005BBB",
    "S_id": "#D62828",
    "S_mse": "#2A9D2F",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot the latest GNN_NOISE v2 summary figure.")
    parser.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect_records(run_tag: str):
    records = []
    cls_root = NOISE_ROOT / "CLS_modelo3_ft_v2" / "outputs" / run_tag
    reg_root = NOISE_ROOT / "REG_o4a2_ft_v2" / "outputs" / run_tag
    joint_root = WORKSPACE_ROOT / "gnn" / "GNN_CMEI_INFERENCE" / "outputs"

    for spec in LEVEL_SPECS:
        if spec["suffix"] == "clean":
            cls_path = cls_root / "inference_eval_clean.json"
            reg_path = reg_root / "inference_eval_clean.json"
        else:
            cls_path = cls_root / f"noise_eval_{spec['json_suffix']}.json"
            reg_path = reg_root / f"noise_eval_{spec['json_suffix']}.json"

        joint_dir = joint_root / f"gnn_cmei_{run_tag}_{spec['suffix']}" / run_tag
        joint_path = joint_dir / "cmei_metrics.json"

        cls_metrics = load_json(cls_path)
        reg_metrics = load_json(reg_path)
        joint_metrics = load_json(joint_path)
        record = {
            "label": spec["label"],
            "noise_std": spec["noise_std"],
            "cls_macro_f1": cls_metrics["test_macro_f1"],
            "reg_mae_all": reg_metrics["mae_all"],
            "reg_mae_changed": reg_metrics["mae_changed"],
            "reg_count_macro_f1": reg_metrics["count_macro_f1"],
            "joint_cmei": joint_metrics["scores"]["CMEI"],
            "joint_num_accuracy": joint_metrics["num_accuracy"],
            "joint_macro_f1": joint_metrics["macro_f1"],
            "joint_id_recall": joint_metrics["id_recall"],
            "joint_mse_all_edges": joint_metrics["mse_all_edges"],
            "joint_scores": {
                "S_num": joint_metrics["scores"]["S_num"],
                "S_F1": joint_metrics["scores"]["S_F1"],
                "S_id": joint_metrics["scores"]["S_id"],
                "S_mse": joint_metrics["scores"]["S_mse"],
            },
        }
        records.append(record)
    return records


def save_summary_json(records, output_path: Path, run_tag: str):
    payload = {
        "summary_name": "GNN_NOISE v2 latest robustness summary",
        "run_tag": run_tag,
        "notes": [
            "Metrics are read from actual single-model outputs and GNN_CMEI_INFERENCE joint outputs.",
            "Noise is injected only on measurable boundary-node voltages.",
            "This figure reflects the complete v2 curve for clean, 40 dB, 30 dB and 20 dB.",
        ],
        "levels": records,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.9,
            "grid.color": "#D9D9D9",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
        }
    )


def annotate_line(ax, x, y, color, fmt):
    for px, py in zip(x, y):
        ax.text(px, py, format(float(py), fmt), color=color, fontsize=8, ha="center", va="bottom")


def plot_single_metric(ax, labels, values, color, title, ylabel, ylim, fmt):
    x = np.arange(len(labels))
    ax.plot(x, values, color=color, marker="o", markersize=5.5, linewidth=2.1)
    annotate_line(ax, x, values + (ylim[1] - ylim[0]) * 0.012, color, fmt)
    ax.set_xticks(x, labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)


def build_figure(records, png_path: Path, pdf_path: Path):
    configure_style()
    labels = [record["label"] for record in records]
    cls_values = np.array([record["cls_macro_f1"] for record in records], dtype=np.float64)
    reg_values = np.array([record["reg_mae_changed"] for record in records], dtype=np.float64)
    joint_values = np.array([record["joint_cmei"] for record in records], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), dpi=200, constrained_layout=False)
    fig.patch.set_facecolor("white")

    plot_single_metric(
        axes[0, 0],
        labels,
        cls_values,
        "#005BBB",
        "CLS Macro-F1",
        "Macro-F1",
        (0.72, 0.93),
        ".3f",
    )
    plot_single_metric(
        axes[0, 1],
        labels,
        reg_values,
        "#D62828",
        "REG MAE_changed (lower is better)",
        "MAE_changed",
        (20.0, 62.0),
        ".2f",
    )
    plot_single_metric(
        axes[1, 0],
        labels,
        joint_values,
        "#2A9D2F",
        "Joint CMEI",
        "CMEI",
        (78.0, 95.0),
        ".2f",
    )

    x = np.arange(len(labels))
    ax = axes[1, 1]
    for key in ("S_num", "S_F1", "S_id", "S_mse"):
        values = np.array([record["joint_scores"][key] for record in records], dtype=np.float64)
        ax.plot(
            x,
            values,
            color=SUBSCORE_COLORS[key],
            marker="o",
            markersize=5.2,
            linewidth=2.0,
            label=key,
        )
    ax.set_xticks(x, labels)
    ax.set_title("Joint Sub-scores")
    ax.set_ylabel("Score")
    ax.set_ylim(68.0, 101.0)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", ncol=2, frameon=False)

    fig.suptitle("GNN_NOISE v2 Robustness Summary", fontsize=15, fontweight="bold", y=0.97)
    fig.text(
        0.06,
        0.02,
        "Noise is injected only on measurable boundary-node voltages. Metrics are read from the completed v2 evaluation outputs.",
        ha="left",
        va="bottom",
        fontsize=8.8,
        color="#555555",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.90, bottom=0.11, hspace=0.30, wspace=0.22)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    records = collect_records(args.run_tag)
    json_path = NOISE_ROOT / "noise_v2_summary_metrics.json"
    png_path = FIGURE_DIR / "noise_v2_summary.png"
    pdf_path = FIGURE_DIR / "noise_v2_summary.pdf"
    save_summary_json(records, json_path, args.run_tag)
    build_figure(records, png_path, pdf_path)
    print(f"Saved summary JSON to {json_path}")
    print(f"Saved figure to {png_path}")
    print(f"Saved figure to {pdf_path}")


if __name__ == "__main__":
    main()
