from __future__ import annotations

import json
import sys
from pathlib import Path


EXPAND_ROOT = Path(__file__).resolve().parent
FIGURE_DIR = EXPAND_ROOT / "Figure"
VENDOR_PLOT = EXPAND_ROOT / ".vendor_plot"
if VENDOR_PLOT.exists() and str(VENDOR_PLOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_PLOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STAGE_SPECS = [
    {
        "stage_name": "stage1_square_10x10",
        "label_lines": ["Stage1", "Square 10x10", "100 nodes"],
        "short_label": "Stage1 Square 10x10",
        "tag": "square_10x10",
    },
    {
        "stage_name": "stage2_rect_6x10",
        "label_lines": ["Stage2", "Rect 6x10", "60 nodes"],
        "short_label": "Stage2 Rect 6x10",
        "tag": "rect_6x10",
    },
    {
        "stage_name": "stage3_honeycomb_63",
        "label_lines": ["Stage3", "Honeycomb", "63 nodes"],
        "short_label": "Stage3 Honeycomb 63",
        "tag": "honeycomb_63",
    },
    {
        "stage_name": "stage4_transfer_circlecut_69",
        "label_lines": ["Stage4", "Circle-Cut", "69 nodes"],
        "short_label": "Stage4 Circle-Cut 69",
        "tag": "circlecut_69",
    },
]

STAGE_COLORS = ["#005BBB", "#D62828", "#2A9D2F", "#F28E2B"]
COMPONENT_KEYS = ["S_num", "S_F1", "S_id", "S_mse"]
COMPONENT_LABELS = ["S_num", "S_F1", "S_id", "S_mse"]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect_stage_records(expand_root: Path):
    records = []
    for spec in STAGE_SPECS:
        stage_root = expand_root / spec["stage_name"]
        tag = spec["tag"]
        cls_metrics = load_json(stage_root / "cls" / "outputs" / tag / "metrics.json")
        reg_metrics = load_json(stage_root / "reg" / "outputs" / tag / "metrics.json")
        joint_metrics = load_json(stage_root / "joint_inference" / "outputs" / tag / "cmei_metrics.json")
        score_block = joint_metrics["scores"]
        record = {
            "stage_name": spec["stage_name"],
            "label_lines": spec["label_lines"],
            "short_label": spec["short_label"],
            "dataset_tag": tag,
            "topology_title": cls_metrics["topology_title"],
            "num_nodes": cls_metrics["num_nodes"],
            "num_resistors": cls_metrics["num_resistors"],
            "cls_macro_f1": cls_metrics["test_macro_f1"],
            "cls_warm_loaded": cls_metrics.get("warm_start", {}).get("loaded", 0),
            "cls_pretrained_model_path": cls_metrics.get("pretrained_model_path", ""),
            "reg_mae_all": reg_metrics["mae_all"],
            "reg_mae_changed": reg_metrics["mae_changed"],
            "reg_count_macro_f1": reg_metrics["val_count_macro_f1"],
            "reg_warm_loaded": reg_metrics.get("warm_start", {}).get("loaded", 0),
            "reg_pretrained_model_path": reg_metrics.get("pretrained_model_path", ""),
            "joint_cmei": score_block["CMEI"],
            "joint_num_accuracy": joint_metrics["num_accuracy"],
            "joint_macro_f1": joint_metrics["macro_f1"],
            "joint_id_recall": joint_metrics["id_recall"],
            "joint_mse_all_edges": joint_metrics["mse_all_edges"],
            "score_components": {key: score_block[key] for key in COMPONENT_KEYS},
        }
        records.append(record)
    return records


def save_summary_json(records, output_path: Path):
    payload = {
        "summary_name": "GNN_EXPAND latest stage results",
        "notes": [
            "Metrics are read from each stage's actual outputs/ directory.",
            "The official visualization is now emitted as PNG and PDF.",
            "Stage4 current run successfully loaded Stage1 warm-start weights for both cls and reg.",
            "Stage1/2/3 classifier warm-start weights were not loaded in the current recorded run.",
        ],
        "stages": records,
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


def draw_component_panel(ax, records):
    x = np.arange(len(COMPONENT_LABELS))
    for idx, record in enumerate(records):
        values = [record["score_components"][key] for key in COMPONENT_KEYS]
        ax.plot(
            x,
            values,
            color=STAGE_COLORS[idx],
            marker="o",
            markersize=5.5,
            linewidth=2.0,
            label=record["short_label"],
        )
        for px, py in zip(x, values):
            ax.text(px, py + 0.55, f"{py:.1f}", color=STAGE_COLORS[idx], ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x, COMPONENT_LABELS)
    ax.set_ylim(78.0, 101.0)
    ax.set_ylabel("Score")
    ax.set_title("Component Scores")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=2, frameon=False)


def draw_cmei_panel(ax, records):
    x = np.arange(len(records))
    y = np.array([record["joint_cmei"] for record in records], dtype=np.float64)
    ax.plot(x, y, color="#6F6F6F", linewidth=1.6, linestyle="-", zorder=1)
    for idx, record in enumerate(records):
        ax.scatter(x[idx], y[idx], s=58, color=STAGE_COLORS[idx], edgecolors="#222222", linewidths=0.8, zorder=2)
        ax.text(x[idx], y[idx] + 0.35, f"{y[idx]:.2f}", color=STAGE_COLORS[idx], ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x, [line[0] for line in (record["label_lines"] for record in records)])
    ax.set_ylim(87.0, 96.0)
    ax.set_ylabel("CMEI")
    ax.set_title("Overall CMEI")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)


def build_figure(records, png_path: Path, pdf_path: Path):
    configure_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 5.8),
        dpi=200,
        gridspec_kw={"width_ratios": [1.8, 1.0]},
        constrained_layout=False,
    )
    fig.patch.set_facecolor("white")

    draw_component_panel(axes[0], records)
    draw_cmei_panel(axes[1], records)

    fig.suptitle("GNN_EXPAND Summary", fontsize=15, fontweight="bold", y=0.975)
    note = (
        "Notes: scores are read from each stage output. Stage4 current run successfully loaded "
        "Stage1 warm-start weights for both cls and reg. Stage1/2/3 classifier warm-start weights "
        "were still unavailable in the current recorded run."
    )
    fig.text(0.07, 0.02, note, ha="left", va="bottom", fontsize=8.8, color="#555555")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.18, wspace=0.22)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main():
    records = collect_stage_records(EXPAND_ROOT)
    json_path = EXPAND_ROOT / "expand_summary_metrics.json"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / "expand_summary.png"
    pdf_path = FIGURE_DIR / "expand_summary.pdf"
    save_summary_json(records, json_path)
    build_figure(records, png_path, pdf_path)
    print(f"Saved summary JSON to {json_path}")
    print(f"Saved figure to {png_path}")
    print(f"Saved figure to {pdf_path}")


if __name__ == "__main__":
    main()
