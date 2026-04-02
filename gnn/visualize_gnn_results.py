import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from vendor_bootstrap import bootstrap_vendor_paths, format_dependency_import_error

_BOOTSTRAP_RESULTS = bootstrap_vendor_paths(Path(__file__).resolve().parents[1])

try:
    import numpy as np
except Exception as exc:
    raise ImportError(format_dependency_import_error("numpy", exc, _BOOTSTRAP_RESULTS)) from None

try:
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import FancyBboxPatch
except Exception as exc:
    raise ImportError(format_dependency_import_error("matplotlib", exc, _BOOTSTRAP_RESULTS)) from None


FIG_BG = "#F6F1E8"
PANEL_BG = "#FFFDF9"
INK = "#18242F"
MUTED = "#6E7A83"
GRID = "#D7D0C6"
TEAL = "#1C8C7D"
ORANGE = "#C96A32"
BLUE = "#2F6EA6"
GOLD = "#C29A2B"
PURPLE = "#7A5AA6"


@dataclass
class RunArtifact:
    label: str
    mode: str
    run_dir: Path
    metrics_path: Path
    metrics: dict
    samples_path: Path | None
    samples: list
    confusion_matrix: np.ndarray | None


def sanitize_name(value):
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    safe = safe.strip("._-")
    return safe or "run"


def resolve_path(raw_path, script_dir):
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    candidates = [path, script_dir / path, script_dir.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (script_dir / path).resolve()


def resolve_output_dir(raw_path, script_dir):
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] == "outputs":
        return (script_dir / path).resolve()
    return (script_dir.parent / path).resolve()


def parse_run_item(raw_item, script_dir):
    if "=" in raw_item:
        label, raw_path = raw_item.split("=", 1)
        return sanitize_name(label), resolve_path(raw_path, script_dir)
    path = resolve_path(raw_item, script_dir)
    label_parts = [part for part in path.parts[-3:] if part not in {"outputs", "cache"}]
    return sanitize_name("_".join(label_parts)), path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_confusion_matrix_text(text):
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("["):
            continue
        nums = [int(x) for x in re.findall(r"-?\d+", stripped)]
        if nums:
            rows.append(nums)
    if not rows:
        return None
    return np.asarray(rows, dtype=np.int64)


def load_confusion_matrix(run_dir, metrics):
    if "confusion_matrix" in metrics:
        return np.asarray(metrics["confusion_matrix"], dtype=np.int64)

    candidates = [
        "confusion_matrix.txt",
        "confusion_matrix_test.txt",
        "confusion_matrix_count_test.txt",
    ]
    for filename in candidates:
        path = run_dir / filename
        if path.exists():
            matrix = parse_confusion_matrix_text(path.read_text(encoding="utf-8"))
            if matrix is not None:
                return matrix
    return None


def detect_run_artifact(path, label):
    if path.is_file():
        run_dir = path.parent
    else:
        run_dir = path

    candidates = [
        ("joint", run_dir / "cmei_metrics.json", run_dir / "detail_samples.json"),
        ("candidate", run_dir / "candidate_metrics.json", run_dir / "candidate_samples.json"),
        ("generic", run_dir / "metrics.json", None),
    ]
    for mode, metrics_path, samples_path in candidates:
        if not metrics_path.exists():
            continue
        metrics = read_json(metrics_path)
        if mode == "generic":
            if "test_macro_f1" in metrics:
                mode = "cls"
                inferred_samples = run_dir / "inference_samples.json"
            elif "mae_changed" in metrics:
                mode = "reg"
                inferred_samples = run_dir / "inference_full_samples.json"
                if not inferred_samples.exists():
                    inferred_samples = run_dir / "inference_samples.json"
            else:
                mode = "generic"
                inferred_samples = None
            samples_path = inferred_samples
        samples = read_json(samples_path) if samples_path and samples_path.exists() else []
        confusion = load_confusion_matrix(run_dir, metrics)
        return RunArtifact(
            label=label,
            mode=mode,
            run_dir=run_dir,
            metrics_path=metrics_path,
            metrics=metrics,
            samples_path=samples_path if samples_path and samples_path.exists() else None,
            samples=samples,
            confusion_matrix=confusion,
        )
    raise FileNotFoundError(f"Could not detect metrics in {path}")


def add_panel_background(ax):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)


def format_metric(value, precision=4):
    if value is None:
        return "-"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.{precision}f}"


def metric_card(ax, title, value, subtitle=None, accent=BLUE, x=0.02, y=0.15, w=0.22, h=0.7):
    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=0,
        facecolor="#F9F4EC",
        transform=ax.transAxes,
        zorder=1,
    )
    ax.add_patch(card)
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            0.02,
            h,
            boxstyle="round,pad=0.0,rounding_size=0.025",
            linewidth=0,
            facecolor=accent,
            transform=ax.transAxes,
            zorder=2,
        )
    )
    ax.text(x + 0.04, y + h - 0.18, title, transform=ax.transAxes, fontsize=10, color=MUTED, va="top")
    ax.text(
        x + 0.04,
        y + 0.23,
        value,
        transform=ax.transAxes,
        fontsize=19,
        fontweight="bold",
        color=INK,
        va="bottom",
        path_effects=[patheffects.withStroke(linewidth=3, foreground="#FFFDF9")],
    )
    if subtitle:
        ax.text(x + 0.04, y + 0.08, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")


def build_resistor_edges(rows, cols):
    edges = []
    for r in range(rows):
        for c in range(cols - 1):
            edges.append((r * cols + c, r * cols + c + 1))
        if r < rows - 1:
            for c in range(cols):
                edges.append((r * cols + c, (r + 1) * cols + c))
    return edges


def build_grid_positions(rows, cols):
    positions = {}
    for r in range(rows):
        for c in range(cols):
            positions[r * cols + c] = (c, rows - 1 - r)
    return positions


def infer_grid_shape(rows, cols, fallback=(8, 8)):
    if rows and cols:
        return rows, cols
    return fallback


def normalize_sample(sample, mode):
    if mode == "joint":
        return {
            "index": sample.get("sample_index"),
            "true_count": sample.get("true_k"),
            "pred_count": sample.get("pred_k"),
            "true_ids": sample.get("true_ids", []),
            "pred_ids": sample.get("pred_ids", []),
            "true_values": sample.get("true_deltas", []),
            "pred_values": sample.get("pred_deltas", []),
            "top_ids": sample.get("reg_top_abs_ids", []),
            "top_values": sample.get("reg_top_abs_values", []),
            "aux_values": sample.get("reg_top_mask_probs", []),
            "probs": sample.get("cls_probs", []),
            "thresholds": [],
        }
    if mode == "candidate":
        return {
            "index": sample.get("index"),
            "true_count": sample.get("true_change_count"),
            "pred_count": sample.get("pred_gt_threshold"),
            "true_ids": sample.get("true_change_ids", []),
            "pred_ids": sample.get("pred_change_ids", []),
            "true_values": sample.get("true_change_deltas", []),
            "pred_values": sample.get("pred_change_deltas", []),
            "top_ids": sample.get("pred_change_ids", []),
            "top_values": sample.get("pred_change_deltas", []),
            "aux_values": [],
            "probs": [],
            "thresholds": [],
        }
    if mode == "reg":
        return {
            "index": sample.get("index"),
            "true_count": sample.get("true_change_count"),
            "pred_count": sample.get("cls_pred_count", sample.get("pred_gt_threshold")),
            "true_ids": sample.get("true_change_ids", []),
            "pred_ids": sample.get("final_change_ids", sample.get("pred_change_ids", [])),
            "true_values": sample.get("true_change_deltas", []),
            "pred_values": sample.get("final_change_deltas", sample.get("pred_change_deltas", [])),
            "top_ids": sample.get("reg_top_abs_ids", sample.get("pred_change_ids", [])),
            "top_values": sample.get("pred_change_deltas", []),
            "aux_values": sample.get("pred_mask_prob", []),
            "probs": sample.get("cls_probs", []),
            "thresholds": sample.get("cls_thresholds", []),
        }
    if mode == "cls":
        return {
            "index": sample.get("index"),
            "true_count": sample.get("true_label"),
            "pred_count": sample.get("pred_label"),
            "true_ids": [],
            "pred_ids": [],
            "true_values": [],
            "pred_values": [],
            "top_ids": list(range(1, len(sample.get("coral_probs", [])) + 1)),
            "top_values": sample.get("coral_probs", []),
            "aux_values": sample.get("thresholds", []),
            "probs": sample.get("coral_probs", []),
            "thresholds": sample.get("thresholds", []),
        }
    return {
        "index": sample.get("index"),
        "true_count": None,
        "pred_count": None,
        "true_ids": [],
        "pred_ids": [],
        "true_values": [],
        "pred_values": [],
        "top_ids": [],
        "top_values": [],
        "aux_values": [],
        "probs": [],
        "thresholds": [],
    }


def draw_edge_network(ax, sample, rows, cols):
    add_panel_background(ax)
    ax.set_title(f"Sample {sample['index']}", loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.set_xticks([])
    ax.set_yticks([])

    positions = build_grid_positions(rows, cols)
    edges = build_resistor_edges(rows, cols)

    for idx, (u, v) in enumerate(edges):
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        color = "#DDD7CF"
        lw = 0.9
        alpha = 0.65
        if idx in sample["true_ids"] and idx in sample["pred_ids"]:
            color = TEAL
            lw = 3.2
            alpha = 1.0
        elif idx in sample["true_ids"]:
            color = BLUE
            lw = 3.0
            alpha = 0.95
        elif idx in sample["pred_ids"]:
            color = ORANGE
            lw = 3.0
            alpha = 0.95
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha, solid_capstyle="round")

    xs = [positions[i][0] for i in positions]
    ys = [positions[i][1] for i in positions]
    ax.scatter(xs, ys, s=14, color=INK, zorder=3)
    ax.set_xlim(-0.6, cols - 0.4)
    ax.set_ylim(-0.6, rows - 0.4)
    ax.set_aspect("equal")

    true_only = sorted(set(sample["true_ids"]) - set(sample["pred_ids"]))
    pred_only = sorted(set(sample["pred_ids"]) - set(sample["true_ids"]))
    overlap = sorted(set(sample["true_ids"]).intersection(sample["pred_ids"]))
    meta = [
        f"true_k={sample['true_count']}",
        f"pred_k={sample['pred_count']}",
        f"match={len(overlap)}",
        f"miss={len(true_only)}",
        f"extra={len(pred_only)}",
    ]
    ax.text(
        0.02,
        0.02,
        " | ".join(meta),
        transform=ax.transAxes,
        fontsize=9,
        color=MUTED,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFDF9", edgecolor="none"),
    )


def draw_sample_side(ax, sample, mode):
    add_panel_background(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.04, 0.94, "Prediction Snapshot", fontsize=12, fontweight="bold", color=INK, va="top")

    lines = []
    if sample["true_count"] is not None:
        lines.append(f"True / Pred count: {sample['true_count']} / {sample['pred_count']}")
    if sample["true_ids"]:
        lines.append(f"True ids: {sample['true_ids']}")
    if sample["pred_ids"]:
        lines.append(f"Pred ids: {sample['pred_ids']}")
    if sample["thresholds"]:
        lines.append("Thresholds: " + ", ".join(f"{x:.2f}" for x in sample["thresholds"]))

    y = 0.86
    for line in lines[:4]:
        ax.text(0.04, y, line, fontsize=9.5, color=MUTED, va="top")
        y -= 0.08

    if sample["top_values"]:
        if sample["top_ids"]:
            labels = [str(x) for x in sample["top_ids"][: min(6, len(sample["top_ids"]))]]
            vals = np.asarray(sample["top_values"][: len(labels)], dtype=np.float64)
            colors = [ORANGE if v >= 0 else BLUE for v in vals]
            bar_ax = ax.inset_axes([0.07, 0.08, 0.86, 0.42])
            add_panel_background(bar_ax)
            y_pos = np.arange(len(labels))
            bar_ax.barh(y_pos, np.abs(vals), color=colors, alpha=0.9)
            bar_ax.set_yticks(y_pos, labels)
            bar_ax.invert_yaxis()
            bar_ax.tick_params(axis="x", colors=MUTED, labelsize=8)
            bar_ax.tick_params(axis="y", colors=INK, labelsize=8)
            bar_ax.set_title("Top Responses", fontsize=10, loc="left", color=INK)
            bar_ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.7)
            for spine in bar_ax.spines.values():
                spine.set_visible(False)
        elif mode == "cls":
            bar_ax = ax.inset_axes([0.07, 0.08, 0.86, 0.42])
            add_panel_background(bar_ax)
            vals = np.asarray(sample["top_values"], dtype=np.float64)
            x_pos = np.arange(1, len(vals) + 1)
            bar_ax.bar(x_pos, vals, color=PURPLE, width=0.6)
            if sample["aux_values"]:
                bar_ax.plot(x_pos, sample["aux_values"], color=GOLD, marker="o", lw=1.5)
            bar_ax.set_ylim(0, 1.05)
            bar_ax.set_xticks(x_pos)
            bar_ax.set_title("CORAL Probabilities", fontsize=10, loc="left", color=INK)
            bar_ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
            for spine in bar_ax.spines.values():
                spine.set_visible(False)


def draw_confusion_matrix(ax, matrix, title):
    add_panel_background(ax)
    if matrix is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "No confusion matrix available", ha="center", va="center", color=MUTED)
        return
    im = ax.imshow(matrix, cmap="YlGnBu")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.set_xlabel("Pred", color=MUTED)
    ax.set_ylabel("True", color=MUTED)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            color = "#FFFFFF" if matrix[r, c] > matrix.max() * 0.55 else INK
            ax.text(c, r, str(matrix[r, c]), ha="center", va="center", fontsize=10, color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def draw_joint_score_bars(ax, metrics):
    add_panel_background(ax)
    scores = metrics.get("scores", {})
    labels = ["S_num", "S_F1", "S_id", "S_mse", "CMEI"]
    values = [scores.get(name, 0.0) for name in labels]
    colors = [BLUE, PURPLE, TEAL, GOLD, ORANGE]
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.65)
    ax.set_ylim(0, 100)
    ax.set_xticks(x, labels)
    ax.set_title("CMEI Composition", loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    for idx, value in enumerate(values):
        ax.text(idx, value + 1.2, f"{value:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_reg_snapshot(ax, metrics):
    add_panel_background(ax)
    labels = ["mae_all", "mae_changed", "count_F1", "avg_active", "avg_mask"]
    values = [
        metrics.get("mae_all"),
        metrics.get("mae_changed"),
        metrics.get("val_count_macro_f1", metrics.get("val_macro_f1")),
        metrics.get("avg_abs_gt_threshold"),
        metrics.get("avg_mask_prob"),
    ]
    scales = [1.0, 1.0, 100.0, 25.0, 100.0]
    display_values = []
    for value, scale in zip(values, scales):
        display_values.append(0.0 if value is None else float(value) * scale)
    colors = [BLUE, ORANGE, TEAL, GOLD, PURPLE]
    x = np.arange(len(labels))
    ax.bar(x, display_values, color=colors, width=0.65)
    ax.set_xticks(x, labels)
    ax.set_title("Regression Snapshot", loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    height_base = max(display_values + [1.0]) * 0.03
    for idx, raw_value in enumerate(values):
        text = "-" if raw_value is None else format_metric(float(raw_value))
        ax.text(idx, display_values[idx] + height_base, text, ha="center", va="bottom", fontsize=9, color=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_candidate_snapshot(ax, metrics):
    add_panel_background(ax)
    labels = ["top3", "top4", "top5", "top3_changed", "top4_changed"]
    values = [
        metrics.get("top3_candidate_cover", 0.0) * 100.0,
        metrics.get("top4_candidate_cover", 0.0) * 100.0,
        metrics.get("top5_candidate_cover", 0.0) * 100.0,
        metrics.get("top3_candidate_cover_changed_only", 0.0) * 100.0,
        metrics.get("top4_candidate_cover_changed_only", 0.0) * 100.0,
    ]
    colors = [BLUE, TEAL, GOLD, ORANGE, PURPLE]
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.65)
    ax.set_ylim(0, 100)
    ax.set_xticks(x, labels, rotation=10)
    ax.set_title("Candidate Coverage", loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    for idx, value in enumerate(values):
        ax.text(idx, value + 1.2, f"{value:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_cls_snapshot(ax, metrics):
    add_panel_background(ax)
    thresholds = metrics.get("best_thresholds", [])
    probs = [float(x) * 100.0 for x in thresholds]
    x = np.arange(1, len(probs) + 1)
    ax.bar(x, probs, color=PURPLE, width=0.6)
    ax.set_ylim(0, 100)
    ax.set_xticks(x, [f"T{i}" for i in x])
    ax.set_title("CORAL Thresholds", loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    for idx, value in enumerate(probs, start=1):
        ax.text(idx, value + 2, f"{value / 100:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_overview(run, out_dir):
    fig = plt.figure(figsize=(14, 10), facecolor=FIG_BG)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.9, 1.1, 1.2], hspace=0.28, wspace=0.18)

    title_ax = fig.add_subplot(gs[0, :])
    add_panel_background(title_ax)
    title_ax.axis("off")
    title_ax.text(0.03, 0.88, run.label, fontsize=24, fontweight="bold", color=INK, va="top")
    title_ax.text(0.03, 0.62, f"Mode: {run.mode.upper()}", fontsize=11, color=MUTED, va="top")
    dataset_tag = run.metrics.get("dataset_tag", run.run_dir.name)
    title_ax.text(0.03, 0.48, f"Dataset: {dataset_tag}", fontsize=11, color=MUTED, va="top")
    title_ax.text(0.03, 0.34, f"Source: {run.run_dir}", fontsize=9, color=MUTED, va="top")

    if run.mode == "joint":
        metric_card(title_ax, "CMEI", format_metric(run.metrics.get("scores", {}).get("CMEI"), 2), accent=ORANGE, x=0.48)
        metric_card(title_ax, "Macro-F1", format_metric(run.metrics.get("macro_f1")), accent=PURPLE, x=0.71)
        metric_card(title_ax, "ID Recall", format_metric(run.metrics.get("id_recall")), accent=TEAL, x=0.81)
    elif run.mode == "reg":
        metric_card(title_ax, "MAE Changed", format_metric(run.metrics.get("mae_changed")), accent=ORANGE, x=0.48)
        metric_card(title_ax, "MAE All", format_metric(run.metrics.get("mae_all")), accent=BLUE, x=0.71)
        metric_card(
            title_ax,
            "Count F1",
            format_metric(run.metrics.get("val_count_macro_f1", run.metrics.get("val_macro_f1"))),
            accent=TEAL,
            x=0.81,
        )
    elif run.mode == "candidate":
        metric_card(title_ax, "Top-3", format_metric(run.metrics.get("top3_candidate_cover"), 3), accent=ORANGE, x=0.48)
        metric_card(title_ax, "Top-4", format_metric(run.metrics.get("top4_candidate_cover"), 3), accent=TEAL, x=0.71)
        metric_card(title_ax, "Top-5", format_metric(run.metrics.get("top5_candidate_cover"), 3), accent=GOLD, x=0.81)
    elif run.mode == "cls":
        metric_card(title_ax, "Macro-F1", format_metric(run.metrics.get("test_macro_f1")), accent=PURPLE, x=0.48)
        metric_card(title_ax, "Best Epoch", str(run.metrics.get("best_epoch", "-")), accent=BLUE, x=0.71)
        metric_card(title_ax, "GAT Heads", str(run.metrics.get("gat_heads", "-")), accent=TEAL, x=0.81)

    cm_ax = fig.add_subplot(gs[1:, 0])
    draw_confusion_matrix(cm_ax, run.confusion_matrix, "Confusion Matrix")

    snap_ax = fig.add_subplot(gs[1, 1])
    if run.mode == "joint":
        draw_joint_score_bars(snap_ax, run.metrics)
    elif run.mode == "reg":
        draw_reg_snapshot(snap_ax, run.metrics)
    elif run.mode == "candidate":
        draw_candidate_snapshot(snap_ax, run.metrics)
    elif run.mode == "cls":
        draw_cls_snapshot(snap_ax, run.metrics)
    else:
        snap_ax.axis("off")

    text_ax = fig.add_subplot(gs[2, 1])
    add_panel_background(text_ax)
    text_ax.axis("off")
    notes = []
    if run.mode == "joint":
        notes = [
            f"num_accuracy={format_metric(run.metrics.get('num_accuracy'))}",
            f"macro_f1={format_metric(run.metrics.get('macro_f1'))}",
            f"id_recall={format_metric(run.metrics.get('id_recall'))}",
            f"mse_all_edges={format_metric(run.metrics.get('mse_all_edges'))}",
            f"near_miss={run.metrics.get('near_miss', {}).get('enabled', False)}",
        ]
    elif run.mode == "reg":
        notes = [
            f"best_count_threshold={format_metric(run.metrics.get('best_count_threshold'), 1)}",
            f"avg_active={format_metric(run.metrics.get('avg_abs_gt_threshold'))}",
            f"avg_mask_prob={format_metric(run.metrics.get('avg_mask_prob'))}",
            f"best_epoch={run.metrics.get('best_epoch', '-')}",
            f"best_val_score={format_metric(run.metrics.get('best_val_score'))}",
        ]
    elif run.mode == "candidate":
        notes = [
            f"top3_changed_only={format_metric(run.metrics.get('top3_candidate_cover_changed_only'))}",
            f"top4_changed_only={format_metric(run.metrics.get('top4_candidate_cover_changed_only'))}",
            f"top5_changed_only={format_metric(run.metrics.get('top5_candidate_cover_changed_only'))}",
            f"test_size={run.metrics.get('test_size', '-')}",
        ]
    elif run.mode == "cls":
        notes = [
            "thresholds=" + ", ".join(format_metric(x, 2) for x in run.metrics.get("best_thresholds", [])),
            f"best_epoch={run.metrics.get('best_epoch', '-')}",
            f"best_val_score={format_metric(run.metrics.get('best_val_score'))}",
            f"heads={run.metrics.get('gat_heads', '-')}",
        ]

    text_ax.text(0.04, 0.93, "Run Notes", fontsize=12, fontweight="bold", color=INK, va="top")
    ypos = 0.82
    for line in notes:
        text_ax.text(0.05, ypos, f"- {line}", fontsize=10, color=MUTED, va="top")
        ypos -= 0.12

    out_path = out_dir / f"{run.label}_overview.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)
    return out_path


def render_samples(run, out_dir, rows, cols, max_samples):
    if not run.samples:
        return None
    sample_count = min(max_samples, len(run.samples))
    selected = [normalize_sample(sample, run.mode) for sample in run.samples[:sample_count]]

    if run.mode == "cls":
        fig = plt.figure(figsize=(14, 3.8 * sample_count), facecolor=FIG_BG)
        gs = GridSpec(sample_count, 1, figure=fig, hspace=0.18)
        for idx, sample in enumerate(selected):
            ax = fig.add_subplot(gs[idx, 0])
            draw_sample_side(ax, sample, run.mode)
            ax.text(0.92, 0.92, f"Sample {sample['index']}", fontsize=12, fontweight="bold", color=INK, ha="right", va="top")
        out_path = out_dir / f"{run.label}_samples.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=FIG_BG)
        plt.close(fig)
        return out_path

    fig = plt.figure(figsize=(15, 5.2 * sample_count), facecolor=FIG_BG)
    gs = GridSpec(sample_count, 2, figure=fig, width_ratios=[1.2, 1.0], hspace=0.22, wspace=0.12)
    for idx, sample in enumerate(selected):
        left_ax = fig.add_subplot(gs[idx, 0])
        right_ax = fig.add_subplot(gs[idx, 1])
        draw_edge_network(left_ax, sample, rows, cols)
        draw_sample_side(right_ax, sample, run.mode)

    out_path = out_dir / f"{run.label}_samples.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)
    return out_path


def get_metric_specs(mode):
    if mode == "reg":
        return [
            ("mae_all", "MAE All", False),
            ("mae_changed", "MAE Changed", False),
            ("val_count_macro_f1", "Count F1", True),
            ("avg_abs_gt_threshold", "Avg Active", False),
            ("avg_mask_prob", "Avg Mask", False),
        ]
    if mode == "cls":
        return [
            ("test_macro_f1", "Macro-F1", True),
            ("best_epoch", "Best Epoch", False),
            ("gat_heads", "GAT Heads", False),
        ]
    if mode == "joint":
        return [
            ("scores.CMEI", "CMEI", True),
            ("num_accuracy", "Num Acc", True),
            ("macro_f1", "Macro-F1", True),
            ("id_recall", "ID Recall", True),
            ("mse_all_edges", "All-edge MSE", False),
        ]
    if mode == "candidate":
        return [
            ("top3_candidate_cover", "Top-3", True),
            ("top4_candidate_cover", "Top-4", True),
            ("top5_candidate_cover", "Top-5", True),
        ]
    return []


def nested_metric(metrics, key):
    if "." not in key:
        return metrics.get(key)
    current = metrics
    for part in key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def render_comparison(runs, out_dir):
    if len(runs) < 2:
        return None

    modes = {run.mode for run in runs}
    shared_mode = next(iter(modes)) if len(modes) == 1 else None

    fig = plt.figure(figsize=(15, 9), facecolor=FIG_BG)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.05, 1.15], hspace=0.22)

    top_ax = fig.add_subplot(gs[0, 0])
    add_panel_background(top_ax)
    top_ax.axis("off")
    title = "GNN Model Evolution" if shared_mode else "GNN Run Comparison"
    top_ax.text(0.03, 0.90, title, fontsize=24, fontweight="bold", color=INK, va="top")
    top_ax.text(0.03, 0.74, " -> ".join(run.label for run in runs), fontsize=11, color=MUTED, va="top")

    specs = get_metric_specs(shared_mode) if shared_mode else []
    if shared_mode and specs:
        left_ax = top_ax.inset_axes([0.04, 0.08, 0.54, 0.52])
        right_ax = top_ax.inset_axes([0.63, 0.08, 0.33, 0.52])
        add_panel_background(left_ax)
        add_panel_background(right_ax)

        x = np.arange(len(runs))
        palette = [BLUE, TEAL, ORANGE, PURPLE, GOLD]
        for idx, (key, label, _higher_is_better) in enumerate(specs):
            values = [nested_metric(run.metrics, key) for run in runs]
            if any(value is None for value in values):
                continue
            color = palette[idx % len(palette)]
            left_ax.plot(x, values, marker="o", lw=2.2, color=color, label=label)
            for px, value in zip(x, values):
                left_ax.text(px, value, format_metric(float(value), 3), fontsize=8, color=color, ha="center", va="bottom")
        left_ax.set_xticks(x, [run.label for run in runs], rotation=10)
        left_ax.set_title("Metric Trajectories", loc="left", fontsize=12, fontweight="bold", color=INK)
        left_ax.grid(color=GRID, linewidth=0.8, alpha=0.7)
        left_ax.legend(frameon=False, fontsize=9, loc="best")
        for spine in left_ax.spines.values():
            spine.set_visible(False)

        heat = np.zeros((len(specs), len(runs)), dtype=np.float64)
        labels = []
        for row_idx, (key, label, higher_is_better) in enumerate(specs):
            values = np.asarray([float(nested_metric(run.metrics, key)) for run in runs], dtype=np.float64)
            vmin = values.min()
            vmax = values.max()
            if math.isclose(vmin, vmax):
                norm = np.full_like(values, 0.5)
            else:
                norm = (values - vmin) / (vmax - vmin)
            if not higher_is_better:
                norm = 1.0 - norm
            heat[row_idx] = norm
            labels.append(label)
        im = right_ax.imshow(heat, cmap="YlOrBr", aspect="auto", vmin=0.0, vmax=1.0)
        right_ax.set_xticks(range(len(runs)), [run.label for run in runs], rotation=10)
        right_ax.set_yticks(range(len(labels)), labels)
        right_ax.set_title("Relative Quality", loc="left", fontsize=12, fontweight="bold", color=INK)
        for row_idx, (key, _label, _higher_is_better) in enumerate(specs):
            for col_idx, run in enumerate(runs):
                value = nested_metric(run.metrics, key)
                right_ax.text(col_idx, row_idx, format_metric(float(value), 3), ha="center", va="center", fontsize=8, color=INK)
        for spine in right_ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(im, ax=right_ax, fraction=0.046, pad=0.04)
    else:
        top_ax.text(
            0.03,
            0.54,
            "Mixed-mode comparison detected. This figure will focus on summary cards instead of shared trajectories.",
            fontsize=11,
            color=MUTED,
            va="top",
        )

    bottom_ax = fig.add_subplot(gs[1, 0])
    add_panel_background(bottom_ax)
    bottom_ax.axis("off")
    bottom_ax.text(0.03, 0.94, "Run Cards", fontsize=13, fontweight="bold", color=INK, va="top")

    card_width = min(0.29, 0.92 / max(len(runs), 1))
    gap = 0.03
    for idx, run in enumerate(runs):
        x0 = 0.03 + idx * (card_width + gap)
        if x0 + card_width > 0.99:
            break
        card = FancyBboxPatch(
            (x0, 0.10),
            card_width,
            0.74,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=0,
            facecolor="#F9F4EC",
            transform=bottom_ax.transAxes,
        )
        bottom_ax.add_patch(card)
        bottom_ax.text(x0 + 0.03, 0.78, run.label, transform=bottom_ax.transAxes, fontsize=12, fontweight="bold", color=INK)
        bottom_ax.text(x0 + 0.03, 0.70, f"mode={run.mode}", transform=bottom_ax.transAxes, fontsize=9, color=MUTED)
        if run.mode == "reg":
            lines = [
                f"mae_changed={format_metric(run.metrics.get('mae_changed'))}",
                f"mae_all={format_metric(run.metrics.get('mae_all'))}",
                f"count_f1={format_metric(run.metrics.get('val_count_macro_f1', run.metrics.get('val_macro_f1')))}",
                f"avg_active={format_metric(run.metrics.get('avg_abs_gt_threshold'))}",
            ]
        elif run.mode == "cls":
            lines = [
                f"macro_f1={format_metric(run.metrics.get('test_macro_f1'))}",
                f"best_epoch={run.metrics.get('best_epoch', '-')}",
                "thresholds=" + ", ".join(format_metric(x, 2) for x in run.metrics.get("best_thresholds", [])),
            ]
        elif run.mode == "joint":
            scores = run.metrics.get("scores", {})
            lines = [
                f"CMEI={format_metric(scores.get('CMEI'), 2)}",
                f"num_acc={format_metric(run.metrics.get('num_accuracy'))}",
                f"macro_f1={format_metric(run.metrics.get('macro_f1'))}",
                f"id_recall={format_metric(run.metrics.get('id_recall'))}",
            ]
        elif run.mode == "candidate":
            lines = [
                f"top3={format_metric(run.metrics.get('top3_candidate_cover'))}",
                f"top4={format_metric(run.metrics.get('top4_candidate_cover'))}",
                f"top5={format_metric(run.metrics.get('top5_candidate_cover'))}",
            ]
        else:
            lines = [f"metrics={run.metrics_path.name}"]
        ypos = 0.60
        for line in lines[:5]:
            bottom_ax.text(x0 + 0.03, ypos, line, transform=bottom_ax.transAxes, fontsize=9.5, color=MUTED)
            ypos -= 0.10

    out_path = out_dir / "comparison.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)
    return out_path


def write_manifest(runs, outputs, out_dir):
    manifest = {
        "runs": [
            {
                "label": run.label,
                "mode": run.mode,
                "run_dir": str(run.run_dir),
                "metrics_path": str(run.metrics_path),
                "samples_path": str(run.samples_path) if run.samples_path else None,
            }
            for run in runs
        ],
        "outputs": [str(path) for path in outputs if path is not None],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def parse_args():
    parser = argparse.ArgumentParser(description="Create polished overview and evolution charts for GNN runs.")
    parser.add_argument(
        "runs",
        nargs="+",
        help="Run directories or metrics files. Use label=path to set a custom legend name.",
    )
    parser.add_argument("--out-dir", default="outputs/visualizations")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    out_dir = resolve_output_dir(args.out_dir, script_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, cols = infer_grid_shape(args.rows, args.cols)
    runs = []
    for raw_item in args.runs:
        label, path = parse_run_item(raw_item, script_dir)
        runs.append(detect_run_artifact(path, label))

    output_paths = []
    for run in runs:
        output_paths.append(render_overview(run, out_dir))
        sample_path = render_samples(run, out_dir, rows, cols, args.max_samples)
        if sample_path is not None:
            output_paths.append(sample_path)

    comparison_path = render_comparison(runs, out_dir)
    if comparison_path is not None:
        output_paths.append(comparison_path)

    manifest_path = write_manifest(runs, output_paths, out_dir)
    print("Saved visualizations:")
    for path in output_paths:
        if path is not None:
            print(f"  {path}")
    print(f"  {manifest_path}")


if __name__ == "__main__":
    main()
