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


EPOCH_RE = re.compile(
    r"Epoch (?P<epoch>\d+) \| train_loss=(?P<train_loss>[0-9.]+) "
    r"\| train_id=(?P<train_id>[0-9.]+) \| train_value=(?P<train_value>[0-9.]+) "
    r"\| val_id=(?P<val_id>[0-9.]+) \| val_value=(?P<val_value>[0-9.]+) "
    r"\| val_mae=(?P<val_mae>[0-9.]+)"
)


def load_train_rows(outputs_root: Path) -> list[dict]:
    rows: list[dict] = []
    for metrics_path in sorted(outputs_root.rglob("train_metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        best_train = payload.get("best_train_metrics") or {}
        best_val = payload.get("best_val_metrics") or {}
        rows.append(
            {
                "model_name": outputs_root.name,
                "run_dir": str(metrics_path.parent),
                "N": int(payload["grid_size"]),
                "P": int(payload["port_count"]),
                "K": int(payload["k"]),
                "best_epoch": int(payload.get("best_epoch") or 0),
                "train_id_exact_rate": float(best_train.get("id_exact_rate") or 0.0),
                "train_value_accuracy": float(best_train.get("value_accuracy") or 0.0),
                "val_id_exact_rate": float(best_val.get("id_exact_rate") or 0.0),
                "val_value_accuracy": float(best_val.get("value_accuracy") or 0.0),
                "val_mae_changed": float(best_val.get("mae_changed") or 0.0),
                "param_count": int(payload.get("param_count") or 0),
            }
        )
    rows.sort(key=lambda item: (item["N"], item["K"]))
    return rows


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_name",
                "run_dir",
                "N",
                "P",
                "K",
                "best_epoch",
                "train_id_exact_rate",
                "train_value_accuracy",
                "val_id_exact_rate",
                "val_value_accuracy",
                "val_mae_changed",
                "param_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def render_table_png(path: Path, rows: list[dict]) -> None:
    if plt is None or not rows:
        return

    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["N"], []).append(row)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(
        len(grouped),
        1,
        figsize=(10.4, 2.6 * len(grouped) + 0.5),
        constrained_layout=True,
    )
    try:
        axes = axes.flatten()
    except Exception:
        axes = [axes]

    for ax, (n, group) in zip(axes, sorted(grouped.items())):
        ax.axis("off")
        cell_text = [
            [
                str(row["K"]),
                str(row["best_epoch"]),
                f"{row['val_id_exact_rate']:.3f}",
                f"{row['val_value_accuracy']:.3f}",
                f"{row['val_mae_changed']:.2f}",
            ]
            for row in group
        ]
        table = ax.table(
            cellText=cell_text,
            colLabels=["K", "Best epoch", "Best val ID", "Best val Value", "Best val MAE"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.28)
        ax.set_title(f"modelo2_mlp best validation metrics, N={n}, P={group[0]['P']}", fontsize=11, pad=8)

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_dropoff(rows: list[dict], output_path: Path) -> None:
    if plt is None or not rows:
        return

    ks = [row["K"] for row in rows]
    id_vals = [row["val_id_exact_rate"] for row in rows]
    value_vals = [row["val_value_accuracy"] for row in rows]
    n = rows[0]["N"]
    p = rows[0]["P"]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    width = 0.36
    x = list(range(len(ks)))
    ax.bar([v - width / 2 for v in x], id_vals, width=width, color="#1f77b4", label="Best val ID")
    ax.bar([v + width / 2 for v in x], value_vals, width=width, color="#ff7f0e", label="Best val Value")
    ax.axhline(0.98, color="#1f77b4", linestyle=":", linewidth=1.2)
    ax.axhline(0.90, color="#ff7f0e", linestyle=":", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Changed resistor count K")
    ax.set_ylabel("Metric")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"modelo2_mlp: N={n}, P={p}, best validation drop-off")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_scale_compare(rows: list[dict], output_path: Path) -> None:
    if plt is None or not rows:
        return

    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["N"], []).append(row)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for idx, (n, group) in enumerate(sorted(grouped.items())):
        ks = [row["K"] for row in group]
        axes[0].plot(ks, [row["val_id_exact_rate"] for row in group], marker="o", linewidth=2.0, color=palette[idx % len(palette)], label=f"N={n}")
        axes[1].plot(ks, [row["val_value_accuracy"] for row in group], marker="o", linewidth=2.0, color=palette[idx % len(palette)], label=f"N={n}")

    axes[0].axhline(0.98, color="#666666", linestyle=":", linewidth=1.1)
    axes[1].axhline(0.90, color="#666666", linestyle=":", linewidth=1.1)
    axes[0].set_title("Best validation ID vs K")
    axes[1].set_title("Best validation value accuracy vs K")
    for ax in axes:
        ax.set_xlabel("Changed resistor count K")
        ax.set_ylabel("Metric")
        ax.set_ylim(0.0, 1.05)
        ax.legend(frameon=False)

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def resolve_log_path(log_path_arg: str) -> Path | None:
    if log_path_arg:
        candidate = Path(log_path_arg).resolve()
        return candidate if candidate.exists() else None
    default_path = PROJECT_ROOT / "训练日志.txt"
    if default_path.exists():
        return default_path
    candidates = sorted(PROJECT_ROOT.glob("*.txt"))
    return candidates[0] if candidates else None


def extract_epoch_block(text: str, marker: str) -> list[dict]:
    start_idx = text.find(marker)
    if start_idx < 0:
        return []

    lines = text[start_idx:].splitlines()
    rows: list[dict] = []
    started = False
    for line in lines:
        if not started:
            if marker in line:
                started = True
            continue
        match = EPOCH_RE.search(line)
        if match:
            rows.append(
                {
                    "epoch": int(match.group("epoch")),
                    "train_loss": float(match.group("train_loss")),
                    "train_id": float(match.group("train_id")),
                    "train_value": float(match.group("train_value")),
                    "val_id": float(match.group("val_id")),
                    "val_value": float(match.group("val_value")),
                    "val_mae": float(match.group("val_mae")),
                }
            )
        elif rows:
            break
    return rows


def build_partial_row(model_name: str, n: int, k: int, rows: list[dict]) -> dict | None:
    if not rows:
        return None
    best = max(rows, key=lambda item: (item["val_id"], item["val_value"], -item["val_mae"]))
    last = rows[-1]
    return {
        "model_name": model_name,
        "N": n,
        "K": k,
        "epochs_seen": len(rows),
        "best_epoch_seen": best["epoch"],
        "best_val_id_exact_rate": best["val_id"],
        "best_val_value_accuracy": best["val_value"],
        "best_val_mae_changed": best["val_mae"],
        "last_epoch_seen": last["epoch"],
        "last_val_id_exact_rate": last["val_id"],
        "last_val_value_accuracy": last["val_value"],
        "last_val_mae_changed": last["val_mae"],
    }


def write_partial_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_name",
                "N",
                "K",
                "epochs_seen",
                "best_epoch_seen",
                "best_val_id_exact_rate",
                "best_val_value_accuracy",
                "best_val_mae_changed",
                "last_epoch_seen",
                "last_val_id_exact_rate",
                "last_val_value_accuracy",
                "last_val_mae_changed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_n9_compare(mlp_rows: list[dict], gnn_rows: list[dict], output_path: Path) -> None:
    if plt is None or (not mlp_rows and not gnn_rows):
        return

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    if mlp_rows:
        axes[0].plot([r["epoch"] for r in mlp_rows], [r["val_id"] for r in mlp_rows], color="#d62728", linewidth=2.0, label="modelo2_mlp")
        axes[1].plot([r["epoch"] for r in mlp_rows], [r["val_value"] for r in mlp_rows], color="#d62728", linewidth=2.0, label="modelo2_mlp")
    if gnn_rows:
        axes[0].plot([r["epoch"] for r in gnn_rows], [r["val_id"] for r in gnn_rows], color="#1f77b4", linewidth=2.0, label="modelo2_gnn")
        axes[1].plot([r["epoch"] for r in gnn_rows], [r["val_value"] for r in gnn_rows], color="#1f77b4", linewidth=2.0, label="modelo2_gnn")

    axes[0].set_title("N=9, K=1: validation ID during training")
    axes[1].set_title("N=9, K=1: validation value accuracy during training")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric")
        ax.set_ylim(0.0, 1.05)
        ax.legend(frameon=False)

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize modelo2 stage progress from train_metrics and log files.")
    parser.add_argument("--outputs-root", default=str(PROJECT_ROOT / "Outputs_o2mlp"))
    parser.add_argument("--figure-dir", default=str(PROJECT_ROOT / "Figure" / "modelo2_progress"))
    parser.add_argument("--log-path", default="")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root).resolve()
    figure_dir = Path(args.figure_dir).resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows = load_train_rows(outputs_root)
    write_summary_csv(figure_dir / "modelo2_mlp_bestval_summary.csv", rows)
    render_table_png(figure_dir / "modelo2_mlp_bestval_table.png", rows)

    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["N"], []).append(row)
    for n, group in sorted(grouped.items()):
        plot_dropoff(group, figure_dir / f"N{n}_P{group[0]['P']}_bestval_dropoff.png")
    plot_scale_compare(rows, figure_dir / "modelo2_mlp_scale_compare.png")

    partial_rows: list[dict] = []
    log_path = resolve_log_path(args.log_path)
    if log_path is not None:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        mlp_n9k1 = extract_epoch_block(text, "START modelo2_mlp N9x9 K1")
        gnn_n9k1 = extract_epoch_block(text, "START modelo2_gnn N9x9 K1")
        if not gnn_n9k1:
            gnn_n9k1 = extract_epoch_block(
                text,
                "python square_scale_study/models/modelo2_gnn/train.py",
            )
        if mlp_n9k1 or gnn_n9k1:
            plot_n9_compare(mlp_n9k1, gnn_n9k1, figure_dir / "modelo2_n9_k1_training_compare.png")
        partial_mlp = build_partial_row("modelo2_mlp", 9, 1, mlp_n9k1)
        partial_gnn = build_partial_row("modelo2_gnn", 9, 1, gnn_n9k1)
        if partial_mlp is not None:
            partial_rows.append(partial_mlp)
        if partial_gnn is not None:
            partial_rows.append(partial_gnn)
    write_partial_csv(figure_dir / "modelo2_partial_log_summary.csv", partial_rows)


if __name__ == "__main__":
    main()
