from __future__ import annotations

import argparse
import csv
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



def load_json(path: Path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def collect_rows(outputs_root: Path) -> list[dict]:
    rows: list[dict] = []
    for run_dir in sorted(outputs_root.glob("N*x*_K*")):
        if not re.fullmatch(r"N\d+x\d+_K\d+", run_dir.name):
            continue
        train_path = run_dir / "train_metrics.json"
        infer_path = run_dir / "inference_metrics.json"
        if not train_path.exists() or not infer_path.exists():
            continue
        train = load_json(train_path)
        infer = load_json(infer_path)
        row = {
            "run_dir": str(run_dir),
            "N": int(infer["grid_size"]),
            "P": int(infer["port_count"]),
            "M": int(infer["num_resistors"]),
            "K": int(infer["k"]),
            "train_id_exact_rate": float(train["best_train_metrics"]["id_exact_rate"]),
            "train_value_accuracy": float(train["best_train_metrics"]["value_accuracy"]),
            "val_id_exact_rate": float(train["best_val_metrics"]["id_exact_rate"]),
            "val_value_accuracy": float(train["best_val_metrics"]["value_accuracy"]),
            "test_id_exact_rate": float(infer["id_exact_rate"]),
            "test_value_accuracy": float(infer["value_accuracy"]),
            "test_mae_changed": float(infer["mae_changed"]),
            "pass_flag": int(bool(infer["pass_flag"])),
        }
        rows.append(row)
    rows.sort(key=lambda row: (row["N"], row["K"]))
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_kmax_rows(rows: list[dict]) -> list[dict]:
    by_n: dict[int, list[dict]] = {}
    for row in rows:
        by_n.setdefault(int(row["N"]), []).append(row)

    summary_rows: list[dict] = []
    for n in sorted(by_n):
        candidates = [row for row in by_n[n] if int(row["pass_flag"]) == 1]
        if not candidates:
            summary_rows.append(
                {
                    "N": n,
                    "P": 4 * n - 4,
                    "M": 2 * n * (n - 1),
                    "K_max": 0,
                    "test_id_exact_rate": 0.0,
                    "test_value_accuracy": 0.0,
                    "run_dir": "",
                }
            )
            continue
        best = max(candidates, key=lambda row: int(row["K"]))
        summary_rows.append(
            {
                "N": n,
                "P": int(best["P"]),
                "M": int(best["M"]),
                "K_max": int(best["K"]),
                "test_id_exact_rate": float(best["test_id_exact_rate"]),
                "test_value_accuracy": float(best["test_value_accuracy"]),
                "run_dir": best["run_dir"],
            }
        )
    return summary_rows


def plot_summary(kmax_rows: list[dict], figure_path: Path) -> None:
    if plt is None or not kmax_rows:
        return
    ns = [row["N"] for row in kmax_rows]
    ports = [row["P"] for row in kmax_rows]
    kmax = [row["K_max"] for row in kmax_rows]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)
    axes[0].plot(ns, kmax, marker="o", color="#1f77b4", linewidth=1.8)
    axes[0].set_xlabel("Grid size N")
    axes[0].set_ylabel("Maximum identifiable K")
    axes[0].set_title("N vs K_max")

    axes[1].plot(ports, kmax, marker="s", color="#d62728", linewidth=1.8)
    axes[1].set_xlabel("Port count P")
    axes[1].set_ylabel("Maximum identifiable K")
    axes[1].set_title("P vs K_max")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def summarize_outputs(outputs_root: Path, figure_path: Path | None = None) -> tuple[Path, Path]:
    rows = collect_rows(outputs_root)
    scale_summary = outputs_root / "scale_k_sweep_summary.csv"
    write_csv(
        scale_summary,
        rows,
        [
            "run_dir",
            "N",
            "P",
            "M",
            "K",
            "train_id_exact_rate",
            "train_value_accuracy",
            "val_id_exact_rate",
            "val_value_accuracy",
            "test_id_exact_rate",
            "test_value_accuracy",
            "test_mae_changed",
            "pass_flag",
        ],
    )
    kmax_rows = build_kmax_rows(rows)
    kmax_summary = outputs_root / "port_vs_kmax_summary.csv"
    write_csv(
        kmax_summary,
        kmax_rows,
        ["N", "P", "M", "K_max", "test_id_exact_rate", "test_value_accuracy", "run_dir"],
    )
    if figure_path is not None:
        plot_summary(kmax_rows, figure_path)
    return scale_summary, kmax_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize square_scale_study outputs into CSV tables and a figure.")
    parser.add_argument("--outputs-root", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--figure-path", default=str(PROJECT_ROOT / "Figure" / "scale_kmax_summary.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize_outputs(Path(args.outputs_root).resolve(), Path(args.figure_path).resolve())


if __name__ == "__main__":
    main()
