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


MODEL_ROOTS = {
    "modelv3": PROJECT_ROOT / "outputs_modelv3",
    "modelo1_gnn": PROJECT_ROOT / "outputs_modelo1_gnn",
    "modelo1_gnn_mse": PROJECT_ROOT / "outputs_modelo1_gnn_mse",
    "modelo1_mlp1": PROJECT_ROOT / "outputs_modelo1_mlp1",
    "modelo1_mlp2": PROJECT_ROOT / "outputs_modelo1_mlp2",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_rows(grid_size: int) -> list[dict]:
    rows: list[dict] = []
    pattern = f"N{grid_size}x{grid_size}_K*"
    for model_name, root in MODEL_ROOTS.items():
        if not root.exists():
            continue
        for run_dir in sorted(root.glob(pattern)):
            infer_path = run_dir / "inference_metrics.json"
            if not infer_path.exists():
                continue
            infer = load_json(infer_path)
            rows.append(
                {
                    "model": model_name,
                    "N": int(infer["grid_size"]),
                    "P": int(infer["port_count"]),
                    "K": int(infer["k"]),
                    "test_id_exact_rate": float(infer["id_exact_rate"]),
                    "test_value_accuracy": float(infer["value_accuracy"]),
                    "test_mae_changed": float(infer["mae_changed"]),
                    "pass_flag": int(bool(infer["pass_flag"])),
                }
            )
    rows.sort(key=lambda row: (row["K"], row["model"]))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "N",
                "P",
                "K",
                "test_id_exact_rate",
                "test_value_accuracy",
                "test_mae_changed",
                "pass_flag",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_rows(path: Path, rows: list[dict], grid_size: int) -> None:
    if plt is None or not rows:
        return

    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    style_map = {
        "modelv3": ("#7f7f7f", "o", "modelv3"),
        "modelo1_gnn": ("#1f77b4", "s", "modelo1_gnn"),
        "modelo1_gnn_mse": ("#8c564b", "P", "modelo1_gnn_mse"),
        "modelo1_mlp1": ("#ff7f0e", "^", "modelo1_mlp1"),
        "modelo1_mlp2": ("#2ca02c", "D", "modelo1_mlp2"),
    }

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.30,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True)

    for model_name, group in sorted(by_model.items()):
        group = sorted(group, key=lambda item: item["K"])
        color, marker, label = style_map.get(model_name, ("#333333", "o", model_name))
        ks = [row["K"] for row in group]
        ids = [row["test_id_exact_rate"] for row in group]
        vals = [row["test_value_accuracy"] for row in group]
        axes[0].plot(ks, ids, marker=marker, color=color, linewidth=1.8, label=label)
        axes[1].plot(ks, vals, marker=marker, color=color, linewidth=1.8, label=label)

    axes[0].axhline(0.98, color="#444444", linestyle=":", linewidth=1.1)
    axes[1].axhline(0.90, color="#444444", linestyle=":", linewidth=1.1)
    axes[0].set_title(f"N={grid_size}: test ID exact rate")
    axes[1].set_title(f"N={grid_size}: test value accuracy")
    axes[0].set_xlabel("Changed resistor count K")
    axes[1].set_xlabel("Changed resistor count K")
    axes[0].set_ylabel("ID exact rate")
    axes[1].set_ylabel("Value accuracy")
    axes[0].set_ylim(0.6, 1.02)
    axes[1].set_ylim(0.95, 1.0)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot N-fixed model-family comparison for ID and value metrics.")
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--png-out", required=True)
    args = parser.parse_args()

    rows = collect_rows(args.grid_size)
    write_csv(Path(args.csv_out).resolve(), rows)
    plot_rows(Path(args.png_out).resolve(), rows, args.grid_size)


if __name__ == "__main__":
    main()
