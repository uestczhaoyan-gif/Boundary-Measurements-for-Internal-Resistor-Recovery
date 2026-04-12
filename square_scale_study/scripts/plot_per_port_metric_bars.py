from __future__ import annotations

import argparse
import csv
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



def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "N": int(row["N"]),
                    "P": int(row["P"]),
                    "K": int(row["K"]),
                    "test_id_exact_rate": float(row["test_id_exact_rate"]),
                    "test_value_accuracy": float(row["test_value_accuracy"]),
                    "pass_flag": int(row["pass_flag"]),
                }
            )
    rows.sort(key=lambda item: (item["P"], item["K"]))
    return rows


def plot_group(rows: list[dict], output_path: Path) -> None:
    if plt is None or not rows:
        return

    ks = [row["K"] for row in rows]
    id_vals = [row["test_id_exact_rate"] for row in rows]
    value_vals = [row["test_value_accuracy"] for row in rows]
    n = rows[0]["N"]
    p = rows[0]["P"]

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

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    width = 0.36
    x = list(range(len(ks)))
    ax.bar([v - width / 2 for v in x], id_vals, width=width, color="#1f77b4", label="ID exact rate")
    ax.bar([v + width / 2 for v in x], value_vals, width=width, color="#ff7f0e", label="Value accuracy")

    ax.axhline(0.98, color="#1f77b4", linestyle=":", linewidth=1.2)
    ax.axhline(0.90, color="#ff7f0e", linestyle=":", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Changed resistor count K")
    ax.set_ylabel("Metric")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"N={n}, P={p}: metric drop-off vs K")
    ax.legend(frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-port grouped bar charts for test metrics vs K.")
    parser.add_argument("--summary-csv", default=str(PROJECT_ROOT / "outputs" / "scale_k_sweep_summary.csv"))
    parser.add_argument("--figure-dir", default=str(PROJECT_ROOT / "Figure" / "metric_dropoff_by_port"))
    args = parser.parse_args()

    rows = load_rows(Path(args.summary_csv).resolve())
    by_port: dict[int, list[dict]] = {}
    for row in rows:
        by_port.setdefault(row["P"], []).append(row)

    figure_dir = Path(args.figure_dir).resolve()
    for p, group in sorted(by_port.items()):
        n = group[0]["N"]
        plot_group(group, figure_dir / f"N{n}_P{p}_metric_dropoff.png")


if __name__ == "__main__":
    main()
