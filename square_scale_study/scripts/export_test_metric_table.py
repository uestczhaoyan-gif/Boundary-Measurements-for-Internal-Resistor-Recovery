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


def load_rows(path: Path, id_field: str, value_field: str, pass_field: str) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "N": int(row["N"]),
                    "P": int(row["P"]),
                    "K": int(row["K"]),
                    "test_id_exact_rate": float(row[id_field]),
                    "test_value_accuracy": float(row[value_field]),
                    "pass_flag": int(row[pass_field]),
                }
            )
    rows.sort(key=lambda item: (item["N"], item["K"]))
    return rows


def write_compact_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["N", "P", "K", "test_id_exact_rate", "test_value_accuracy", "pass_flag"],
        )
        writer.writeheader()
        writer.writerows(rows)


def render_table_png(path: Path, rows: list[dict], title: str) -> None:
    if plt is None or not rows:
        return

    by_n: dict[int, list[dict]] = {}
    for row in rows:
        by_n.setdefault(row["N"], []).append(row)

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(
        len(by_n),
        1,
        figsize=(9.2, 2.45 * len(by_n) + 0.8),
        constrained_layout=True,
    )
    try:
        axes = axes.flatten()
    except Exception:
        axes = [axes]

    fig.suptitle(title, fontsize=13, y=1.01)

    for ax, (n, group) in zip(axes, sorted(by_n.items())):
        ax.axis("off")
        cell_text = []
        for row in group:
            cell_text.append(
                [
                    str(row["K"]),
                    f"{row['test_id_exact_rate']:.3f}",
                    f"{row['test_value_accuracy']:.3f}",
                    "PASS" if row["pass_flag"] else "FAIL",
                ]
            )

        table = ax.table(
            cellText=cell_text,
            colLabels=["K", "ID", "Value", "Pass"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.28)
        ax.set_title(f"N={n}, P={group[0]['P']}", fontsize=11, pad=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a compact metric table as CSV and PNG.")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--png-out", required=True)
    parser.add_argument("--title", default="Test Metric Table")
    parser.add_argument("--id-field", default="test_id_exact_rate")
    parser.add_argument("--value-field", default="test_value_accuracy")
    parser.add_argument("--pass-field", default="pass_flag")
    args = parser.parse_args()

    rows = load_rows(
        Path(args.summary_csv).resolve(),
        id_field=args.id_field,
        value_field=args.value_field,
        pass_field=args.pass_field,
    )
    write_compact_csv(Path(args.csv_out).resolve(), rows)
    render_table_png(Path(args.png_out).resolve(), rows, args.title)


if __name__ == "__main__":
    main()

