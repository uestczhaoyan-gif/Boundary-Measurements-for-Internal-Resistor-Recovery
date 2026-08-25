from __future__ import annotations

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


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def apply_style() -> None:
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
        }
    )


def plot_line(x: list[float], y: list[float], xlabel: str, ylabel: str, title: str, output_path: Path) -> None:
    if plt is None:
        return
    apply_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.1), constrained_layout=True)
    ax.plot(x, y, marker="o", linewidth=2.0, color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def unique_keyed(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    seen: set[tuple[str, ...]] = set()
    unique_rows: list[dict] = []
    for row in rows:
        key = tuple(str(row[k]) for k in keys)
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def plot_subproject1() -> None:
    rows = read_csv_rows(PROJECT_ROOT / "outputs_modelg1" / "modelg1_subproj1_threshold_kmax_summary.csv")
    rows = unique_keyed(rows, ("N", "P", "M"))
    rows.sort(key=lambda item: int(item["P"]))
    x = [float(row["P"]) for row in rows]
    y = [float(row["M"]) / float(row["P"]) for row in rows]
    plot_line(
        x,
        y,
        xlabel="Port count P",
        ylabel="Resistors per port M / P",
        title="subproject1 structural load",
        output_path=PROJECT_ROOT / "Figure" / "modelg1_subproj1" / "subproject1_resistors_per_port.png",
    )


def plot_subproject2() -> None:
    rows = read_csv_rows(PROJECT_ROOT / "outputs_subproj2_varcand_modelg2" / "subproject2_varcand_threshold_kmax_summary.csv")
    rows = unique_keyed(rows, ("M_var", "P"))
    rows.sort(key=lambda item: int(item["M_var"]))
    x = [float(row["M_var"]) for row in rows]
    y = [float(row["M_var"]) / float(row["P"]) for row in rows]
    plot_line(
        x,
        y,
        xlabel="Candidate resistor count R",
        ylabel="R / P",
        title="subproject2 candidate load per port",
        output_path=PROJECT_ROOT / "Figure" / "subproject2_varcand" / "subproject2_candidate_resistors_per_port.png",
    )


def plot_subproject3() -> None:
    rows = read_csv_rows(PROJECT_ROOT / "outputs_subproj3_activeport_modelg2" / "subproject3_activeport_threshold_kmax_summary.csv")
    rows = unique_keyed(rows, ("P_active", "M"))
    rows.sort(key=lambda item: int(item["P_active"]))
    x = [float(row["P_active"]) for row in rows]
    y = [float(row["M"]) / float(row["P_active"]) for row in rows]
    plot_line(
        x,
        y,
        xlabel="Active port count P_active",
        ylabel="M / P_active",
        title="subproject3 structural load",
        output_path=PROJECT_ROOT / "Figure" / "subproject3_activeport" / "subproject3_resistors_per_active_port.png",
    )


def plot_subproject4() -> None:
    rows = read_csv_rows(PROJECT_ROOT / "outputs_subproj4_excitation_modelg2" / "subproject4_excitation_test_summary.csv")
    rows = unique_keyed(rows, ("E", "M"))
    rows.sort(key=lambda item: int(item["E"]))
    x = [float(row["E"]) for row in rows]
    y = [float(row["M"]) / float(row["E"]) for row in rows]
    plot_line(
        x,
        y,
        xlabel="Available excitation ports Pa",
        ylabel="R / Pa",
        title="subproject4 structural load",
        output_path=PROJECT_ROOT / "Figure" / "subproject4_excitation" / "subproject4_resistors_per_excitation.png",
    )


def main() -> None:
    if plt is None:
        raise SystemExit("matplotlib is unavailable")
    plot_subproject1()
    plot_subproject2()
    plot_subproject3()
    plot_subproject4()
    print("Generated channel-load figures for subprojects 1-4.")


if __name__ == "__main__":
    main()
