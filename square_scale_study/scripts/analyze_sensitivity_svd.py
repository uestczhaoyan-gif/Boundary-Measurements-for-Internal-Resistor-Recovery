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
from project_common import (
    BASE_R,
    DEFAULT_CURRENT_A,
    build_boundary_excitations,
    build_conductance,
    build_rhs_matrix,
    build_square_topology,
    fmt_float,
    solve_all_excitations,
)

prepend_vendor_dir(VENDOR_PLOT, required_version=(3, 11))

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build electrical sensitivity matrix and run SVD analysis.")
    parser.add_argument("--grid-list", default="3,4,5,6,7,8")
    parser.add_argument("--delta-r", type=float, default=10.0)
    parser.add_argument(
        "--model-result-roots",
        nargs="*",
        default=[
            str(PROJECT_ROOT / "outputs_modelg2_subproj1"),
            str(PROJECT_ROOT / "outputs_modelg2_task1_n78"),
        ],
    )
    parser.add_argument(
        "--summary-dir",
        default=str(PROJECT_ROOT / "outputs_sensitivity_svd"),
    )
    parser.add_argument(
        "--figure-dir",
        default=str(PROJECT_ROOT / "Figure" / "sensitivity_svd"),
    )
    return parser.parse_args()


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
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "legend.fontsize": 9.5,
        }
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stacked_boundary_voltage_response(grid_size: int, resistor_values: np.ndarray) -> tuple[np.ndarray, int, int]:
    topology = build_square_topology(grid_size)
    excitations = build_boundary_excitations(topology, excitation_count=None)
    ref_node = topology.boundary_nodes_clockwise[0]
    keep_idx, rhs = build_rhs_matrix(
        topology.num_nodes,
        ref_node=ref_node,
        excitations=excitations,
        current_a=DEFAULT_CURRENT_A,
    )
    gmat = build_conductance(topology.num_nodes, resistor_values, topology.resistor_edges)
    voltages = solve_all_excitations(gmat, keep_idx, ref_node, rhs, excitations)
    boundary_nodes = list(topology.boundary_nodes_clockwise)
    # Rows later correspond to all (excitation, measured boundary node) observations.
    return voltages[boundary_nodes, :].T.reshape(-1), len(excitations), len(boundary_nodes)


def build_sensitivity_matrix(grid_size: int, delta_r: float) -> tuple[np.ndarray, dict]:
    topology = build_square_topology(grid_size)
    base_values = np.full(topology.num_resistors, BASE_R, dtype=np.float64)
    base_response, excitation_count, measurement_count = stacked_boundary_voltage_response(grid_size, base_values)
    j = np.zeros((base_response.shape[0], topology.num_resistors), dtype=np.float64)
    for rid in range(topology.num_resistors):
        values = base_values.copy()
        values[rid] += delta_r
        perturbed, _e, _p = stacked_boundary_voltage_response(grid_size, values)
        j[:, rid] = (perturbed - base_response) / delta_r
    meta = {
        "N": grid_size,
        "P": topology.port_count,
        "M": topology.num_resistors,
        "num_observations": int(j.shape[0]),
        "excitation_count": int(excitation_count),
        "measurement_count": int(measurement_count),
        "delta_r": float(delta_r),
    }
    return j, meta


def svd_metrics(j: np.ndarray, meta: dict) -> tuple[dict, np.ndarray]:
    s = np.linalg.svd(j, compute_uv=False)
    total_energy = float(np.sum(s**2))
    normalized = s / max(float(s[0]), 1e-30)
    probs = (s**2) / max(total_energy, 1e-30)
    entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-30))))
    effective_rank = float(np.exp(entropy))
    cumulative = np.cumsum(s**2) / max(total_energy, 1e-30)
    row = {
        **meta,
        "rank_1e_1": int(np.sum(normalized >= 1e-1)),
        "rank_1e_2": int(np.sum(normalized >= 1e-2)),
        "rank_1e_3": int(np.sum(normalized >= 1e-3)),
        "rank_1e_4": int(np.sum(normalized >= 1e-4)),
        "effective_rank": effective_rank,
        "stable_rank": float(total_energy / max(float(s[0] ** 2), 1e-30)),
        "condition_1e_12": float(s[0] / max(float(s[-1]), 1e-12)),
        "energy_modes_90": int(np.searchsorted(cumulative, 0.90) + 1),
        "energy_modes_95": int(np.searchsorted(cumulative, 0.95) + 1),
        "energy_modes_99": int(np.searchsorted(cumulative, 0.99) + 1),
        "sigma_max": float(s[0]),
        "sigma_min": float(s[-1]),
    }
    m = max(int(meta["M"]), 1)
    row["effective_rank_per_M"] = float(row["effective_rank"] / m)
    row["stable_rank_per_M"] = float(row["stable_rank"] / m)
    row["rank_1e_2_per_M"] = float(row["rank_1e_2"] / m)
    return row, s


def collect_model_kmax(result_roots: list[Path]) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[int, int, Path]] = set()
    for root in result_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("inference_metrics.json")):
            payload = read_json(path)
            if "grid_size" not in payload or "k" not in payload:
                continue
            n = int(payload["grid_size"])
            k = int(payload["k"])
            key = (n, k, path.parent)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "N": n,
                    "K": k,
                    "id_exact_rate": float(payload.get("id_exact_rate", 0.0)),
                    "value_accuracy": float(payload.get("value_accuracy", 0.0)),
                    "mae_changed": float(payload.get("mae_changed", 0.0)),
                    "run_dir": str(path.parent),
                }
            )
    rows.sort(key=lambda row: (int(row["N"]), int(row["K"])))
    return rows


def summarize_kmax(rows: list[dict], thresholds: list[float]) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["N"]), []).append(row)
    output: list[dict] = []
    for n, group in sorted(grouped.items()):
        for threshold in thresholds:
            passed = [
                int(row["K"])
                for row in group
                if float(row["id_exact_rate"]) >= threshold and float(row["value_accuracy"]) >= 0.90
            ]
            output.append(
                {
                    "N": n,
                    "id_threshold": threshold,
                    "K_max": max(passed) if passed else 0,
                }
            )
    return output


def plot_spectrum(spectrum_rows: list[dict], figure_dir: Path) -> None:
    if plt is None or not spectrum_rows:
        return
    apply_style()
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    grouped: dict[int, list[dict]] = {}
    for row in spectrum_rows:
        grouped.setdefault(int(row["N"]), []).append(row)
    for n in sorted(grouped):
        group = sorted(grouped[n], key=lambda item: int(item["mode"]))
        sigma = np.asarray([float(item["sigma_normalized"]) for item in group], dtype=float)
        ax.semilogy(np.arange(1, len(sigma) + 1), sigma, marker="o", markersize=3.0, linewidth=1.8, label=f"N={n}")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Normalized singular value")
    ax.set_title("Electrical sensitivity spectrum")
    ax.legend(frameon=False, ncol=2)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "sensitivity_svd_spectrum.png", dpi=220)
    plt.close(fig)


def plot_rank_metrics(metric_rows: list[dict], figure_dir: Path) -> None:
    if plt is None or not metric_rows:
        return
    apply_style()
    rows = sorted(metric_rows, key=lambda item: int(item["N"]))
    n = [int(row["N"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    axes[0].plot(n, [float(row["effective_rank"]) for row in rows], marker="o", linewidth=2.0, label="Effective rank")
    axes[0].plot(n, [float(row["stable_rank"]) for row in rows], marker="s", linewidth=2.0, label="Stable rank")
    axes[0].plot(n, [int(row["rank_1e_2"]) for row in rows], marker="^", linewidth=2.0, label="Rank >= 1e-2")
    axes[0].set_xlabel("Grid size N")
    axes[0].set_ylabel("Rank-like metric")
    axes[0].set_title("Information capacity from SVD")
    axes[0].legend(frameon=False)

    axes[1].plot(n, [int(row["num_observations"]) for row in rows], marker="o", linewidth=2.0, label="Obs. rows")
    axes[1].plot(n, [int(row["M"]) for row in rows], marker="s", linewidth=2.0, label="Variable edges M")
    axes[1].plot(n, [int(row["P"]) for row in rows], marker="^", linewidth=2.0, label="Ports P")
    axes[1].set_xlabel("Grid size N")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Matrix size")
    axes[1].legend(frameon=False)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "sensitivity_svd_rank_metrics.png", dpi=220)
    plt.close(fig)


def plot_svd_vs_kmax(metric_rows: list[dict], kmax_rows: list[dict], figure_dir: Path) -> None:
    if plt is None or not metric_rows or not kmax_rows:
        return
    apply_style()
    metric_by_n = {int(row["N"]): row for row in metric_rows}
    thresholds = sorted({float(row["id_threshold"]) for row in kmax_rows}, reverse=True)
    ns = sorted(set(metric_by_n) & {int(row["N"]) for row in kmax_rows})
    if not ns:
        return
    fig, ax1 = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    ax2 = ax1.twinx()
    ax1.plot(ns, [float(metric_by_n[n]["effective_rank"]) for n in ns], marker="o", linewidth=2.2, color="#1f77b4", label="Effective rank")
    ax1.plot(ns, [float(metric_by_n[n]["rank_1e_2"]) for n in ns], marker="s", linewidth=2.2, color="#2ca02c", label="Rank >= 1e-2")
    for threshold in thresholds:
        vals = []
        for n in ns:
            matches = [row for row in kmax_rows if int(row["N"]) == n and abs(float(row["id_threshold"]) - threshold) < 1e-9]
            vals.append(int(matches[0]["K_max"]) if matches else 0)
        ax2.plot(ns, vals, marker="^", linestyle="--", linewidth=1.8, label=f"Kmax @{threshold:.2f}")
    ax1.set_xlabel("Grid size N")
    ax1.set_ylabel("SVD rank-like metric")
    ax2.set_ylabel("Observed K_max")
    ax1.set_title("SVD sensitivity metrics vs learned K_max")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=False, loc="upper left", ncol=2)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "sensitivity_svd_vs_kmax.png", dpi=220)
    plt.close(fig)


def plot_observability_ratio_vs_kmax(metric_rows: list[dict], kmax_rows: list[dict], figure_dir: Path) -> None:
    if plt is None or not metric_rows or not kmax_rows:
        return
    apply_style()
    metric_by_n = {int(row["N"]): row for row in metric_rows}
    thresholds = [0.95, 0.90, 0.85]
    ns = sorted(set(metric_by_n) & {int(row["N"]) for row in kmax_rows})
    if not ns:
        return
    fig, ax1 = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    ax2 = ax1.twinx()
    ax1.plot(ns, [float(metric_by_n[n]["effective_rank_per_M"]) for n in ns], marker="o", linewidth=2.2, color="#1f77b4", label="Effective rank / M")
    ax1.plot(ns, [float(metric_by_n[n]["rank_1e_2_per_M"]) for n in ns], marker="s", linewidth=2.2, color="#2ca02c", label="Rank>=1e-2 / M")
    for threshold in thresholds:
        vals = []
        for n in ns:
            matches = [row for row in kmax_rows if int(row["N"]) == n and abs(float(row["id_threshold"]) - threshold) < 1e-9]
            vals.append(int(matches[0]["K_max"]) if matches else 0)
        ax2.plot(ns, vals, marker="^", linestyle="--", linewidth=1.8, label=f"Kmax @{threshold:.2f}")
    ax1.set_xlabel("Grid size N")
    ax1.set_ylabel("Per-variable SVD metric")
    ax2.set_ylabel("Observed K_max")
    ax1.set_title("Per-variable observability vs learned K_max")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=False, loc="upper right", ncol=2)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "sensitivity_svd_observability_ratio_vs_kmax.png", dpi=220)
    plt.close(fig)


def render_metric_table(metric_rows: list[dict], output_path: Path) -> None:
    if plt is None or not metric_rows:
        return
    rows = sorted(metric_rows, key=lambda item: int(item["N"]))
    table_rows = [
        [
            f"{int(row['N'])}x{int(row['N'])}",
            str(int(row["P"])),
            str(int(row["M"])),
            str(int(row["num_observations"])),
            f"{float(row['effective_rank']):.2f}",
            f"{float(row['effective_rank_per_M']):.3f}",
            f"{float(row['stable_rank']):.2f}",
            str(int(row["rank_1e_2"])),
            f"{float(row['rank_1e_2_per_M']):.3f}",
            str(int(row["energy_modes_95"])),
        ]
        for row in rows
    ]
    headers = ["Topology", "P", "M", "Rows", "Eff.", "Eff./M", "Stable", "R1e-2", "R/M", "Modes95"]
    fig_h = max(3.0, 0.45 * len(table_rows) + 1.4)
    fig, ax = plt.subplots(figsize=(10.8, fig_h), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    table.scale(1.0, 1.24)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    grid_list = [int(item) for item in args.grid_list.split(",") if item.strip()]
    summary_dir = Path(args.summary_dir).resolve()
    figure_dir = Path(args.figure_dir).resolve()
    metric_rows: list[dict] = []
    spectrum_rows: list[dict] = []

    for n in grid_list:
        j, meta = build_sensitivity_matrix(n, delta_r=args.delta_r)
        metric, s = svd_metrics(j, meta)
        metric_rows.append(metric)
        s_norm = s / max(float(s[0]), 1e-30)
        for idx, value in enumerate(s):
            spectrum_rows.append(
                {
                    "N": n,
                    "mode": idx + 1,
                    "sigma": float(value),
                    "sigma_normalized": float(s_norm[idx]),
                }
            )

    model_rows = collect_model_kmax([Path(item).resolve() for item in args.model_result_roots])
    kmax_rows = summarize_kmax(model_rows, thresholds=[0.98, 0.95, 0.90, 0.85])

    write_csv(
        summary_dir / "sensitivity_svd_metrics.csv",
        metric_rows,
        [
            "N",
            "P",
            "M",
            "num_observations",
            "excitation_count",
            "measurement_count",
            "delta_r",
            "rank_1e_1",
            "rank_1e_2",
            "rank_1e_3",
            "rank_1e_4",
            "effective_rank",
            "stable_rank",
            "condition_1e_12",
            "energy_modes_90",
            "energy_modes_95",
            "energy_modes_99",
            "sigma_max",
            "sigma_min",
            "effective_rank_per_M",
            "stable_rank_per_M",
            "rank_1e_2_per_M",
        ],
    )
    write_csv(summary_dir / "sensitivity_svd_spectrum.csv", spectrum_rows, ["N", "mode", "sigma", "sigma_normalized"])
    write_csv(summary_dir / "sensitivity_svd_model_kmax.csv", kmax_rows, ["N", "id_threshold", "K_max"])

    plot_spectrum(spectrum_rows, figure_dir)
    plot_rank_metrics(metric_rows, figure_dir)
    plot_svd_vs_kmax(metric_rows, kmax_rows, figure_dir)
    plot_observability_ratio_vs_kmax(metric_rows, kmax_rows, figure_dir)
    render_metric_table(metric_rows, figure_dir / "sensitivity_svd_metric_table.png")

    print(f"wrote_csv={summary_dir / 'sensitivity_svd_metrics.csv'}")
    print(f"wrote_csv={summary_dir / 'sensitivity_svd_spectrum.csv'}")
    print(f"wrote_csv={summary_dir / 'sensitivity_svd_model_kmax.csv'}")
    print(f"wrote_figure={figure_dir / 'sensitivity_svd_spectrum.png'}")
    print(f"wrote_figure={figure_dir / 'sensitivity_svd_rank_metrics.png'}")
    print(f"wrote_figure={figure_dir / 'sensitivity_svd_vs_kmax.png'}")
    print(f"wrote_figure={figure_dir / 'sensitivity_svd_observability_ratio_vs_kmax.png'}")
    print(f"wrote_figure={figure_dir / 'sensitivity_svd_metric_table.png'}")


if __name__ == "__main__":
    main()
