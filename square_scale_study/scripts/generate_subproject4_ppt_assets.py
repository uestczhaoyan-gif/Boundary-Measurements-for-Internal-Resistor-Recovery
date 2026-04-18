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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from project_common import (
    BASE_R,
    DEFAULT_CURRENT_A,
    build_boundary_excitations,
    build_conductance,
    build_rhs_matrix,
    build_square_topology,
    solve_all_excitations,
)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def node_xy(topology, node_id: int) -> tuple[float, float]:
    x, y = topology.node_coords[node_id]
    return float(x), float(y)


def draw_base_grid(ax, topology, changed_edge: tuple[int, int]) -> None:
    changed = {tuple(changed_edge), tuple(reversed(changed_edge))}
    for u, v in topology.resistor_edges:
        x1, y1 = node_xy(topology, u)
        x2, y2 = node_xy(topology, v)
        if (u, v) in changed:
            ax.plot([x1, x2], [y1, y2], color="#d62728", linewidth=4.2, solid_capstyle="round", zorder=3)
        else:
            ax.plot([x1, x2], [y1, y2], color="#b7bec8", linewidth=1.8, zorder=1)

    boundary_set = set(topology.boundary_nodes_clockwise)
    for node_id in range(topology.num_nodes):
        x, y = node_xy(topology, node_id)
        if node_id in boundary_set:
            ax.scatter(x, y, s=58, facecolor="white", edgecolor="#1f77b4", linewidth=1.6, zorder=4)
        else:
            ax.scatter(x, y, s=42, color="#404040", zorder=4)

    ax.set_aspect("equal")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_excitation_arrow(ax, topology, src: int, gnd: int, color: str, rad: float) -> None:
    x1, y1 = node_xy(topology, src)
    x2, y2 = node_xy(topology, gnd)
    arrow = patches.FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=2.0,
        color=color,
        alpha=0.92,
        zorder=5,
    )
    ax.add_patch(arrow)
    ax.scatter([x1], [y1], s=78, color=color, zorder=6)
    ax.scatter([x2], [y2], s=78, facecolor="white", edgecolor=color, linewidth=2.0, zorder=6)


def generate_physical_schematic(output_path: Path) -> None:
    apply_style()
    topology = build_square_topology(4)
    changed_edge = (5, 9)
    excitation_sets = {
        1: build_boundary_excitations(topology, excitation_count=1),
        4: build_boundary_excitations(topology, excitation_count=4),
        12: build_boundary_excitations(topology, excitation_count=12),
    }
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0), constrained_layout=True)
    for ax, e in zip(axes, [1, 4, 12]):
        draw_base_grid(ax, topology, changed_edge)
        excitations = excitation_sets[e]

        if e == 1:
            draw_list = [(excitations[0], colors[0], 0.18)]
        elif e == 4:
            draw_list = [(pair, colors[idx], 0.18 if idx % 2 == 0 else -0.18) for idx, pair in enumerate(excitations)]
        else:
            chosen_idx = [0, 2, 4, 6, 8, 10]
            draw_list = [
                (excitations[idx], colors[j % len(colors)], 0.15 if j % 2 == 0 else -0.15)
                for j, idx in enumerate(chosen_idx)
            ]

        for pair, color, rad in draw_list:
            src, gnd = pair
            draw_excitation_arrow(ax, topology, src, gnd, color=color, rad=rad)

        ax.set_title(f"E = {e}")

    fig.suptitle("Physical intuition: more excitation directions illuminate the same hidden resistor", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def solve_boundary_response_for_excitation(topology, resistor_values: np.ndarray, excitation: tuple[int, int]) -> np.ndarray:
    ref_node = topology.boundary_nodes_clockwise[0]
    keep_idx, rhs = build_rhs_matrix(
        topology.num_nodes,
        ref_node=ref_node,
        excitations=[excitation],
        current_a=DEFAULT_CURRENT_A,
    )
    gmat = build_conductance(topology.num_nodes, resistor_values, topology.resistor_edges)
    voltages = solve_all_excitations(gmat, keep_idx, ref_node, rhs, [excitation])
    boundary_nodes = list(topology.boundary_nodes_clockwise)
    return voltages[boundary_nodes, 0]


def generate_boundary_response_compare(output_path: Path) -> None:
    apply_style()
    topology = build_square_topology(4)
    resistor_values = np.full(topology.num_resistors, BASE_R, dtype=np.float64)
    changed_edge = (5, 9)
    changed_rid = topology.resistor_edges.index(changed_edge)
    resistor_values[changed_rid] = 1200.0

    excitations = build_boundary_excitations(topology, excitation_count=12)
    all_profiles = []
    all_drops = []
    for excitation in excitations:
        profile = solve_boundary_response_for_excitation(topology, resistor_values, excitation)
        all_profiles.append(profile)

        ref_node = topology.boundary_nodes_clockwise[0]
        keep_idx, rhs = build_rhs_matrix(
            topology.num_nodes,
            ref_node=ref_node,
            excitations=[excitation],
            current_a=DEFAULT_CURRENT_A,
        )
        gmat = build_conductance(topology.num_nodes, resistor_values, topology.resistor_edges)
        voltages = solve_all_excitations(gmat, keep_idx, ref_node, rhs, [excitation])
        u, v = changed_edge
        all_drops.append(abs(float(voltages[u, 0] - voltages[v, 0])))

    chosen = []
    seen_levels: list[float] = []
    for idx in np.argsort(np.array(all_drops)):
        val = float(all_drops[int(idx)])
        if all(abs(val - prev) > 1e-9 for prev in seen_levels):
            chosen.append(int(idx))
            seen_levels.append(val)
        if len(chosen) == 4:
            break
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]

    boundary_profiles = []
    edge_drops = []
    labels = []
    for idx in chosen:
        boundary_profiles.append(all_profiles[idx])
        edge_drops.append(all_drops[idx])
        labels.append(f"p{idx + 1}: {idx + 1}->{(idx + 2 - 1) % 12 + 1}")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)

    ax = axes[0]
    x = np.arange(1, len(topology.boundary_nodes_clockwise) + 1)
    for profile, label, color in zip(boundary_profiles, labels, colors):
        ax.plot(x, profile, marker="o", linewidth=2.0, markersize=4.0, color=color, label=label)
    ax.set_xlabel("Boundary port index")
    ax.set_ylabel("Boundary voltage (V)")
    ax.set_title("Same hidden resistor, different boundary voltage profiles")
    ax.legend(frameon=False, ncol=2, loc="best")

    ax = axes[1]
    ax.bar(np.arange(len(labels)), edge_drops, color=colors, width=0.62)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=12)
    ax.set_ylabel("|Voltage drop| on changed resistor (V)")
    ax.set_title("The same resistor is illuminated differently")
    for i, val in enumerate(edge_drops):
        ax.text(i, val + 0.008, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Direct evidence: changing excitation pair changes both boundary response and local resistor drop", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def stacked_jacobian(topology, excitation_count: int, delta_r: float = 10.0) -> tuple[np.ndarray, np.ndarray, float, float]:
    excitations = build_boundary_excitations(topology, excitation_count=excitation_count)
    ref_node = topology.boundary_nodes_clockwise[0]
    keep_idx, rhs = build_rhs_matrix(
        topology.num_nodes,
        ref_node=ref_node,
        excitations=excitations,
        current_a=DEFAULT_CURRENT_A,
    )

    base_values = np.full(topology.num_resistors, BASE_R, dtype=np.float64)
    base_g = build_conductance(topology.num_nodes, base_values, topology.resistor_edges)
    base_v = solve_all_excitations(base_g, keep_idx, ref_node, rhs, excitations)
    boundary_nodes = list(topology.boundary_nodes_clockwise)
    base_stack = base_v[boundary_nodes, :].T.reshape(-1)

    num_obs = base_stack.shape[0]
    jacobian = np.zeros((num_obs, topology.num_resistors), dtype=np.float64)
    for rid in range(topology.num_resistors):
        values = base_values.copy()
        values[rid] += delta_r
        gmat = build_conductance(topology.num_nodes, values, topology.resistor_edges)
        voltages = solve_all_excitations(gmat, keep_idx, ref_node, rhs, excitations)
        stack = voltages[boundary_nodes, :].T.reshape(-1)
        jacobian[:, rid] = (stack - base_stack) / delta_r

    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    p = singular_values / np.sum(singular_values)
    effective_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))

    col_norms = np.linalg.norm(jacobian, axis=0, keepdims=True)
    normalized = jacobian / np.maximum(col_norms, 1e-12)
    corr = np.abs(normalized.T @ normalized)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    mean_abs_corr = float(np.mean(corr[mask]))
    return jacobian, singular_values, effective_rank, mean_abs_corr


def generate_sensitivity_diagnostics(output_path: Path) -> None:
    apply_style()
    topology = build_square_topology(4)
    results = {}
    for e in [1, 4, 12]:
        jacobian, sv, eff_rank, mean_abs_corr = stacked_jacobian(topology, excitation_count=e)
        results[e] = {
            "J": jacobian,
            "sv": sv,
            "eff_rank": eff_rank,
            "mean_abs_corr": mean_abs_corr,
        }

    fig = plt.figure(figsize=(13.0, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 6)

    vmax = max(float(np.max(np.abs(results[e]["J"]))) for e in [1, 4, 12])
    for idx, e in enumerate([1, 4, 12]):
        ax = fig.add_subplot(gs[0, idx * 2 : (idx + 1) * 2])
        im = ax.imshow(np.abs(results[e]["J"]), aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(f"|J|, E={e}")
        ax.set_xlabel("Resistor index")
        ax.set_ylabel("Observation equation")
        if idx == 2:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label("|sensitivity|")

    ax_sv = fig.add_subplot(gs[1, :3])
    colors = {1: "#1f77b4", 4: "#ff7f0e", 12: "#2ca02c"}
    for e in [1, 4, 12]:
        sv = results[e]["sv"]
        ax_sv.plot(
            np.arange(1, len(sv) + 1),
            sv / np.max(sv),
            marker="o",
            markersize=3.5,
            linewidth=2.0,
            color=colors[e],
            label=f"E={e}",
        )
    ax_sv.set_xlabel("Singular value index")
    ax_sv.set_ylabel("Normalized singular value")
    ax_sv.set_title("Stacked sensitivity spectrum")
    ax_sv.legend(frameon=False)

    ax_sum = fig.add_subplot(gs[1, 3:])
    e_vals = np.array([1, 4, 12], dtype=float)
    eff = np.array([results[e]["eff_rank"] / topology.num_resistors for e in [1, 4, 12]], dtype=float)
    indep = np.array([1.0 - results[e]["mean_abs_corr"] for e in [1, 4, 12]], dtype=float)
    ax_sum.plot(e_vals, eff, marker="o", linewidth=2.0, color="#2ca02c", label="effective-rank ratio")
    ax_sum.plot(e_vals, indep, marker="s", linewidth=2.0, color="#d62728", label="1 - mean |col corr|")
    ax_sum.set_xticks(e_vals)
    ax_sum.set_xlabel("Excitation count E")
    ax_sum.set_ylabel("Normalized score")
    ax_sum.set_ylim(0.0, 1.02)
    ax_sum.set_title("Richer span, lower ambiguity")
    ax_sum.legend(frameon=False, loc="lower right")

    fig.suptitle("Mathematical view: stacking excitation-dependent sensitivities expands useful observation space", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def load_test_summary(summary_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "E": float(row["E"]),
                    "K": float(row["K"]),
                    "test_id_exact_rate": float(row["test_id_exact_rate"]),
                    "test_value_accuracy": float(row["test_value_accuracy"]),
                }
            )
    return rows


def generate_experiment_bridge(output_path: Path) -> None:
    apply_style()
    summary_path = PROJECT_ROOT / "outputs_subproj4_excitation_modelg2" / "subproject4_excitation_test_summary.csv"
    rows = load_test_summary(summary_path)
    topology = build_square_topology(4)

    e_vals = np.array([1.0, 4.0, 12.0], dtype=float)
    avg_id = []
    hard_avg_id = []
    avg_value = []
    eff_rank_ratio = []
    indep_score = []

    for e in e_vals:
        e_rows = [row for row in rows if row["E"] == e]
        avg_id.append(float(np.mean([row["test_id_exact_rate"] for row in e_rows])))
        hard_avg_id.append(float(np.mean([row["test_id_exact_rate"] for row in e_rows if row["K"] >= 3])))
        avg_value.append(float(np.mean([row["test_value_accuracy"] for row in e_rows])))

        _jacobian, _sv, eff_rank, mean_abs_corr = stacked_jacobian(topology, excitation_count=int(e))
        eff_rank_ratio.append(float(eff_rank / topology.num_resistors))
        indep_score.append(float(1.0 - mean_abs_corr))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), constrained_layout=True)

    ax = axes[0]
    ax.plot(e_vals, avg_id, marker="o", linewidth=2.2, color="#1f77b4", label="mean ID accuracy")
    ax.plot(e_vals, hard_avg_id, marker="s", linewidth=2.2, color="#d62728", label="hard-case ID (K>=3)")
    ax.plot(e_vals, avg_value, marker="^", linewidth=2.2, color="#2ca02c", label="mean value accuracy")
    ax.set_xticks(e_vals)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Excitation count E")
    ax.set_ylabel("Accuracy")
    ax.set_title("Observed performance gain")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    ax.plot(e_vals, eff_rank_ratio, marker="o", linewidth=2.2, color="#9467bd", label="effective-rank ratio")
    ax.plot(e_vals, indep_score, marker="s", linewidth=2.2, color="#ff7f0e", label="1 - mean |column corr|")
    ax.set_xticks(e_vals)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Excitation count E")
    ax.set_ylabel("Normalized score")
    ax.set_title("Why the gain appears")
    ax.legend(frameon=False, loc="lower right")

    fig.suptitle("Experiment-theory bridge: more excitation directions bring richer and less ambiguous observations", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def write_short_markdown(output_path: Path) -> None:
    text = """# 子项目 4：循环激励增加信息通道（PPT短版）

## 目的

- 固定 `4x4` 拓扑和全部 `12` 个测量端口，只改变激励次数 `E`，判断多端口循环激励带来的收益究竟来自“新增有效信息”，还是“仅提升数值稳定性”。

## 实践

- 比较 `E = 1 / 4 / 12` 在 `K = 1..6` 下的 `ID / Value` 精度，并结合不同 `ID` 阈值下的 `K_max` 判断信息增益是否真实存在。

## 结果

- `E=1 -> 4` 时，`ID` 精度在 `K>=2` 后明显跃升，而 `Value` 精度始终较高，说明提升主要来自“更容易区分不同变化模式”。
- `E=4 -> 12` 时，总体增益变缓，但在 `K>=3` 的复杂场景下仍有稳定收益，说明新增激励仍在提供新的可辨识信息。

## 解释

- 不同激励会形成不同电流分布，相当于从不同方向“照亮”同一个内部电阻网络。
- 数学上可写为 `Δv_stack ≈ [J_1; J_2; ...; J_E] Δr`；当新增激励对应的灵敏度方向不重复时，就会提高有效秩、缩小不可辨识空间，从而提升 support 恢复能力。

## 一句话结论

- 多端口循环激励带来的提升不只是“更稳”，而是确实增加了对内部电阻变化的有效观测信息。"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate short PPT text and figures for subproject4 explanation.")
    parser.add_argument(
        "--figure-dir",
        default=str(PROJECT_ROOT / "Figure" / "subproject4_excitation"),
    )
    parser.add_argument(
        "--text-path",
        default=str(PROJECT_ROOT / "subproject4_info_channel_explanation_ppt_short.md"),
    )
    args = parser.parse_args()

    figure_dir = Path(args.figure_dir).resolve()
    text_path = Path(args.text_path).resolve()

    generate_physical_schematic(figure_dir / "subproject4_excitation_physical_schematic.png")
    generate_boundary_response_compare(figure_dir / "subproject4_excitation_boundary_response_compare.png")
    generate_sensitivity_diagnostics(figure_dir / "subproject4_excitation_sensitivity_diagnostics.png")
    generate_experiment_bridge(figure_dir / "subproject4_excitation_info_bridge.png")
    write_short_markdown(text_path)

    print(f"wrote_text={text_path}")
    print(f"wrote_figure={figure_dir / 'subproject4_excitation_physical_schematic.png'}")
    print(f"wrote_figure={figure_dir / 'subproject4_excitation_boundary_response_compare.png'}")
    print(f"wrote_figure={figure_dir / 'subproject4_excitation_sensitivity_diagnostics.png'}")
    print(f"wrote_figure={figure_dir / 'subproject4_excitation_info_bridge.png'}")


if __name__ == "__main__":
    main()
