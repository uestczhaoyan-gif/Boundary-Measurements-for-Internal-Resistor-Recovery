from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_common import (
    build_boundary_excitations,
    build_square_topology,
    select_active_boundary_nodes,
)
from scripts.generate_subproject2_varcand_grid import build_candidate_edge_order


OUT_DIR = PROJECT_ROOT / "PPT_Figure" / "ppt_revision_cn"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "SimHei",
                "Microsoft YaHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 10.5,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 9.3,
        }
    )


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fval(row: dict, key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    return default if raw in ("", None) else float(raw)


def ival(row: dict, key: str, default: int = 0) -> int:
    raw = row.get(key, "")
    return default if raw in ("", None) else int(float(raw))


def savefig(fig: plt.Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return path


def read_subtask1_threshold_rows() -> list[dict]:
    rows: list[dict] = []
    for rel_path in [
        "outputs_modelg2_subproj1/modelg2_subproj1_threshold_kmax_summary.csv",
        "outputs_modelg2_task1_n78/modelg2_task1_n78_threshold_kmax_summary.csv",
    ]:
        rows.extend(read_csv(PROJECT_ROOT / rel_path))
    return rows


def read_subtask2_rows() -> list[dict]:
    return read_csv(PROJECT_ROOT / "outputs_subproj2_varcand_modelg2" / "subproject2_varcand_test_summary.csv")


def read_subtask3_rows() -> list[dict]:
    return read_csv(PROJECT_ROOT / "outputs_subproj3_activeport_modelg2" / "subproject3_activeport_test_summary.csv")


def read_subtask4_rows() -> list[dict]:
    return read_csv(PROJECT_ROOT / "outputs_subproj4_excitation_modelg2" / "subproject4_excitation_test_summary.csv")


def plot_subtask1_threshold_combined() -> Path:
    rows = read_subtask1_threshold_rows()
    thresholds = [0.95, 0.90, 0.85]
    colors = {0.95: "#d62728", 0.90: "#1f77b4", 0.85: "#111111"}

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    for threshold in thresholds:
        subset = sorted(
            [row for row in rows if abs(fval(row, "id_threshold") - threshold) < 1e-9],
            key=lambda row: ival(row, "P"),
        )
        x = [ival(row, "P") for row in subset]
        y = [ival(row, "K_max") for row in subset]
        ax.plot(
            x,
            y,
            marker="o",
            linestyle="-",
            color=colors[threshold],
            linewidth=1.75,
            markersize=5.0,
            label=f"位置识别阈值 {int(threshold * 100)}%",
        )

    ax.set_title("最大可识别电阻数随阈值变化", fontsize=14.2, weight="bold")
    ax.set_xlabel("边界端口数量 P")
    ax.set_ylabel("最大可识别变化电阻数 R_max")
    ax.set_xticks([8, 12, 16, 20, 24, 28])
    ax.set_ylim(0, 6.5)
    ax.set_yticks(range(0, 7))
    ax.legend(frameon=False, loc="upper right")
    return savefig(fig, "subtask1_threshold_combined_cn.png")


def plot_subtask1_scale_complexity() -> Path:
    ns = np.arange(3, 9, dtype=int)
    ports = 4 * ns - 4
    resistors = 2 * ns * (ns - 1)

    fig, axes = plt.subplots(2, 1, figsize=(6.45, 5.15), constrained_layout=True, sharex=True)
    axes[0].plot(ns, ports, marker="o", color="#1f77b4", linewidth=1.8, markersize=5.0, label="边界端口数量 P")
    axes[0].plot(ns, resistors, marker="s", color="#d62728", linewidth=1.8, markersize=5.0, label="总电阻数量 M")
    axes[0].set_title("端口数量与电阻数量随规模增长", fontsize=13.6, weight="bold")
    axes[0].set_ylabel("数量")
    axes[0].legend(frameon=False, loc="upper left", ncol=2)

    ratio = resistors / ports
    axes[1].plot(ns, ratio, marker="o", color="#111111", linewidth=1.8, markersize=5.0)
    axes[1].set_title("平均每个端口对应的候选电阻数量", fontsize=13.6, weight="bold")
    axes[1].set_xlabel("正方形网络规模 N×N")
    axes[1].set_ylabel("M/P")
    axes[1].set_xticks(ns)
    return savefig(fig, "subtask1_port_resistor_growth_cn.png")


def draw_square_network(
    ax: plt.Axes,
    grid_size: int,
    candidate_edges: set[int] | None = None,
    active_ports: set[int] | None = None,
    excitation_pairs: list[tuple[int, int]] | None = None,
) -> None:
    topology = build_square_topology(grid_size)
    coords = np.asarray(topology.node_coords, dtype=float)
    candidate_edges = candidate_edges or set()
    boundary = set(topology.boundary_nodes_clockwise)
    active_ports = boundary if active_ports is None else active_ports

    segments = [[coords[u], coords[v]] for u, v in topology.resistor_edges]
    colors = ["#1f77b4" if idx in candidate_edges else "#c7c7c7" for idx in range(topology.num_resistors)]
    widths = [2.1 if idx in candidate_edges else 1.0 for idx in range(topology.num_resistors)]
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=widths, capstyle="round", zorder=1))

    for node, (x, y) in enumerate(coords):
        if node in boundary:
            face = "#ffbf00" if node in active_ports else "#eeeeee"
            edge = "#333333" if node in active_ports else "#999999"
            radius = 0.036
        else:
            face = "white"
            edge = "#555555"
            radius = 0.026
        ax.add_patch(Circle((x, y), radius, facecolor=face, edgecolor=edge, linewidth=0.9, zorder=4))

    if excitation_pairs:
        for src, gnd in excitation_pairs:
            start = coords[src]
            end = coords[gnd]
            vec = end - start
            shrink = 0.08
            start2 = start + shrink * vec
            end2 = end - shrink * vec
            ax.annotate(
                "",
                xy=end2,
                xytext=start2,
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#d62728",
                    "lw": 1.35,
                    "mutation_scale": 10,
                    "alpha": 0.88,
                },
                zorder=5,
            )

    ax.set_aspect("equal")
    ax.set_xlim(-0.13, 1.13)
    ax.set_ylim(-0.13, 1.13)
    ax.axis("off")


def plot_subtask2_candidate_pool_panels() -> Path:
    grid_size = 4
    topology = build_square_topology(grid_size)
    order = build_candidate_edge_order(topology)
    c_values = [8, 12, 16, 20, 24]
    fig, axes = plt.subplots(1, len(c_values), figsize=(12.2, 2.9), constrained_layout=True)

    for ax, c_value in zip(axes, c_values):
        draw_square_network(ax, grid_size, candidate_edges=set(order[:c_value]))
        ax.set_title(f"C = {c_value}", fontsize=12)

    fig.suptitle("固定端口数量，逐步扩大候选可变电阻池", fontsize=14.5, weight="bold", y=1.08)
    fig.text(
        0.5,
        -0.01,
        "蓝色边：候选可变电阻；灰色边：固定不变；黄色节点：全部边界端口",
        ha="center",
        fontsize=10.5,
    )
    return savefig(fig, "subtask2_candidate_pool_panels_cn.png")


def plot_subtask2_id_accuracy_combined() -> Path:
    rows = read_subtask2_rows()
    ks = sorted({ival(row, "K") for row in rows})
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    for k, color in zip(ks, colors):
        group = sorted([row for row in rows if ival(row, "K") == k], key=lambda row: ival(row, "M_var"))
        x = [ival(row, "M_var") for row in group]
        y = [fval(row, "test_id_exact_rate") for row in group]
        ax.plot(x, y, marker="o", linewidth=1.35, markersize=4.7, color=color, label=f"实际变化数 R={k}")

    ax.set_title("候选电阻池扩大时的位置识别精度", fontsize=14.2, weight="bold")
    ax.set_xlabel("候选可变电阻数量 C")
    ax.set_ylabel("位置识别精度")
    ax.set_xticks([8, 12, 16, 20, 24])
    ax.set_ylim(0, 1.04)
    ax.legend(frameon=False, ncol=2, loc="lower left")
    return savefig(fig, "subtask2_varcand_id_accuracy_combined_cn.png")


def plot_subtask2_candidate_ratio() -> Path:
    rows = read_subtask2_rows()
    c_values = sorted({ival(row, "M_var") for row in rows})
    port_count = max({ival(row, "P") for row in rows})
    ratios = [c_value / port_count for c_value in c_values]

    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    ax.plot(c_values, ratios, marker="o", color="#111111", linewidth=1.8, markersize=5.0)
    ax.set_title("平均每个端口的候选可变电阻数量的变化曲线", fontsize=13.8, weight="bold")
    ax.set_xlabel("候选可变电阻数量 C")
    ax.set_ylabel("平均每个端口的候选可变电阻数量 C/P")
    ax.set_xticks(c_values)
    return savefig(fig, "subtask2_candidate_ratio_cn.png")


def plot_subtask3_active_port_panels() -> Path:
    grid_size = 4
    topology = build_square_topology(grid_size)
    p_values = [4, 6, 8, 10, 12]
    fig, axes = plt.subplots(1, len(p_values), figsize=(12.2, 2.9), constrained_layout=True)

    for ax, p_active in zip(axes, p_values):
        active_nodes = set(select_active_boundary_nodes(topology.boundary_nodes_clockwise, p_active))
        draw_square_network(
            ax,
            grid_size,
            candidate_edges=set(range(topology.num_resistors)),
            active_ports=active_nodes,
        )
        ax.set_title(f"P = {p_active}", fontsize=12)

    fig.suptitle("固定候选池，逐步增加实际启用端口数量", fontsize=14.5, weight="bold", y=1.08)
    fig.text(
        0.5,
        -0.01,
        "蓝色边：全部电阻均可变化；黄色节点：启用端口；灰色节点：未启用边界端口",
        ha="center",
        fontsize=10.5,
    )
    return savefig(fig, "subtask3_active_port_panels_cn.png")


def plot_subtask3_id_accuracy_combined() -> Path:
    rows = read_subtask3_rows()
    ks = sorted({ival(row, "K") for row in rows})
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    for k, color in zip(ks, colors):
        group = sorted([row for row in rows if ival(row, "K") == k], key=lambda row: ival(row, "P_active"))
        x = [ival(row, "P_active") for row in group]
        y = [fval(row, "test_id_exact_rate") for row in group]
        ax.plot(x, y, marker="o", linewidth=1.35, markersize=4.7, color=color, label=f"实际变化数 R={k}")

    ax.set_title("可用端口增加时的位置识别精度", fontsize=14.2, weight="bold")
    ax.set_xlabel("可用边界端口数量 P")
    ax.set_ylabel("位置识别精度")
    ax.set_xticks([4, 6, 8, 10, 12])
    ax.set_ylim(0, 1.04)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    return savefig(fig, "subtask3_activeport_id_accuracy_combined_cn.png")


def plot_subtask3_candidate_ratio() -> Path:
    rows = read_subtask3_rows()
    p_values = sorted({ival(row, "P_active") for row in rows})
    candidate_count = max({ival(row, "M") for row in rows})
    ratios = [candidate_count / p_value for p_value in p_values]

    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    ax.plot(p_values, ratios, marker="o", color="#111111", linewidth=1.8, markersize=5.0)
    ax.set_title("平均每个端口的候选可变电阻数量的变化曲线", fontsize=13.8, weight="bold")
    ax.set_xlabel("可用边界端口数量 P")
    ax.set_ylabel("平均每个端口的候选可变电阻数量 C/P")
    ax.set_xticks(p_values)
    return savefig(fig, "subtask3_candidate_ratio_cn.png")


def plot_subtask4_excitation_scheme() -> Path:
    topology = build_square_topology(4)
    e_values = [1, 4, len(topology.boundary_nodes_clockwise)]
    titles = ["E = 1", "E = 4", "E = 12（全循环）"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.25), constrained_layout=True)

    for ax, e_value, title in zip(axes, e_values, titles):
        pairs = build_boundary_excitations(topology, excitation_count=e_value)
        draw_square_network(
            ax,
            4,
            candidate_edges=set(range(topology.num_resistors)),
            excitation_pairs=pairs,
        )
        ax.set_title(title, fontsize=12)

    fig.suptitle("不同激励次数下选用的边界激励组合", fontsize=14.2, weight="bold", y=1.06)
    fig.text(0.5, -0.02, "红色箭头表示一次源端到汇端的边界激励；E 越大，覆盖的边界方向越多", ha="center", fontsize=10.2)
    return savefig(fig, "subtask4_excitation_scheme_cn.png")


def plot_subtask4_id_accuracy_by_k(k_values: list[int], name: str, title: str) -> Path:
    rows = read_subtask4_rows()
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    for k, color in zip(k_values, colors):
        group = sorted([row for row in rows if ival(row, "K") == k], key=lambda row: ival(row, "E"))
        x = [ival(row, "E") for row in group]
        y = [fval(row, "test_id_exact_rate") for row in group]
        ax.plot(x, y, marker="o", linewidth=1.45, markersize=4.8, color=color, label=f"实际变化数 R={k}")

    ax.set_title(title, fontsize=14.2, weight="bold")
    ax.set_xlabel("激励次数 E")
    ax.set_ylabel("位置识别精度")
    ax.set_xticks([1, 4, 12])
    ax.set_ylim(0, 1.04)
    ax.legend(frameon=False, loc="lower right")
    return savefig(fig, name)


def plot_subtask4_id_accuracy_combined() -> Path:
    rows = read_subtask4_rows()
    k_values = sorted({ival(row, "K") for row in rows})
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

    fig, ax = plt.subplots(figsize=(7.1, 4.35), constrained_layout=True)
    for k, color in zip(k_values, colors):
        group = sorted([row for row in rows if ival(row, "K") == k], key=lambda row: ival(row, "E"))
        x = [ival(row, "E") for row in group]
        y = [fval(row, "test_id_exact_rate") for row in group]
        ax.plot(x, y, marker="o", linewidth=1.3, markersize=4.5, color=color, label=f"实际变化数 R={k}")

    ax.set_title("激励次数增加时的位置识别精度", fontsize=14.2, weight="bold")
    ax.set_xlabel("激励次数 E")
    ax.set_ylabel("位置识别精度")
    ax.set_xticks([1, 4, 12])
    ax.set_ylim(0, 1.04)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    return savefig(fig, "subtask4_excitation_id_accuracy_combined_cn.png")


def main() -> None:
    setup_style()
    paths = [
        plot_subtask1_threshold_combined(),
        plot_subtask1_scale_complexity(),
        plot_subtask2_candidate_pool_panels(),
        plot_subtask2_id_accuracy_combined(),
        plot_subtask2_candidate_ratio(),
        plot_subtask3_active_port_panels(),
        plot_subtask3_id_accuracy_combined(),
        plot_subtask3_candidate_ratio(),
        plot_subtask4_excitation_scheme(),
        plot_subtask4_id_accuracy_combined(),
        plot_subtask4_id_accuracy_by_k(
            [1, 2, 3],
            "subtask4_excitation_id_accuracy_k1_3_cn.png",
            "低变化数量下的激励次数与位置识别精度",
        ),
        plot_subtask4_id_accuracy_by_k(
            [4, 5, 6],
            "subtask4_excitation_id_accuracy_k4_6_cn.png",
            "高变化数量下的激励次数与位置识别精度",
        ),
    ]
    print("generated:")
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
