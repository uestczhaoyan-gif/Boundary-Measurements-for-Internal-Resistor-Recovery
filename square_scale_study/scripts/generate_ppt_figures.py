from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import prepend_vendor_dir
from project_common import build_square_topology, edge_depths

prepend_vendor_dir(PROJECT_ROOT / ".vendor_plot", required_version=(3, 11))

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


OUT_DIR = PROJECT_ROOT / "PPT_Figure"
THRESHOLDS = [0.95, 0.90, 0.85, 0.80]


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
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "figure.dpi": 140,
        }
    )


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def i(row: dict, key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return int(float(value))


def savefig(fig: plt.Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def kmax_rows(rows: list[dict], group_key: str, thresholds: list[float] = THRESHOLDS) -> dict[float, dict[int, int]]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(i(row, group_key), []).append(row)
    result: dict[float, dict[int, int]] = {threshold: {} for threshold in thresholds}
    for group_value, group in grouped.items():
        for threshold in thresholds:
            passed = [
                i(item, "K")
                for item in group
                if f(item, "test_id_exact_rate") >= threshold and f(item, "test_value_accuracy") >= 0.90
            ]
            result[threshold][group_value] = max(passed) if passed else 0
    return result


def topology_segments(n: int, values: np.ndarray | None = None):
    topo = build_square_topology(n)
    coords = np.asarray(topo.node_coords, dtype=float)
    segments = []
    colors = []
    for rid, (u, v) in enumerate(topo.resistor_edges):
        segments.append([coords[u], coords[v]])
        if values is not None:
            colors.append(values[rid])
    return topo, coords, segments, colors


def draw_network(
    ax: plt.Axes,
    n: int = 4,
    changed: set[int] | None = None,
    candidates: set[int] | None = None,
    active_ports: set[int] | None = None,
    title: str | None = None,
) -> None:
    changed = changed or set()
    candidates = candidates or set()
    topo, coords, segments, _colors = topology_segments(n)
    base_colors = []
    widths = []
    for rid in range(len(topo.resistor_edges)):
        if rid in changed:
            base_colors.append("#d62728")
            widths.append(4.0)
        elif rid in candidates:
            base_colors.append("#1f77b4")
            widths.append(2.2)
        else:
            base_colors.append("#9a9a9a")
            widths.append(1.4)
    ax.add_collection(LineCollection(segments, colors=base_colors, linewidths=widths, capstyle="round"))
    boundary = set(topo.boundary_nodes_clockwise)
    active_ports = active_ports or boundary
    for idx, (x, y) in enumerate(coords):
        if idx in boundary:
            color = "#ffbf00" if idx in active_ports else "#eeeeee"
            edge = "#333333" if idx in active_ports else "#9a9a9a"
            size = 0.035
        else:
            color = "white"
            edge = "#555555"
            size = 0.026
        ax.add_patch(Circle((x, y), size, facecolor=color, edgecolor=edge, linewidth=1.0, zorder=4))
    ax.set_aspect("equal")
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.15)
    ax.axis("off")
    if title:
        ax.set_title(title, pad=8)


def figure_problem_definition() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
    draw_network(axes[0], 4, changed={5, 11}, title="固定正方形电阻网络")
    axes[0].text(0.5, -0.09, "内部电阻可能发生变化", ha="center", va="top", transform=axes[0].transAxes)

    draw_network(axes[1], 4, active_ports=set(build_square_topology(4).boundary_nodes_clockwise), title="边界端口激励与测量")
    axes[1].add_patch(FancyArrowPatch((-0.07, 1.02), (0.17, 1.02), arrowstyle="->", mutation_scale=15, color="#d62728", linewidth=2.2))
    axes[1].text(0.08, 1.08, "输入电流", ha="center", color="#d62728")
    axes[1].text(0.72, -0.10, "读取边界电压响应", ha="center", color="#1f77b4")

    axes[2].axis("off")
    axes[2].set_title("反演目标")
    boxes = [
        ("已知", "拓扑结构、边界激励、边界电压"),
        ("未知", "哪些电阻变化、变化幅值是多少"),
        ("输出", "变化电阻位置识别准确率 + 数值回归精度"),
    ]
    for j, (head, body) in enumerate(boxes):
        y = 0.78 - j * 0.26
        axes[2].add_patch(Rectangle((0.05, y - 0.11), 0.9, 0.17, facecolor="#f7f7f7", edgecolor="#333333", linewidth=1.1))
        axes[2].text(0.12, y - 0.02, head, weight="bold", ha="left", va="center", fontsize=13)
        axes[2].text(0.32, y - 0.02, body, ha="left", va="center", fontsize=12)
    fig.suptitle("问题定义：通过边界电压反演内部电阻变化", fontsize=16, y=1.15)
    return savefig(fig, "00_问题定义_边界电压反演内部电阻变化.png")


def figure_subtask1() -> Path:
    rows = read_csv(PROJECT_ROOT / "outputs_modelg2_subproj1" / "modelg2_subproj1_test_summary.csv")
    rows += read_csv(PROJECT_ROOT / "outputs_modelg2_task1_n78" / "modelg2_task1_n78_test_summary.csv")
    rows = sorted(rows, key=lambda r: (i(r, "N"), i(r, "K")))
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8), constrained_layout=True)

    kmax = kmax_rows(rows, "P")
    ports = sorted({i(row, "P") for row in rows})
    for threshold, marker in zip(THRESHOLDS, ["o", "s", "^", "D"]):
        axes[0].plot(
            ports,
            [kmax[threshold].get(port, 0) for port in ports],
            marker=marker,
            linewidth=2.2,
            label=f"位置准确率阈值 {int(threshold * 100)}%",
        )
    axes[0].set_xlabel("边界端口数量 P")
    axes[0].set_ylabel("最大可识别变化电阻数量 K_max")
    axes[0].set_title("端口数量与最大可识别变化数量")
    axes[0].set_xticks(ports)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False)

    ns = sorted({i(row, "N") for row in rows})
    ks = sorted({i(row, "K") for row in rows})
    matrix = np.full((len(ns), len(ks)), np.nan)
    for row in rows:
        matrix[ns.index(i(row, "N")), ks.index(i(row, "K"))] = f(row, "test_id_exact_rate") * 100
    im = axes[1].imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    axes[1].set_xticks(range(len(ks)), [str(k) for k in ks])
    axes[1].set_yticks(range(len(ns)), [f"{n}×{n}" for n in ns])
    axes[1].set_xlabel("变化电阻数量 K")
    axes[1].set_ylabel("网络规模 N×N")
    axes[1].set_title("不同规模与 K 下的位置识别准确率")
    for y in range(len(ns)):
        for x in range(len(ks)):
            if not math.isnan(matrix[y, x]):
                axes[1].text(x, y, f"{matrix[y, x]:.0f}%", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=axes[1], shrink=0.88)
    cbar.set_label("位置识别准确率")
    fig.suptitle("子任务1：网络规模/端口数量对 K_max 的影响", fontsize=16, y=1.10)
    return savefig(fig, "01_子任务1_端口数量与最大可识别变化电阻数量.png")


def figure_subtask2() -> Path:
    rows = read_csv(PROJECT_ROOT / "outputs_subproj2_varcand_modelg2" / "subproject2_varcand_test_summary.csv")
    rows = sorted(rows, key=lambda r: (i(r, "M_var"), i(r, "K")))
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8), constrained_layout=True)
    mvars = sorted({i(row, "M_var") for row in rows})
    ks = sorted({i(row, "K") for row in rows})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(ks), 3)))
    for color, k in zip(colors, ks):
        group = [row for row in rows if i(row, "K") == k]
        axes[0].plot(
            [i(row, "M_var") for row in group],
            [f(row, "test_id_exact_rate") * 100 for row in group],
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"K={k}",
        )
    axes[0].set_xlabel("候选可变电阻总数 M_var")
    axes[0].set_ylabel("变化电阻位置识别准确率（%）")
    axes[0].set_title("候选集合增大后的识别精度变化")
    axes[0].set_xticks(mvars)
    axes[0].set_ylim(0, 105)
    axes[0].legend(frameon=False, ncol=2)

    kmax = kmax_rows(rows, "M_var")
    for threshold, marker in zip(THRESHOLDS, ["o", "s", "^", "D"]):
        axes[1].plot(
            mvars,
            [kmax[threshold].get(mvar, 0) for mvar in mvars],
            marker=marker,
            linewidth=2.2,
            label=f"阈值 {int(threshold * 100)}%",
        )
    axes[1].set_xlabel("候选可变电阻总数 M_var")
    axes[1].set_ylabel("最大可识别变化电阻数量 K_max")
    axes[1].set_title("候选集合规模与 K_max")
    axes[1].set_xticks(mvars)
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=False)
    fig.suptitle("子任务2：候选可变电阻总数对识别难度的影响", fontsize=16, y=1.10)
    return savefig(fig, "02_子任务2_候选可变电阻总数与识别难度.png")


def figure_subtask3() -> Path:
    rows = read_csv(PROJECT_ROOT / "outputs_subproj3_activeport_modelg2" / "subproject3_activeport_test_summary.csv")
    rows = sorted(rows, key=lambda r: (i(r, "P_active"), i(r, "K")))
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8), constrained_layout=True)
    pvals = sorted({i(row, "P_active") for row in rows})
    ks = sorted({i(row, "K") for row in rows})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(ks), 3)))
    for color, k in zip(colors, ks):
        group = [row for row in rows if i(row, "K") == k]
        axes[0].plot(
            [i(row, "P_active") for row in group],
            [f(row, "test_id_exact_rate") * 100 for row in group],
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"K={k}",
        )
    axes[0].set_xlabel("活动端口数量 P_active")
    axes[0].set_ylabel("变化电阻位置识别准确率（%）")
    axes[0].set_title("可用端口增加后的识别精度变化")
    axes[0].set_xticks(pvals)
    axes[0].set_ylim(0, 105)
    axes[0].legend(frameon=False, ncol=2)

    kmax = kmax_rows(rows, "P_active")
    for threshold, marker in zip(THRESHOLDS, ["o", "s", "^", "D"]):
        axes[1].plot(
            pvals,
            [kmax[threshold].get(p, 0) for p in pvals],
            marker=marker,
            linewidth=2.2,
            label=f"阈值 {int(threshold * 100)}%",
        )
    axes[1].set_xlabel("活动端口数量 P_active")
    axes[1].set_ylabel("最大可识别变化电阻数量 K_max")
    axes[1].set_title("活动端口数量与 K_max")
    axes[1].set_xticks(pvals)
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=False)
    fig.suptitle("子任务3：活动端口数量对识别能力的影响", fontsize=16, y=1.10)
    return savefig(fig, "03_子任务3_活动端口数量与识别能力.png")


def figure_subtask4() -> Path:
    rows = read_csv(PROJECT_ROOT / "outputs_subproj4_excitation_modelg2" / "subproject4_excitation_test_summary.csv")
    rows = sorted(rows, key=lambda r: (i(r, "K"), i(r, "E")))
    es = sorted({i(row, "E") for row in rows})
    ks = sorted({i(row, "K") for row in rows})
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.6), constrained_layout=True)
    for ax, k in zip(axes.ravel(), ks[:6]):
        group = [row for row in rows if i(row, "K") == k]
        ax.plot([i(row, "E") for row in group], [f(row, "test_id_exact_rate") * 100 for row in group], marker="o", linewidth=2.0, label="位置识别")
        ax.plot([i(row, "E") for row in group], [f(row, "test_value_accuracy") * 100 for row in group], marker="s", linewidth=2.0, label="数值回归")
        ax.set_title(f"变化电阻数量 K={k}")
        ax.set_xlabel("激励次数 E")
        ax.set_ylabel("测试集准确率（%）")
        ax.set_xticks(es)
        ax.set_ylim(0, 105)
        ax.legend(frameon=False, loc="lower right")
    fig.suptitle("子任务4：激励次数对位置识别和数值回归的影响", fontsize=16, y=1.04)
    return savefig(fig, "04_子任务4_激励次数与可辨识信息.png")


def figure_subtask5() -> Path:
    rows = read_csv(PROJECT_ROOT / "outputs_subproj5_depth_followup" / "subproject5_depth_aggregate_rows.csv")
    rows = [row for row in rows if row.get("source") in {"multiseed", "single_n6"}]
    rows = sorted(rows, key=lambda r: (i(r, "N"), i(r, "depth_shell")))
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8), constrained_layout=True)
    ns = sorted({i(row, "N") for row in rows})
    width = 0.22
    all_shells = sorted({i(row, "depth_shell") for row in rows})
    x = np.arange(len(all_shells))
    for idx, n in enumerate(ns):
        group = {i(row, "depth_shell"): row for row in rows if i(row, "N") == n}
        y = [f(group[s], "id_mean") * 100 if s in group else np.nan for s in all_shells]
        axes[0].bar(x + (idx - (len(ns) - 1) / 2) * width, y, width=width, label=f"{n}×{n}")
    axes[0].set_xticks(x, [f"深度{s}" if s else "外层" for s in all_shells])
    axes[0].set_xlabel("电阻边距离边界的深度层")
    axes[0].set_ylabel("变化电阻位置识别准确率（%）")
    axes[0].set_title("不同深度层的识别精度")
    axes[0].set_ylim(85, 101)
    axes[0].legend(frameon=False)

    for idx, n in enumerate(ns):
        group = {i(row, "depth_shell"): row for row in rows if i(row, "N") == n}
        y = [f(group[s], "sample_count_mean") if s in group else np.nan for s in all_shells]
        axes[1].bar(x + (idx - (len(ns) - 1) / 2) * width, y, width=width, label=f"{n}×{n}")
    axes[1].set_xticks(x, [f"深度{s}" if s else "外层" for s in all_shells])
    axes[1].set_xlabel("电阻边距离边界的深度层")
    axes[1].set_ylabel("测试样本数量")
    axes[1].set_title("不同深度层的样本/候选数量差异")
    axes[1].legend(frameon=False)
    fig.suptitle("子任务5：电阻深度与识别难度的关系", fontsize=16, y=1.10)
    return savefig(fig, "05_子任务5_电阻深度与识别难度.png")


def figure_svd() -> Path:
    metrics = read_csv(PROJECT_ROOT / "outputs_sensitivity_svd" / "sensitivity_svd_metrics.csv")
    spectrum = read_csv(PROJECT_ROOT / "outputs_sensitivity_svd" / "sensitivity_svd_spectrum.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8), constrained_layout=True)
    for n in sorted({i(row, "N") for row in spectrum}):
        group = sorted([row for row in spectrum if i(row, "N") == n], key=lambda row: i(row, "mode"))
        axes[0].semilogy(
            [i(row, "mode") for row in group],
            [f(row, "sigma_normalized") for row in group],
            marker="o",
            markersize=2.4,
            linewidth=1.5,
            label=f"{n}×{n}",
        )
    axes[0].set_xlabel("奇异值序号（从强到弱排序）")
    axes[0].set_ylabel("归一化奇异值")
    axes[0].set_title("电学灵敏度谱")
    axes[0].legend(frameon=False, ncol=2)

    metrics = sorted(metrics, key=lambda row: i(row, "N"))
    axes[1].plot([i(row, "P") for row in metrics], [f(row, "effective_rank_per_M") for row in metrics], marker="o", linewidth=2.2, label="有效秩 / 电阻总数")
    axes[1].plot([i(row, "P") for row in metrics], [f(row, "rank_1e_2_per_M") for row in metrics], marker="s", linewidth=2.2, label="强模式数量 / 电阻总数")
    axes[1].set_xlabel("边界端口数量 P")
    axes[1].set_ylabel("单位电阻平均可观测信息")
    axes[1].set_title("规模增大后每条电阻的信息占比")
    axes[1].legend(frameon=False)
    fig.suptitle("补充分析：SVD 灵敏度矩阵反映边界测量的信息结构", fontsize=16, y=1.10)
    return savefig(fig, "06_补充分析_SVD灵敏度谱与信息占比.png")


def figure_animation_storyboard() -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.4), constrained_layout=True)
    titles = [
        "定义问题：边界测量反演内部变化",
        "子任务1：改变网络规模",
        "子任务2：改变候选可变电阻数",
        "子任务3：改变活动端口数",
        "子任务4：改变激励次数",
        "子任务5：改变电阻深度",
    ]
    topo = build_square_topology(4)
    boundary = list(topo.boundary_nodes_clockwise)
    draw_network(axes[0, 0], 4, changed={5, 11}, title=titles[0])
    draw_network(axes[0, 1], 5, changed={10}, title=titles[1])
    draw_network(axes[0, 2], 4, candidates=set(range(8)), changed={3}, title=titles[2])
    draw_network(axes[1, 0], 4, active_ports=set(boundary[::3]), changed={4}, title=titles[3])
    draw_network(axes[1, 1], 4, changed={7}, title=titles[4])
    axes[1, 1].add_patch(FancyArrowPatch((-0.08, 1.02), (0.15, 1.02), arrowstyle="->", mutation_scale=13, color="#d62728", linewidth=2.0))
    axes[1, 1].add_patch(FancyArrowPatch((1.08, 0.25), (1.08, 0.55), arrowstyle="->", mutation_scale=13, color="#1f77b4", linewidth=2.0))
    depths = edge_depths(topo)
    draw_network(axes[1, 2], 4, candidates={rid for rid, d in enumerate(depths) if d == 1}, title=titles[5])
    return savefig(fig, "动画分镜图_五个子任务怎么演示.png")


def make_animation() -> Path | None:
    topo = build_square_topology(4)
    boundary = list(topo.boundary_nodes_clockwise)
    depths = edge_depths(topo)
    frames = [
        ("固定拓扑：内部电阻发生变化", {"changed": {5, 11}}),
        ("边界端口：施加激励并读取电压", {"changed": {5, 11}, "active_ports": set(boundary)}),
        ("子任务1：改变网络规模，观察 K_max", {"n": 5, "changed": {10, 23}}),
        ("子任务2：逐步扩大候选可变电阻集合", {"changed": {3}, "candidates": set(range(12))}),
        ("子任务3：减少或增加活动端口数量", {"changed": {4}, "active_ports": set(boundary[::2])}),
        ("子任务4：改变激励次数，比较信息增益", {"changed": {7}, "active_ports": set(boundary)}),
        ("子任务5：比较外层与内层电阻识别难度", {"changed": {rid for rid, d in enumerate(depths) if d == 1}, "candidates": {rid for rid, d in enumerate(depths) if d == 1}}),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    def update(frame_idx: int):
        ax.clear()
        text, raw_kwargs = frames[frame_idx // 12]
        kwargs = dict(raw_kwargs)
        n = int(kwargs.pop("n", 4)) if "n" in kwargs else 4
        draw_network(ax, n=n, **kwargs)
        ax.text(0.5, 1.08, text, ha="center", va="center", transform=ax.transAxes, fontsize=15, weight="bold")
        ax.text(0.5, -0.08, "实践变量改变 → 测试位置识别准确率与最大可识别变化电阻数量", ha="center", va="center", transform=ax.transAxes, fontsize=11)
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(frames) * 12, interval=160, blit=False)
    path = OUT_DIR / "动画_五个子任务实践流程.gif"
    try:
        ani.save(path, writer=animation.PillowWriter(fps=6))
    except Exception:
        plt.close(fig)
        return None
    plt.close(fig)
    return path


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        figure_problem_definition(),
        figure_subtask1(),
        figure_subtask2(),
        figure_subtask3(),
        figure_subtask4(),
        figure_subtask5(),
        figure_svd(),
        figure_animation_storyboard(),
    ]
    gif = make_animation()
    if gif is not None:
        paths.append(gif)
    print("generated:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
