from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import prepend_vendor_dir
from project_common import (
    build_boundary_excitations,
    build_square_topology,
    edge_depths,
    select_active_boundary_nodes,
)
from scripts.generate_subproject2_varcand_grid import build_candidate_edge_order

prepend_vendor_dir(PROJECT_ROOT / ".vendor_plot", required_version=(3, 11))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from PIL import Image


OUT_DIR = PROJECT_ROOT / "PPT_Figure" / "animation_detailed"
ID_THRESHOLD = 0.90
FRAME_SIZE = (1600, 900)
FRAME_MARGIN = 54


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
            "font.size": 12,
        }
    )


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(row: dict, key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw in ("", None):
        return default
    return float(raw)


def integer(row: dict, key: str, default: int = 0) -> int:
    return int(num(row, key, default))


def compute_kmax(rows: list[dict], group_key: str, threshold: float = ID_THRESHOLD) -> dict[int, int]:
    groups: dict[int, list[dict]] = {}
    for row in rows:
        groups.setdefault(integer(row, group_key), []).append(row)
    out: dict[int, int] = {}
    for group_value, items in groups.items():
        passed = [
            integer(item, "K")
            for item in items
            if num(item, "test_id_exact_rate") >= threshold and num(item, "test_value_accuracy") >= 0.90
        ]
        out[group_value] = max(passed) if passed else 0
    return out


def edge_segments(n: int):
    topo = build_square_topology(n)
    coords = np.asarray(topo.node_coords, dtype=float)
    segments = [[coords[u], coords[v]] for u, v in topo.resistor_edges]
    return topo, coords, segments


def pick_edges(n: int, count: int, pool: list[int] | None = None) -> set[int]:
    if count <= 0:
        return set()
    topo = build_square_topology(n)
    choices = list(range(topo.num_resistors)) if pool is None else list(pool)
    rng = np.random.default_rng(20260424 + n * 31 + count * 17 + len(choices))
    if not choices:
        return set()
    count = min(count, len(choices))
    return set(int(v) for v in rng.choice(choices, size=count, replace=False))


def draw_network(
    ax: plt.Axes,
    n: int,
    title: str,
    changed_edges: set[int] | None = None,
    candidate_edges: set[int] | None = None,
    active_ports: set[int] | None = None,
    excitation_count: int | None = None,
    depth_edges: set[int] | None = None,
) -> None:
    changed_edges = changed_edges or set()
    candidate_edges = candidate_edges or set()
    depth_edges = depth_edges or set()
    topo, coords, segments = edge_segments(n)
    colors = []
    widths = []
    for rid in range(topo.num_resistors):
        if rid in changed_edges:
            colors.append("#d62728")
            widths.append(4.0)
        elif rid in depth_edges:
            colors.append("#9467bd")
            widths.append(3.0)
        elif rid in candidate_edges:
            colors.append("#1f77b4")
            widths.append(2.4)
        else:
            colors.append("#b8b8b8")
            widths.append(1.35)
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=widths, capstyle="round"))

    boundary = set(topo.boundary_nodes_clockwise)
    active_ports = boundary if active_ports is None else active_ports
    for idx, (x, y) in enumerate(coords):
        if idx in boundary:
            face = "#ffbf00" if idx in active_ports else "#eeeeee"
            edge = "#333333" if idx in active_ports else "#9a9a9a"
            size = 0.035
        else:
            face = "white"
            edge = "#555555"
            size = 0.026
        ax.add_patch(Circle((x, y), size, facecolor=face, edgecolor=edge, linewidth=1.0, zorder=4))

    if excitation_count is not None and excitation_count > 0:
        excitations = build_boundary_excitations(topo, excitation_count=excitation_count)
        draw_count = min(len(excitations), 8)
        for src, gnd in excitations[:draw_count]:
            sx, sy = coords[src]
            gx, gy = coords[gnd]
            ax.add_patch(
                FancyArrowPatch(
                    (sx, sy),
                    (gx, gy),
                    arrowstyle="->",
                    mutation_scale=13,
                    color="#d62728",
                    linewidth=1.6,
                    alpha=0.8,
                    zorder=5,
                )
            )
        if len(excitations) > draw_count:
            ax.text(0.5, -0.10, f"仅示意前 {draw_count} 组箭头，共 {len(excitations)} 次激励", ha="center", transform=ax.transAxes, fontsize=10)

    ax.set_aspect("equal")
    ax.set_xlim(-0.16, 1.16)
    ax.set_ylim(-0.16, 1.16)
    ax.axis("off")
    ax.set_title(title, pad=8, fontsize=14)


def frame_to_image(fig: plt.Figure) -> Image.Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    max_w = FRAME_SIZE[0] - 2 * FRAME_MARGIN
    max_h = FRAME_SIZE[1] - 2 * FRAME_MARGIN
    image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", FRAME_SIZE, "white")
    x = (FRAME_SIZE[0] - image.width) // 2
    y = (FRAME_SIZE[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas.convert("P", palette=Image.ADAPTIVE)


def make_text_frame(title: str, purpose: str, practice: str, note: str = "") -> Image.Image:
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    ax.axis("off")
    ax.add_patch(Rectangle((0.03, 0.12), 0.94, 0.76, facecolor="#f7f7f7", edgecolor="#333333", linewidth=1.3))
    ax.text(0.5, 0.78, title, ha="center", va="center", fontsize=22, weight="bold")
    ax.text(0.11, 0.57, "目的", ha="left", va="center", fontsize=17, weight="bold", color="#1f77b4")
    ax.text(0.22, 0.57, purpose, ha="left", va="center", fontsize=15)
    ax.text(0.11, 0.40, "实践", ha="left", va="center", fontsize=17, weight="bold", color="#2ca02c")
    ax.text(0.22, 0.40, practice, ha="left", va="center", fontsize=15)
    if note:
        ax.text(0.5, 0.23, note, ha="center", va="center", fontsize=13, color="#555555")
    return frame_to_image(fig)


def make_network_frame(
    title: str,
    subtitle: str,
    n: int,
    changed_edges: set[int] | None = None,
    candidate_edges: set[int] | None = None,
    active_ports: set[int] | None = None,
    excitation_count: int | None = None,
    depth_edges: set[int] | None = None,
    footer: str = "",
) -> Image.Image:
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    draw_network(
        ax,
        n=n,
        title=subtitle,
        changed_edges=changed_edges,
        candidate_edges=candidate_edges,
        active_ports=active_ports,
        excitation_count=excitation_count,
        depth_edges=depth_edges,
    )
    fig.suptitle(title, fontsize=18, weight="bold", y=0.98)
    if footer:
        fig.text(0.5, 0.045, footer, ha="center", fontsize=12)
    return frame_to_image(fig)


def make_multi_scale_frame(kmax: dict[int, int]) -> Image.Image:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True)
    for ax, n in zip(axes.ravel(), [3, 4, 5, 6]):
        k = kmax.get(4 * n - 4, 0)
        changed = pick_edges(n, k)
        draw_network(ax, n=n, title=f"{n}×{n}，端口数 P={4*n-4}，K_max={k}", changed_edges=changed)
    fig.suptitle("子任务1实践：依次改变网络规模并记录 K_max", fontsize=18, weight="bold", y=1.03)
    return frame_to_image(fig)


def detailed_frames() -> list[Image.Image]:
    frames: list[Image.Image] = []

    sub1_rows = read_csv(PROJECT_ROOT / "outputs_modelg2_subproj1" / "modelg2_subproj1_test_summary.csv")
    sub1_kmax = compute_kmax(sub1_rows, "P")
    sub2_rows = read_csv(PROJECT_ROOT / "outputs_subproj2_varcand_modelg2" / "subproject2_varcand_test_summary.csv")
    sub2_kmax = compute_kmax(sub2_rows, "M_var")
    sub3_rows = read_csv(PROJECT_ROOT / "outputs_subproj3_activeport_modelg2" / "subproject3_activeport_test_summary.csv")
    sub3_kmax = compute_kmax(sub3_rows, "P_active")
    sub4_rows = read_csv(PROJECT_ROOT / "outputs_subproj4_excitation_modelg2" / "subproject4_excitation_test_summary.csv")
    sub4_kmax = compute_kmax(sub4_rows, "E")

    frames.append(
        make_text_frame(
            "研究对象",
            "通过边界电压反演内部电阻变化位置和幅值",
            "固定正方形拓扑，改变关键实验变量，观察位置识别与 K_max",
            "红色边表示模型需要识别的变化电阻；黄色节点表示可用边界端口",
        )
    )

    frames.append(
        make_text_frame(
            "子任务1：规模/端口数",
            "判断网络规模与端口数量增加后，最大可识别变化电阻数量如何变化",
            "依次训练 3×3、4×4、5×5、6×6，并在同一阈值下统计 K_max",
            f"动画中 K_max 使用位置识别阈值 {int(ID_THRESHOLD * 100)}% 作为示例",
        )
    )
    frames.append(make_multi_scale_frame(sub1_kmax))

    frames.append(
        make_text_frame(
            "子任务2：候选可变电阻总数",
            "判断固定端口时，候选可变电阻越多是否越难识别",
            "固定 4×4 和全端口，只改变允许变化的候选电阻集合 M_var",
            "蓝色为可变候选边，灰色为固定不变边，红色为示例变化边",
        )
    )
    topo4 = build_square_topology(4)
    candidate_order = build_candidate_edge_order(topo4)
    for mvar in [8, 12, 16, 20, 24]:
        candidates = set(candidate_order[:mvar])
        kmax = sub2_kmax.get(mvar, 0)
        changed = pick_edges(4, kmax, list(candidates))
        frames.append(
            make_network_frame(
                "子任务2实践：逐步扩大候选可变电阻集合",
                f"候选可变电阻总数 M_var={mvar}，K_max={kmax}",
                4,
                changed_edges=changed,
                candidate_edges=candidates,
                footer="蓝色：允许变化；灰色：固定不变；红色：示例变化电阻",
            )
        )

    frames.append(
        make_text_frame(
            "子任务3：活动端口数量",
            "判断可用边界端口数量是否限制内部变化识别能力",
            "固定 4×4 和全部候选电阻，只改变同时参与激励和测量的活动端口数",
            "黄色节点为活动端口，灰色边界节点暂不参与激励和测量",
        )
    )
    boundary = list(topo4.boundary_nodes_clockwise)
    for p_active in [4, 6, 8, 10, 12]:
        active = set(select_active_boundary_nodes(boundary, p_active))
        kmax = sub3_kmax.get(p_active, 0)
        changed = pick_edges(4, kmax)
        frames.append(
            make_network_frame(
                "子任务3实践：改变活动端口数量",
                f"活动端口 P_active={p_active}，K_max={kmax}",
                4,
                changed_edges=changed,
                active_ports=active,
                footer="端口越少，边界观测越少；端口越多，位置识别通常更稳定",
            )
        )

    frames.append(
        make_text_frame(
            "子任务4：激励次数",
            "判断多次循环激励是否提供额外可辨识信息",
            "固定 4×4、全边可变和全端口测量，只改变激励次数 E",
            "红色箭头表示不同源-汇端口组合形成的激励",
        )
    )
    for e in [1, 4, 12]:
        kmax = sub4_kmax.get(e, 0)
        changed = pick_edges(4, kmax)
        frames.append(
            make_network_frame(
                "子任务4实践：改变激励次数",
                f"激励次数 E={e}，K_max={kmax}",
                4,
                changed_edges=changed,
                excitation_count=e,
                footer="多激励提供互补边界响应，但不代表每次激励完全独立",
            )
        )

    frames.append(
        make_text_frame(
            "子任务5：电阻深度",
            "判断电阻离边界远近是否影响识别难度",
            "固定 K=1，按电阻边深度层统计位置识别与数值回归精度",
            "紫色表示当前关注的深度层，红色表示示例变化电阻",
        )
    )
    depths = edge_depths(topo4)
    for shell in sorted(set(depths.tolist())):
        depth_edges = {rid for rid, depth in enumerate(depths.tolist()) if int(depth) == int(shell)}
        changed = pick_edges(4, 1, list(depth_edges))
        label = "外层" if shell == 0 else f"深度{shell}"
        frames.append(
            make_network_frame(
                "子任务5实践：比较不同深度层",
                f"{label}电阻边",
                4,
                changed_edges=changed,
                depth_edges=depth_edges,
                footer="该任务需要同时关注深度效应与该深度层候选数量",
            )
        )

    frames.append(
        make_text_frame(
            "动画总结",
            "五个子任务分别改变规模、候选电阻数、活动端口数、激励次数和电阻深度",
            "每次只改变一个核心变量，再用位置识别准确率和 K_max 回答问题",
            "表达时先定义变量，再展示图像和结果，可以减少概念确认成本",
        )
    )
    return frames


def save_gif(frames: list[Image.Image], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    durations = [1600 if idx in {0, 1, 3, 9, 15, 19, len(frames) - 1} else 1050 for idx in range(len(frames))]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )


def main() -> None:
    setup_style()
    frames = detailed_frames()
    out = OUT_DIR / "详细动画_五个子任务实践流程.gif"
    save_gif(frames, out)
    print(f"generated={out}")
    print(f"frame_count={len(frames)}")


if __name__ == "__main__":
    main()
