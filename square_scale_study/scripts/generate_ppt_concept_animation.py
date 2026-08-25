from __future__ import annotations

import io
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "PPT_Figure" / "animation_concepts"
FRAME_SIZE = (1600, 900)
FRAME_MARGIN = 70


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
            "font.size": 13,
        }
    )


def grid_topology(n: int) -> tuple[np.ndarray, list[tuple[int, int]], list[int]]:
    coords: list[tuple[float, float]] = []
    denom = max(n - 1, 1)
    for row in range(n):
        for col in range(n):
            coords.append((col / denom, (n - 1 - row) / denom))

    edges: list[tuple[int, int]] = []
    for row in range(n):
        for col in range(n - 1):
            edges.append((row * n + col, row * n + col + 1))
        if row < n - 1:
            for col in range(n):
                edges.append((row * n + col, (row + 1) * n + col))

    top = list(range(n))
    right = [row * n + (n - 1) for row in range(1, n)]
    bottom = list(range(n * n - 2, n * (n - 1) - 1, -1))
    left = [row * n for row in range(n - 2, 0, -1)]
    return np.asarray(coords, dtype=float), edges, top + right + bottom + left


def edge_depth(n: int, edge: tuple[int, int]) -> int:
    depths = []
    for node in edge:
        row, col = divmod(node, n)
        depths.append(min(row, col, n - 1 - row, n - 1 - col))
    return min(depths)


def candidate_pool(n: int, count: int) -> set[int]:
    _coords, edges, _boundary = grid_topology(n)
    ordered = sorted(range(len(edges)), key=lambda rid: (edge_depth(n, edges[rid]), rid))
    return set(ordered[: min(count, len(edges))])


def changed_from_pool(pool: set[int], count: int) -> set[int]:
    ordered = sorted(pool, key=lambda rid: (rid * 17 + 11) % 97)
    return set(ordered[: min(count, len(ordered))])


def draw_network(
    ax: plt.Axes,
    n: int,
    candidate_edges: set[int] | None = None,
    changed_edges: set[int] | None = None,
    active_ports: set[int] | None = None,
) -> None:
    candidate_edges = candidate_edges or set()
    changed_edges = changed_edges or set()
    coords, edges, boundary = grid_topology(n)
    boundary_set = set(boundary)
    active_ports = boundary_set if active_ports is None else active_ports

    segments = [[coords[u], coords[v]] for u, v in edges]
    colors = []
    widths = []
    for rid in range(len(edges)):
        if rid in changed_edges:
            colors.append("#d62728")
            widths.append(4.8)
        elif rid in candidate_edges:
            colors.append("#1f77b4")
            widths.append(3.0)
        else:
            colors.append("#bdbdbd")
            widths.append(1.45)
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=widths, capstyle="round", zorder=1))

    for node, (x, y) in enumerate(coords):
        if node in boundary_set:
            face = "#ffbf00" if node in active_ports else "#eeeeee"
            edge = "#333333" if node in active_ports else "#999999"
            radius = 0.037
        else:
            face = "white"
            edge = "#555555"
            radius = 0.026
        ax.add_patch(Circle((x, y), radius, facecolor=face, edgecolor=edge, linewidth=1.0, zorder=4))

    ax.set_aspect("equal")
    ax.set_xlim(-0.14, 1.14)
    ax.set_ylim(-0.14, 1.14)
    ax.axis("off")


def frame_to_image(fig: plt.Figure) -> Image.Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    image.thumbnail((FRAME_SIZE[0] - 2 * FRAME_MARGIN, FRAME_SIZE[1] - 2 * FRAME_MARGIN), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", FRAME_SIZE, "white")
    canvas.paste(image, ((FRAME_SIZE[0] - image.width) // 2, (FRAME_SIZE[1] - image.height) // 2))
    return canvas.convert("P", palette=Image.ADAPTIVE)


def node_scale_frame() -> Image.Image:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 5.0), constrained_layout=True)
    for ax, n in zip(axes, [3, 4, 5]):
        draw_network(ax, n)
        ax.set_title(f"N = {n}\n节点数 = {n * n}", fontsize=17, pad=8)
    fig.suptitle("节点规模 N", fontsize=26, weight="bold", y=1.05)
    fig.text(0.5, 0.02, "N 表示正方形网络的边长规模：N × N 个节点", ha="center", fontsize=16)
    return frame_to_image(fig)


def port_count_frame() -> Image.Image:
    n = 5
    fig, ax = plt.subplots(figsize=(8.6, 6.5))
    draw_network(ax, n)
    fig.suptitle("端口数量 P", fontsize=26, weight="bold", y=0.97)
    fig.text(0.5, 0.09, "黄色节点 = 边界端口", ha="center", fontsize=17)
    fig.text(0.5, 0.045, f"P = 4N - 4；当 N = {n} 时，P = {4 * n - 4}", ha="center", fontsize=16)
    return frame_to_image(fig)


def candidate_pool_frame(count: int) -> Image.Image:
    n = 4
    pool = candidate_pool(n, count)
    fig, ax = plt.subplots(figsize=(8.6, 6.5))
    draw_network(ax, n, candidate_edges=pool)
    fig.suptitle("候选可变电阻池 C", fontsize=26, weight="bold", y=0.97)
    fig.text(0.5, 0.09, "蓝色边 = 允许发生变化的候选电阻", ha="center", fontsize=17)
    fig.text(0.5, 0.045, f"C = {len(pool)}", ha="center", fontsize=18, color="#1f77b4", weight="bold")
    return frame_to_image(fig)


def actual_change_frame(r_count: int, pool_count: int = 16) -> Image.Image:
    n = 4
    pool = candidate_pool(n, pool_count)
    changed = changed_from_pool(pool, r_count)
    fig, ax = plt.subplots(figsize=(8.6, 6.5))
    draw_network(ax, n, candidate_edges=pool, changed_edges=changed)
    fig.suptitle("实际变化电阻数 R", fontsize=26, weight="bold", y=0.97)
    fig.text(0.5, 0.09, "红色边 = 单个样本中真正变化的电阻", ha="center", fontsize=17)
    fig.text(0.5, 0.045, f"R = {r_count}", ha="center", fontsize=18, color="#d62728", weight="bold")
    return frame_to_image(fig)


def rmax_frame() -> Image.Image:
    n = 4
    pool = candidate_pool(n, 16)
    changed = changed_from_pool(pool, 4)
    fig, ax = plt.subplots(figsize=(8.6, 6.5))
    draw_network(ax, n, candidate_edges=pool, changed_edges=changed)
    fig.suptitle("最大可识别变化电阻数 R_max", fontsize=25, weight="bold", y=0.97)
    fig.text(0.5, 0.09, "在给定精度要求下，能够稳定识别的最大 R", ha="center", fontsize=17)
    fig.text(0.5, 0.045, "示意：R_max = 4", ha="center", fontsize=18, color="#d62728", weight="bold")
    return frame_to_image(fig)


def make_frames() -> list[Image.Image]:
    return [
        node_scale_frame(),
        port_count_frame(),
        candidate_pool_frame(8),
        candidate_pool_frame(16),
        candidate_pool_frame(24),
        actual_change_frame(1),
        actual_change_frame(2),
        actual_change_frame(3),
        rmax_frame(),
    ]


def save_gif(frames: list[Image.Image], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    durations = [3200, 3200, 2800, 2800, 3200, 2800, 2800, 3200, 3600]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )


def save_contact_sheet(frames: list[Image.Image], path: Path) -> None:
    cols = 3
    thumb_w, thumb_h = 430, 242
    rows = math.ceil(len(frames) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 28)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, frame in enumerate(frames):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + 28)
        sheet.paste(frame.convert("RGB").resize((thumb_w, thumb_h)), (x, y + 24))
        draw.text((x + 8, y + 5), f"Frame {idx + 1}", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> None:
    setup_style()
    frames = make_frames()
    gif_path = OUT_DIR / "概念定义_关键变量示意.gif"
    preview_path = OUT_DIR / "概念定义_抽帧预览.png"
    save_gif(frames, gif_path)
    save_contact_sheet(frames, preview_path)
    print(f"generated={gif_path}")
    print(f"preview={preview_path}")
    print(f"frame_count={len(frames)}")


if __name__ == "__main__":
    main()
