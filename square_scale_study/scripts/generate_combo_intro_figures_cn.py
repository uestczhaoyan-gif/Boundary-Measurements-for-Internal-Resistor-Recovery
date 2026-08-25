from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "PPT_Figure" / "ppt_revision_cn"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 11.5,
            "figure.dpi": 140,
        }
    )


def grid_edges(n: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for y in range(n):
        for x in range(n - 1):
            edges.append(((x, y), (x + 1, y)))
    for y in range(n - 1):
        for x in range(n):
            edges.append(((x, y), (x, y + 1)))
    return edges


def is_boundary_node(node: tuple[int, int], n: int) -> bool:
    x, y = node
    return x == 0 or x == n - 1 or y == 0 or y == n - 1


def draw_grid(
    ax: plt.Axes,
    *,
    n: int,
    highlight_edges: list[tuple[tuple[int, int], tuple[int, int]]],
    highlight_colors: list[str],
    title: str,
    subtitle: str,
) -> None:
    for a, b in grid_edges(n):
        ax.plot([a[0], b[0]], [n - 1 - a[1], n - 1 - b[1]], color="#c8c8c8", lw=2.1, zorder=1)

    for edge, color in zip(highlight_edges, highlight_colors):
        a, b = edge
        ax.plot(
            [a[0], b[0]],
            [n - 1 - a[1], n - 1 - b[1]],
            color=color,
            lw=6.0,
            solid_capstyle="round",
            zorder=4,
        )

    xs, ys, fc, sizes = [], [], [], []
    for y in range(n):
        for x in range(n):
            xs.append(x)
            ys.append(n - 1 - y)
            if is_boundary_node((x, y), n):
                fc.append("#f7b500")
                sizes.append(145)
            else:
                fc.append("#ffffff")
                sizes.append(88)

    ax.scatter(xs, ys, s=sizes, c=fc, edgecolors="#333333", linewidths=1.4, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(-0.35, n - 0.65)
    ax.set_ylim(-0.45, n - 0.55)
    ax.axis("off")
    ax.set_title(title, fontsize=15, weight="bold", pad=12)
    ax.text(
        0.5,
        -0.06,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.6,
        color="#444444",
    )


def make_hypothesis_figure() -> Path:
    n = 4
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.25), constrained_layout=False)
    fig.subplots_adjust(top=0.78, bottom=0.15, wspace=0.22)
    fig.suptitle("两电阻组合可重构猜想", fontsize=17, weight="bold", y=0.96)

    easy_edges = [((0, 0), (0, 1)), ((2, 3), (3, 3))]
    hard_edges = [((1, 1), (2, 1)), ((1, 2), (2, 2))]

    draw_grid(
        axes[0],
        n=n,
        highlight_edges=easy_edges,
        highlight_colors=["#d62728", "#1f77b4"],
        title="猜想：更容易重构",
        subtitle="两条电阻位置/响应差异较大，边界电压变化更容易区分",
    )
    draw_grid(
        axes[1],
        n=n,
        highlight_edges=hard_edges,
        highlight_colors=["#d62728", "#1f77b4"],
        title="猜想：更容易混淆",
        subtitle="两条电阻响应模式相似，组合变化可能互相替代或抵消",
    )

    out_path = OUT_DIR / "combo_identifiability_hypothesis_cn.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_hypothesis_r3_figure() -> Path:
    n = 4
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.25), constrained_layout=False)
    fig.subplots_adjust(top=0.76, bottom=0.15, wspace=0.22)
    fig.suptitle("三电阻组合可识别性猜想", fontsize=17, weight="bold", y=0.96)
    fig.text(
        0.5,
        0.86,
        "固定规模 N、候选池 C 和实际变化数量 R=3，只改变“哪三条电阻同时变化”",
        ha="center",
        va="center",
        fontsize=11.2,
        color="#555555",
    )

    easy_edges = [
        ((0, 0), (0, 1)),
        ((2, 3), (3, 3)),
        ((3, 1), (3, 2)),
    ]
    hard_edges = [
        ((1, 1), (2, 1)),
        ((1, 2), (2, 2)),
        ((1, 1), (1, 2)),
    ]
    edge_colors = ["#d62728", "#1f77b4", "#2ca02c"]

    draw_grid(
        axes[0],
        n=n,
        highlight_edges=easy_edges,
        highlight_colors=edge_colors,
        title="猜想：更容易分辨",
        subtitle="三条电阻分布更分散，响应差异更明显",
    )
    draw_grid(
        axes[1],
        n=n,
        highlight_edges=hard_edges,
        highlight_colors=edge_colors,
        title="猜想：更容易混淆",
        subtitle="三条电阻集中且响应相似，组合变化更难拆分",
    )

    out_path = OUT_DIR / "combo_identifiability_hypothesis_R3_cn.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def add_box(ax: plt.Axes, xy: tuple[float, float], text: str, *, fc: str = "#f7f9fc") -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        1.62,
        0.82,
        boxstyle="round,pad=0.035,rounding_size=0.045",
        linewidth=1.35,
        edgecolor="#4d6f91",
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + 0.81, y + 0.41, text, ha="center", va="center", fontsize=10.6, linespacing=1.22)


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.35,
            color="#4a4a4a",
            shrinkA=2,
            shrinkB=2,
        )
    )


def make_method_figure() -> Path:
    fig, ax = plt.subplots(figsize=(10.6, 4.25), constrained_layout=True)
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 3.1)
    ax.axis("off")
    ax.set_title("组合可识别性研究方法", fontsize=17, weight="bold", pad=8)

    positions = [(0.35, 1.65), (2.25, 1.65), (4.15, 1.65), (6.05, 1.65), (7.95, 1.65)]
    texts = [
        "固定条件\nN、C、R=2",
        "枚举组合\n所有两电阻对",
        "平衡样本\n每个组合样本数一致",
        "同一模型评估\n位置识别+数值精度",
        "难易分组\n易识别/困难组合",
    ]
    colors = ["#eef5ff", "#f7f9fc", "#f7f9fc", "#f7f9fc", "#fff4e6"]
    for pos, text, color in zip(positions, texts, colors):
        add_box(ax, pos, text, fc=color)

    for i in range(len(positions) - 1):
        sx = positions[i][0] + 1.62
        sy = positions[i][1] + 0.41
        ex = positions[i + 1][0]
        ey = positions[i + 1][1] + 0.41
        add_arrow(ax, (sx, sy), (ex, ey))

    add_box(ax, (4.15, 0.35), "物理解释\n灵敏度强度+响应相似度", fc="#edf7ed")
    add_arrow(ax, (4.96, 1.65), (4.96, 1.20))
    add_arrow(ax, (5.77, 0.76), (8.00, 1.55))

    ax.text(
        0.38,
        0.42,
        "输出目标：找出哪些电阻组合适合编码光学信息，哪些组合应尽量避免",
        fontsize=11.2,
        color="#333333",
        ha="left",
        va="center",
    )

    out_path = OUT_DIR / "combo_identifiability_method_cn.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [make_hypothesis_figure(), make_hypothesis_r3_figure(), make_method_figure()]
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
