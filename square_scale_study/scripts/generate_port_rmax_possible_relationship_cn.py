from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 10.5,
        }
    )


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Qualitative hypotheses only: keep discrete integer points so the sketch
    # does not look like fitted experimental data.
    ports = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=int)
    curves = {
        "快速增长型": np.array([3, 8, 15, 24, 36, 51, 69, 90], dtype=int),
        "近似线性型": np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=int),
        "饱和增长型": np.array([0, 3, 5, 7, 8, 9, 10, 11], dtype=int),
        "复杂度主导下降型": np.array([14, 9, 6, 4, 3, 2, 2, 1], dtype=int),
    }

    colors = {
        "快速增长型": "#d62728",
        "近似线性型": "#1f77b4",
        "饱和增长型": "#2ca02c",
        "复杂度主导下降型": "#4d4d4d",
    }

    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    for label, values in curves.items():
        ax.plot(
            ports,
            values,
            "-o",
            color=colors[label],
            linewidth=1.9,
            markersize=5.2,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=label,
        )

    ax.set_title("端口-最大电阻变化数量可能关系示意图", fontsize=15, weight="bold", pad=10)
    ax.set_xlabel("边界端口数量 P", fontsize=12.5)
    ax.set_ylabel("最大可识别变化电阻数量 Rmax", fontsize=12.5)
    ax.set_xlim(-1.0, 29.0)
    ax.set_ylim(0, 95)
    ax.set_xticks(ports)
    ax.set_yticks(np.arange(0, 100, 10))
    ax.legend(frameon=False, loc="upper left", handlelength=2.4)

    ax.text(
        0.99,
        0.03,
        "注：本图为待验证关系示意，不代表实验结果",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#555555",
    )

    out_path = OUT_DIR / "second_page_port_rmax_possible_relationship_cn.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out_path.resolve())


if __name__ == "__main__":
    main()
