from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "PPT_Figure" / "original_style_cn"


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
            "font.size": 11.5,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 10.5,
        }
    )


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discrete integer points: this is a qualitative expectation sketch, not a
    # fitted function. Keep both x and y coordinates integer-valued.
    ports = np.arange(0, 29, 4, dtype=int)

    curves = {
        "指数增长": np.array([2, 7, 13, 20, 29, 40, 53, 68], dtype=int),
        "正相关直线（k≈1）": ports.copy(),
        "对数增长": np.array([0, 3, 5, 7, 8, 9, 10, 11], dtype=int),
        "反比下降": np.array([12, 6, 4, 3, 2, 2, 1, 1], dtype=int),
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    linestyles = ["-", "-", "-", "--"]
    for (label, values), color, linestyle in zip(curves.items(), colors, linestyles):
        ax.plot(
            ports,
            values,
            linestyle=linestyle,
            marker="o",
            linewidth=1.8,
            markersize=5.0,
            markerfacecolor="white",
            markeredgewidth=1.2,
            color=color,
            label=label,
        )

    ax.set_title("端口-最大电阻变化数量预期图", fontsize=15, weight="bold", pad=10)
    ax.set_xlabel("边界端口数量 P", fontsize=12.5)
    ax.set_ylabel("最大可识别变化电阻数 R_max（预期）", fontsize=12.5)
    ax.set_xlim(-0.8, 28.8)
    ax.set_ylim(0, 72)
    ax.set_xticks(ports.astype(int))
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])
    ax.legend(frameon=False, loc="upper left", handlelength=2.4)

    for old_name in [
        "second_page_port_rmax_expectation_curves_cn.png",
        "second_page_port_rmax_expectation_curves_v2_cn.png",
    ]:
        old_path = OUT_DIR / old_name
        if old_path.exists():
            old_path.unlink()

    out_path = OUT_DIR / "second_page_port_rmax_expectation_points_cn.png"
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(out_path.resolve())


if __name__ == "__main__":
    main()
