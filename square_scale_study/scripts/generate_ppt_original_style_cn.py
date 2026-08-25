from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import prepend_vendor_dir

prepend_vendor_dir(PROJECT_ROOT / ".vendor_plot", required_version=(3, 11))

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = PROJECT_ROOT / "PPT_Figure" / "original_style_cn"
THRESHOLDS_6 = [0.98, 0.95, 0.90, 0.85, 0.80, 0.75]
THRESHOLDS_4 = [0.95, 0.90, 0.85, 0.80]


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
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "legend.fontsize": 9.5,
        }
    )


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def val(row: dict, key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw in ("", None):
        return default
    return float(raw)


def intval(row: dict, key: str, default: int = 0) -> int:
    raw = row.get(key, "")
    if raw in ("", None):
        return default
    return int(float(raw))


def savefig(fig: plt.Figure, relative: str) -> Path:
    path = OUT_DIR / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def grouped(rows: list[dict], key: str) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for row in rows:
        out.setdefault(intval(row, key), []).append(row)
    return out


def threshold_from_summary(rows: list[dict], group_key: str, thresholds: list[float]) -> list[dict]:
    groups = grouped(rows, group_key)
    output: list[dict] = []
    for threshold in thresholds:
        for group_value, items in sorted(groups.items()):
            passed = [
                item
                for item in items
                if val(item, "test_id_exact_rate") >= threshold and val(item, "test_value_accuracy") >= 0.90
            ]
            best = max(passed, key=lambda item: intval(item, "K")) if passed else None
            output.append(
                {
                    "threshold": threshold,
                    group_key: group_value,
                    "K_max": intval(best, "K") if best else 0,
                }
            )
    return output


def plot_metric_dropoff_grid(rows: list[dict], group_key: str, group_label: str, out_name: str, ncols: int = 3) -> Path:
    groups = grouped(rows, group_key)
    items = sorted(groups.items())
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.15 * ncols, 3.45 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    legend_handles = None
    for ax, (group_value, items_group) in zip(axes, items):
        items_group = sorted(items_group, key=lambda item: intval(item, "K"))
        ks = [intval(row, "K") for row in items_group]
        x = np.arange(len(ks))
        width = 0.36
        id_vals = [val(row, "test_id_exact_rate") for row in items_group]
        value_vals = [val(row, "test_value_accuracy") for row in items_group]
        bars_id = ax.bar(x - width / 2, id_vals, width=width, color="#1f77b4", label="位置识别")
        bars_value = ax.bar(x + width / 2, value_vals, width=width, color="#ff7f0e", label="数值回归")
        ax.axhline(0.98, color="#1f77b4", linestyle=":", linewidth=1.0)
        ax.axhline(0.90, color="#ff7f0e", linestyle=":", linewidth=1.0)
        ax.set_xticks(x, [str(k) for k in ks])
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("变化电阻数量 K")
        ax.set_ylabel("测试集指标")
        ax.set_title(f"{group_label}={group_value}")
        if legend_handles is None:
            legend_handles = (bars_id[0], bars_value[0])
    for ax in axes[len(items):]:
        ax.axis("off")
    if legend_handles is not None and items:
        axes[0].legend(legend_handles, ["位置识别", "数值回归"], frameon=False, loc="lower left")
    return savefig(fig, out_name)


def plot_threshold_facets(threshold_rows: list[dict], group_key: str, xlabel: str, out_name: str, thresholds: list[float]) -> Path:
    by_t: dict[float, list[dict]] = {}
    for row in threshold_rows:
        by_t.setdefault(float(row["threshold"]), []).append(row)
    ncols = 3 if len(thresholds) == 6 else 2
    nrows = int(np.ceil(len(thresholds) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.15 * ncols, 3.35 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, threshold in zip(axes, thresholds):
        group = sorted(by_t.get(float(threshold), []), key=lambda item: int(item[group_key]))
        x = [int(item[group_key]) for item in group]
        y = [int(item["K_max"]) for item in group]
        ax.plot(x, y, marker="o", linewidth=2.0, color="#1f77b4")
        ax.set_xticks(x)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("最大可识别变化电阻数量 K_max")
        ax.set_title(f"位置识别阈值 {int(round(threshold * 100))}%")
    for ax in axes[len(thresholds):]:
        ax.axis("off")
    return savefig(fig, out_name)


def plot_accuracy_by_k(rows: list[dict], group_key: str, xlabel: str, out_name: str, y_min: float = 0.0) -> Path:
    groups = grouped(rows, "K")
    ks = sorted(groups)
    ncols = 3 if len(ks) > 4 else 2
    nrows = int(np.ceil(len(ks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, k in zip(axes, ks):
        items = sorted(groups[k], key=lambda item: intval(item, group_key))
        x = [intval(row, group_key) for row in items]
        id_vals = [val(row, "test_id_exact_rate") for row in items]
        value_vals = [val(row, "test_value_accuracy") for row in items]
        ax.plot(x, id_vals, marker="o", linewidth=2.0, color="#1f77b4", label="位置识别")
        ax.plot(x, value_vals, marker="s", linewidth=2.0, color="#ff7f0e", label="数值回归")
        ax.set_xticks(x)
        ax.set_ylim(y_min, 1.01)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("测试集准确率")
        ax.set_title(f"K={k}")
    for ax in axes[len(ks):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, frameon=False, loc="lower left")
    return savefig(fig, out_name)


def render_table(rows: list[dict], columns: list[tuple[str, str]], title: str, out_name: str, width: float = 12.5) -> Path:
    table_data = [[str(row.get(key, "")) for key, _label in columns] for row in rows]
    labels = [label for _key, label in columns]
    fig_h = max(2.5, 0.42 * len(table_data) + 1.3)
    fig, ax = plt.subplots(figsize=(width, fig_h), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=table_data, colLabels=labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.25)
    ax.set_title(title, fontsize=12, pad=10)
    return savefig(fig, out_name)


def metric_table_from_rows(rows: list[dict], group_key: str, group_label: str) -> list[dict]:
    by_group: dict[int, dict[int, dict]] = {}
    ks = sorted({intval(row, "K") for row in rows})
    for row in rows:
        by_group.setdefault(intval(row, group_key), {})[intval(row, "K")] = row
    table_rows: list[dict] = []
    for group_value in sorted(by_group):
        out: dict[str, str] = {group_label: str(group_value)}
        for k in ks:
            item = by_group[group_value].get(k)
            out[f"K{k} 位置"] = "" if item is None else f"{val(item, 'test_id_exact_rate'):.3f}"
            out[f"K{k} 数值"] = "" if item is None else f"{val(item, 'test_value_accuracy'):.3f}"
        table_rows.append(out)
    return table_rows


def write_table_png(rows: list[dict], group_key: str, group_label: str, title: str, out_name: str) -> Path:
    table_rows = metric_table_from_rows(rows, group_key, group_label)
    columns: list[tuple[str, str]] = []
    if table_rows:
        columns = [(key, key) for key in table_rows[0].keys()]
    return render_table(table_rows, columns, title, out_name)


def generate_subtask1() -> list[Path]:
    base = read_csv(PROJECT_ROOT / "outputs_modelg2_subproj1" / "modelg2_subproj1_test_summary.csv")
    extra = read_csv(PROJECT_ROOT / "outputs_modelg2_task1_n78" / "modelg2_task1_n78_test_summary.csv")
    rows = sorted(base + extra, key=lambda row: (intval(row, "N"), intval(row, "K")))
    paths = [
        plot_metric_dropoff_grid(base, "N", "网络规模 N", "subtask1_scale_metric_dropoff_overview_cn.png", ncols=2),
        plot_metric_dropoff_grid(extra, "N", "网络规模 N", "subtask1_scale_N7_N8_metric_dropoff_overview_cn.png", ncols=2),
        plot_threshold_facets(threshold_from_summary(rows, "P", THRESHOLDS_6), "P", "边界端口数量 P", "subtask1_scale_threshold_kmax_compare_cn.png", THRESHOLDS_6),
    ]
    table_rows = [
        {
            "网络规模": f"{intval(row, 'N')}×{intval(row, 'N')}",
            "端口数": str(intval(row, "P")),
            "电阻总数": str(intval(row, "M")),
            "变化数量K": str(intval(row, "K")),
            "位置识别": f"{val(row, 'test_id_exact_rate'):.3f}",
            "数值回归": f"{val(row, 'test_value_accuracy'):.3f}",
        }
        for row in rows
    ]
    paths.append(
        render_table(
            table_rows,
            [(key, key) for key in table_rows[0].keys()],
            "子任务1测试集指标表",
            "subtask1_scale_metric_table_cn.png",
            width=10.5,
        )
    )
    return paths


def generate_subtask2() -> list[Path]:
    rows = read_csv(PROJECT_ROOT / "outputs_subproj2_varcand_modelg2" / "subproject2_varcand_test_summary.csv")
    rows = sorted(rows, key=lambda row: (intval(row, "M_var"), intval(row, "K")))
    return [
        plot_metric_dropoff_grid(rows, "M_var", "候选可变电阻总数", "subtask2_varcand_metric_dropoff_overview_cn.png", ncols=3),
        plot_accuracy_by_k(rows, "M_var", "候选可变电阻总数 M_var", "subtask2_varcand_accuracy_by_k_overview_cn.png", y_min=0.70),
        plot_threshold_facets(threshold_from_summary(rows, "M_var", THRESHOLDS_6), "M_var", "候选可变电阻总数 M_var", "subtask2_varcand_threshold_kmax_compare_cn.png", THRESHOLDS_6),
        write_table_png(rows, "M_var", "候选数", "子任务2测试集指标表", "subtask2_varcand_metric_table_cn.png"),
    ]


def generate_subtask3() -> list[Path]:
    rows = read_csv(PROJECT_ROOT / "outputs_subproj3_activeport_modelg2" / "subproject3_activeport_test_summary.csv")
    rows = sorted(rows, key=lambda row: (intval(row, "P_active"), intval(row, "K")))
    return [
        plot_metric_dropoff_grid(rows, "P_active", "活动端口数", "subtask3_activeport_metric_dropoff_overview_cn.png", ncols=3),
        plot_accuracy_by_k(rows, "P_active", "活动端口数量 P_active", "subtask3_activeport_accuracy_by_k_overview_cn.png", y_min=0.0),
        plot_threshold_facets(threshold_from_summary(rows, "P_active", THRESHOLDS_6), "P_active", "活动端口数量 P_active", "subtask3_activeport_threshold_kmax_compare_cn.png", THRESHOLDS_6),
        write_table_png(rows, "P_active", "活动端口数", "子任务3测试集指标表", "subtask3_activeport_metric_table_cn.png"),
    ]


def generate_subtask4() -> list[Path]:
    rows = read_csv(PROJECT_ROOT / "outputs_subproj4_excitation_modelg2" / "subproject4_excitation_test_summary.csv")
    rows = sorted(rows, key=lambda row: (intval(row, "E"), intval(row, "K")))
    return [
        plot_metric_dropoff_grid(rows, "E", "激励次数", "subtask4_excitation_metric_dropoff_overview_cn.png", ncols=3),
        plot_accuracy_by_k(rows, "E", "激励次数 E", "subtask4_excitation_accuracy_by_k_overview_cn.png", y_min=0.0),
        plot_threshold_facets(threshold_from_summary(rows, "E", THRESHOLDS_4), "E", "激励次数 E", "subtask4_excitation_threshold_kmax_compare_cn.png", THRESHOLDS_4),
        write_table_png(rows, "E", "激励次数", "子任务4测试集指标表", "subtask4_excitation_metric_table_cn.png"),
    ]


def generate_subtask5() -> list[Path]:
    rows = read_csv(PROJECT_ROOT / "outputs_subproj5_depth_followup" / "subproject5_depth_aggregate_rows.csv")
    rows = [row for row in rows if row.get("source") in {"multiseed", "single_n6"}]
    rows = sorted(rows, key=lambda row: (intval(row, "N"), intval(row, "depth_shell")))
    ns = sorted({intval(row, "N") for row in rows})
    shells = sorted({intval(row, "depth_shell") for row in rows})
    x = np.arange(len(shells))
    width = 0.22
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), constrained_layout=True)
    for idx, n in enumerate(ns):
        group = {intval(row, "depth_shell"): row for row in rows if intval(row, "N") == n}
        offset = (idx - (len(ns) - 1) / 2) * width
        axes[0].bar(x + offset, [val(group[s], "id_mean") if s in group else np.nan for s in shells], width=width, label=f"{n}×{n}")
        axes[1].bar(x + offset, [val(group[s], "value_mean") if s in group else np.nan for s in shells], width=width, label=f"{n}×{n}")
    labels = ["外层" if s == 0 else f"深度{s}" for s in shells]
    axes[0].set_xticks(x, labels)
    axes[1].set_xticks(x, labels)
    axes[0].set_xlabel("电阻边距离边界的深度层")
    axes[1].set_xlabel("电阻边距离边界的深度层")
    axes[0].set_ylabel("位置识别准确率")
    axes[1].set_ylabel("数值回归精度")
    axes[0].set_ylim(0.85, 1.01)
    axes[1].set_ylim(0.98, 1.01)
    axes[0].set_title("位置识别随深度变化")
    axes[1].set_title("数值回归随深度变化")
    axes[0].legend(frameon=False)
    p1 = savefig(fig, "subtask5_depth_metric_summary_cn.png")

    table_rows = [
        {
            "来源": row.get("source", ""),
            "规模": f"{intval(row, 'N')}×{intval(row, 'N')}",
            "深度层": "外层" if intval(row, "depth_shell") == 0 else f"深度{intval(row, 'depth_shell')}",
            "样本数": f"{val(row, 'sample_count_mean'):.0f}",
            "位置识别": f"{val(row, 'id_mean'):.3f}",
            "数值回归": f"{val(row, 'value_mean'):.3f}",
        }
        for row in rows
    ]
    p2 = render_table(table_rows, [(key, key) for key in table_rows[0].keys()], "子任务5深度实验指标表", "subtask5_depth_metric_table_cn.png", width=9.5)
    return [p1, p2]


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths.extend(generate_subtask1())
    paths.extend(generate_subtask2())
    paths.extend(generate_subtask3())
    paths.extend(generate_subtask4())
    paths.extend(generate_subtask5())
    print("generated:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
