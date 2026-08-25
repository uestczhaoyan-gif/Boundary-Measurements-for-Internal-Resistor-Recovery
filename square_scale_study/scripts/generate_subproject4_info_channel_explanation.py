from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | float | int | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def to_int(value: str | float | int | None) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def summarize_metrics(summary_rows: list[dict]) -> dict:
    by_e: dict[int, list[dict]] = defaultdict(list)
    by_threshold: dict[float, dict[int, int]] = defaultdict(dict)

    for row in summary_rows:
        by_e[to_int(row["E"])].append(row)

    e_stats: dict[int, dict] = {}
    for e, rows in sorted(by_e.items()):
        rows = sorted(rows, key=lambda item: to_int(item["K"]))
        avg_id_all = sum(to_float(row["test_id_exact_rate"]) for row in rows) / len(rows)
        avg_value_all = sum(to_float(row["test_value_accuracy"]) for row in rows) / len(rows)

        hard_rows = [row for row in rows if to_int(row["K"]) >= 3]
        avg_id_hard = sum(to_float(row["test_id_exact_rate"]) for row in hard_rows) / len(hard_rows)
        avg_value_hard = sum(to_float(row["test_value_accuracy"]) for row in hard_rows) / len(hard_rows)

        e_stats[e] = {
            "rows": rows,
            "avg_id_all": avg_id_all,
            "avg_value_all": avg_value_all,
            "avg_id_hard": avg_id_hard,
            "avg_value_hard": avg_value_hard,
        }

    return {"by_e": e_stats, "by_threshold": by_threshold}


def load_threshold_kmax(threshold_rows: list[dict]) -> dict[float, dict[int, int]]:
    grouped: dict[float, dict[int, int]] = defaultdict(dict)
    for row in threshold_rows:
        grouped[to_float(row["id_threshold"])][to_int(row["E"])] = to_int(row["K_max"])
    return grouped


def fmt3(x: float) -> str:
    return f"{x:.3f}"


def build_markdown(summary_rows: list[dict], threshold_rows: list[dict]) -> str:
    stats = summarize_metrics(summary_rows)["by_e"]
    threshold_map = load_threshold_kmax(threshold_rows)

    e_values = sorted(stats)
    if e_values != [1, 4, 12]:
        e_values = sorted(stats)

    lines: list[str] = []
    lines.append("# 子项目4：循环激励增加信息通道的物理与数学解释")
    lines.append("")
    lines.append("## 目的")
    lines.append("")
    lines.append("- 在固定 `4x4` 拓扑、固定全部 `12` 个边界测量端口、固定全边可变的条件下，只改变激励次数 `E`，判断循环激励提升的到底是“有效信息量”还是仅仅是“数值稳定性”。")
    lines.append("- 该说明文件不是再次训练模型，而是基于已经得到的子项目4结果，对现象做物理与数学上的统一解释。")
    lines.append("")
    lines.append("## 实践")
    lines.append("")
    lines.append("- 比较对象为 `E ∈ {1, 4, 12}`，变化数量为 `K ∈ {1,2,3,4,5,6}`。")
    lines.append("- 使用的正式结果文件为：")
    lines.append(f"  - `outputs_subproj4_excitation_modelg2/subproject4_excitation_test_summary.csv`")
    lines.append(f"  - `outputs_subproj4_excitation_modelg2/subproject4_excitation_threshold_kmax_summary.csv`")
    lines.append("- 观察量包括两层：")
    lines.append("  - 第一层：固定 `K` 比较 `id_exact_rate` 与 `value_accuracy` 随 `E` 的变化。")
    lines.append("  - 第二层：在 `95% / 90% / 85% / 80%` 四个 `ID` 阈值下比较 `K_max`。")
    lines.append("")
    lines.append("### 结果摘录")
    lines.append("")
    for e in e_values:
        row_text = "，".join(
            f"K={to_int(row['K'])}: ID={fmt3(to_float(row['test_id_exact_rate']))}, Value={fmt3(to_float(row['test_value_accuracy']))}"
            for row in stats[e]["rows"]
        )
        lines.append(f"- `E={e}`：{row_text}")
    lines.append("")
    lines.append("### 聚合观察")
    lines.append("")
    for e in e_values:
        lines.append(
            f"- `E={e}`：全 `K` 平均 `ID={fmt3(stats[e]['avg_id_all'])}`，困难区间 `K>=3` 平均 `ID={fmt3(stats[e]['avg_id_hard'])}`；"
            f"全 `K` 平均 `Value={fmt3(stats[e]['avg_value_all'])}`，困难区间 `K>=3` 平均 `Value={fmt3(stats[e]['avg_value_hard'])}`。"
        )
    lines.append("")
    lines.append("### 阈值侧结果")
    lines.append("")
    for threshold in sorted(threshold_map):
        row = threshold_map[threshold]
        parts = [f"`E={e}` -> `K_max={row.get(e, 0)}`" for e in sorted(row)]
        lines.append(f"- `ID阈值={int(round(threshold * 100))}%`：{'，'.join(parts)}。")
    lines.append("")
    lines.append("## 结果")
    lines.append("")
    lines.append("- `E=1 -> 4` 出现了实质性跃迁：在 `K>=2` 时，`ID` 精度明显上升，而 `Value` 精度本来就不低，说明循环激励带来的主收益不是简单把回归值再磨细，而是让不同变化模式更可区分。")
    lines.append("- `E=4 -> 12` 的提升没有前一步剧烈，但在 `K>=3` 的困难区间仍保持稳定增益，说明额外激励继续提供了新的辨识信息，只是边际收益开始下降。")
    lines.append("- 因而，子项目4支持的结论不是“多激励只提高稳健性”，而是“多激励确实增加了有效可辨识信息，只不过这种信息增益在低复杂度场景中更容易饱和”。")
    lines.append("")
    lines.append("## 物理解释")
    lines.append("")
    lines.append("- 每一组边界激励都会在网络内部形成一套不同的电流分布。某条电阻是否容易被识别，取决于它在这套电流流线下是否被充分“照亮”。")
    lines.append("- 单次激励相当于只从一个方向观察网络：很多内部边在该方向下产生的边界响应形状会非常相似，因此不同变化组合会在测量端表现出较强混淆。")
    lines.append("- 循环激励本质上是在改变电流穿过网络的路径方向。随着激励次数增加，原本在某一组激励下不敏感或高度相似的边，会在另一组激励下显现出差异。")
    lines.append("- 所以，多端口循环激励的物理意义不是“重复测同一件事”，而是“从多个方向照射同一网络”，从而减少遮蔽、对称混淆和局部不可见区域。")
    lines.append("")
    lines.append("## 数学解释")
    lines.append("")
    lines.append("- 在基准电阻附近做小扰动线性化，可以把每个激励下的边界响应写成：")
    lines.append("  - `Δv^(e) ≈ J_e · Δr`")
    lines.append("- 其中：")
    lines.append("  - `Δr` 是全部候选电阻的变化向量；")
    lines.append("  - `J_e` 是第 `e` 个激励下，边界电压对各电阻变化的灵敏度矩阵。")
    lines.append("- 把多个激励堆叠起来，就得到：")
    lines.append("  - `Δv_stack ≈ [J_1; J_2; ...; J_E] · Δr`")
    lines.append("- 这说明增加激励次数，本质上是在给同一个未知向量 `Δr` 增加新的观测方程。")
    lines.append("- 如果新增的 `J_e` 与已有激励对应的灵敏度方向不同，那么：")
    lines.append("  - 堆叠后的观测算子有效秩会上升；")
    lines.append("  - 零空间会缩小；")
    lines.append("  - 不同电阻变化模式在边界响应空间中的距离会拉开；")
    lines.append("  - support 排序会更容易稳定。")
    lines.append("- 如果新增激励与已有激励高度相关，那么它的主要作用就不是增加新信息，而是通过冗余观测提高数值稳定性。")
    lines.append("- 子项目4的实验结果表现为：`E=1 -> 4` 的提升非常显著，而 `E=4 -> 12` 仍有增益但变缓，这正符合“先明显增加有效秩，再逐渐进入边际收益递减”的典型规律。")
    lines.append("")
    lines.append("## 对 `P_active * E` 的正确理解")
    lines.append("")
    lines.append("- `P_active * E` 不是严格的信息通道数，只是一个很粗的“观测预算代理量”。")
    lines.append("- 它表示：如果每次激励都能读取 `P_active` 个端口，那么总观测标量数大致与 `P_active * E` 成正比。")
    lines.append("- 但这个量不能直接等同于真实信息维数，因为：")
    lines.append("  - 不同端口读数之间存在相关性；")
    lines.append("  - 不同激励之间也可能高度相关；")
    lines.append("  - 还有基准节点、基尔霍夫约束和拓扑对称性带来的冗余。")
    lines.append("- 因此，`P_active * E` 适合作为粗指标，但若要真正跨拓扑、跨规模判断难度，分母更应该接近“有效独立观测维数”，例如灵敏度矩阵的有效秩。")
    lines.append("")
    lines.append("## 当前可直接汇报的一句话")
    lines.append("")
    lines.append("- 子项目4说明，多端口循环激励不是简单重复测量，而是在改变电流穿过网络的方向；从数学上看，它通过堆叠更多彼此不完全相关的灵敏度方程，缩小了不可辨识空间，因此能够显著提高多电阻同时变化场景下的可辨识性。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a formal explanation note for subproject4 information gain.")
    parser.add_argument(
        "--summary-csv",
        default=str(PROJECT_ROOT / "outputs_subproj4_excitation_modelg2" / "subproject4_excitation_test_summary.csv"),
    )
    parser.add_argument(
        "--threshold-csv",
        default=str(PROJECT_ROOT / "outputs_subproj4_excitation_modelg2" / "subproject4_excitation_threshold_kmax_summary.csv"),
    )
    parser.add_argument(
        "--output-path",
        default=str(PROJECT_ROOT / "subproject4_info_channel_explanation.md"),
    )
    args = parser.parse_args()

    summary_rows = read_csv_rows(Path(args.summary_csv).resolve())
    threshold_rows = read_csv_rows(Path(args.threshold_csv).resolve())
    markdown = build_markdown(summary_rows, threshold_rows)

    output_path = Path(args.output_path).resolve()
    output_path.write_text(markdown, encoding="utf-8")
    print(f"wrote_explanation={output_path}")


if __name__ == "__main__":
    main()
