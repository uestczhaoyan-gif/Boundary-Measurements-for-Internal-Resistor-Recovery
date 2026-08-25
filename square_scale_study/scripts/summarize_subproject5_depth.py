from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_PLOT = PROJECT_ROOT / ".vendor_plot"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import prepend_vendor_dir
from project_common import BASE_R, DEFAULT_CHANGE_LIMIT, build_square_topology, edge_depths, load_json

prepend_vendor_dir(VENDOR_PLOT, required_version=(3, 11))

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def default_depth_label(depth_shell: int) -> str:
    return "outer" if depth_shell == 0 else f"inner_d{depth_shell}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize subproject-5 depth-effect results.")
    parser.add_argument("--outputs-root", required=True)
    parser.add_argument(
        "--data-roots",
        nargs="+",
        default=[str(PROJECT_ROOT / "data"), str(PROJECT_ROOT / "data_subproj5_depth")],
        help="One or more roots to search for <dataset_stem>_meta.json.",
    )
    parser.add_argument("--grid-list", default="4,5")
    parser.add_argument("--figure-dir", default=str(PROJECT_ROOT / "Figure" / "subproject5_depth"))
    return parser.parse_args()


def _apply_style() -> None:
    if plt is None:
        return
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "legend.fontsize": 9.5,
        }
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_meta_path(dataset_stem: str, recorded_meta_path: str | None, data_roots: list[Path]) -> Path:
    if recorded_meta_path:
        candidate = Path(recorded_meta_path)
        if candidate.exists():
            return candidate
    target_name = f"{dataset_stem}_meta.json"
    for root in data_roots:
        if not root.exists():
            continue
        matches = sorted(root.rglob(target_name))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not resolve meta file for dataset_stem={dataset_stem}")


def load_predictions_map(predictions_path: Path) -> dict[int, dict[str, float]]:
    output: dict[int, dict[str, float]] = {}
    with predictions_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = int(row["sample_id"])
            output[sample_id] = {
                "support_exact": float(row["support_exact"]),
                "mae_changed_sample": float(row["mae_changed_sample"]),
                "support_overlap": float(row["support_overlap"]),
            }
    return output


def load_test_shell_assignments(meta_path: Path) -> tuple[dict[int, int], dict[int, str], dict]:
    meta = load_json(meta_path)
    topology = build_square_topology(int(meta["topology"]["grid_size"]))
    edge_depth = edge_depths(topology)
    labels_from_meta = {int(shell): str(label) for shell, label in (meta.get("depth_labels") or {}).items()}
    csv_path = meta_path.parent / meta["files"]["test"]
    shell_by_sample: dict[int, int] = {}
    label_by_sample: dict[int, str] = {}

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        current_sample = None
        for row in reader:
            sample_id = int(row["sample_id"])
            if sample_id == current_sample:
                continue
            current_sample = sample_id
            if "depth_shell" in row and str(row["depth_shell"]).strip():
                shell = int(row["depth_shell"])
                label = str(row.get("depth_label") or labels_from_meta.get(shell) or default_depth_label(shell))
            else:
                rid = int(row["r1_id"])
                shell = int(edge_depth[rid])
                label = labels_from_meta.get(shell) or default_depth_label(shell)
            shell_by_sample[sample_id] = shell
            label_by_sample[sample_id] = label
    return shell_by_sample, label_by_sample, meta


def shell_edge_count_from_meta(meta: dict, shell: int) -> int:
    if meta.get("edge_ids_by_depth"):
        return len((meta.get("edge_ids_by_depth") or {}).get(str(shell), []))
    topology = build_square_topology(int(meta["topology"]["grid_size"]))
    return int(sum(1 for depth in edge_depths(topology).tolist() if int(depth) == int(shell)))


def collect_rows(outputs_root: Path, data_roots: list[Path], grid_filter: set[int]) -> list[dict]:
    rows: list[dict] = []
    for infer_path in sorted(outputs_root.rglob("inference_metrics.json")):
        pred_path = infer_path.parent / "predictions.csv"
        if not pred_path.exists():
            continue
        infer_payload = json.loads(infer_path.read_text(encoding="utf-8"))
        k = int(infer_payload.get("k") or 0)
        grid_size = int(infer_payload.get("grid_size") or 0)
        if k != 1 or grid_size not in grid_filter:
            continue
        study_protocol = str(infer_payload.get("study_protocol") or "")
        if study_protocol not in {"full_scale", "depth_balanced", "depth_edge_balanced"}:
            continue

        dataset_stem = str(infer_payload["dataset_stem"])
        meta_path = resolve_meta_path(dataset_stem, infer_payload.get("meta_path"), data_roots)
        shell_by_sample, label_by_sample, meta = load_test_shell_assignments(meta_path)
        predictions_map = load_predictions_map(pred_path)
        shell_groups: dict[int, list[dict[str, float]]] = {}
        for sample_id, pred in predictions_map.items():
            if sample_id not in shell_by_sample:
                continue
            shell_groups.setdefault(shell_by_sample[sample_id], []).append(pred)

        for shell, group in sorted(shell_groups.items()):
            sample_count = len(group)
            id_exact_rate = sum(item["support_exact"] for item in group) / sample_count if sample_count else 0.0
            mae_changed = sum(item["mae_changed_sample"] for item in group) / sample_count if sample_count else 0.0
            value_accuracy = max(0.0, 1.0 - mae_changed / (DEFAULT_CHANGE_LIMIT * BASE_R))
            depth_label = next(
                (label for sample_id, label in label_by_sample.items() if shell_by_sample.get(sample_id) == shell),
                default_depth_label(shell),
            )
            rows.append(
                {
                    "run_dir": infer_path.parent.name,
                    "dataset_stem": dataset_stem,
                    "study_protocol": study_protocol,
                    "N": grid_size,
                    "P": int(infer_payload.get("port_count") or meta["topology"]["port_count"]),
                    "M": int(infer_payload.get("num_resistors") or meta["topology"]["num_resistors"]),
                    "depth_shell": shell,
                    "depth_label": depth_label,
                    "shell_edge_count": shell_edge_count_from_meta(meta, shell),
                    "sample_count": sample_count,
                    "id_exact_rate": id_exact_rate,
                    "value_accuracy": value_accuracy,
                    "mae_changed": mae_changed,
                }
            )
    rows.sort(key=lambda item: (item["N"], item["depth_shell"], item["run_dir"]))
    return rows


def plot_outer_inner_bars(rows: list[dict], figure_dir: Path) -> None:
    if plt is None or not rows:
        return
    _apply_style()
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["N"]), []).append(row)

    ns = sorted(grouped)
    fig, axes = plt.subplots(1, len(ns), figsize=(5.8 * max(len(ns), 1), 4.6), constrained_layout=True)
    if len(ns) == 1:
        axes = [axes]

    legend_handles = None
    for ax, n in zip(axes, ns):
        group = sorted(grouped[n], key=lambda item: item["depth_shell"])
        labels = [str(row["depth_label"]) for row in group]
        x = np.arange(len(group))
        width = 0.36
        id_vals = [float(row["id_exact_rate"]) for row in group]
        value_vals = [float(row["value_accuracy"]) for row in group]
        bars_id = ax.bar(x - width / 2, id_vals, width=width, color="#1f77b4", label="ID exact")
        bars_value = ax.bar(x + width / 2, value_vals, width=width, color="#ff7f0e", label="Value accuracy")
        if legend_handles is None:
            legend_handles = (bars_id[0], bars_value[0])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.03)
        ax.set_xlabel("Depth shell")
        ax.set_ylabel("Metric")
        ax.set_title(f"N = {n}")

    if legend_handles is not None:
        fig.legend(legend_handles, ["ID exact", "Value accuracy"], frameon=False, loc="upper center", ncol=2)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "subproject5_depth_outer_inner_compare.png", dpi=220)
    plt.close(fig)


def plot_depth_profile(rows: list[dict], figure_dir: Path) -> None:
    if plt is None or not rows:
        return
    _apply_style()
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["N"]), []).append(row)

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)

    for idx, n in enumerate(sorted(grouped)):
        group = sorted(grouped[n], key=lambda item: item["depth_shell"])
        x = [int(row["depth_shell"]) for row in group]
        id_vals = [float(row["id_exact_rate"]) for row in group]
        value_vals = [float(row["value_accuracy"]) for row in group]
        color = colors[idx % len(colors)]
        axes[0].plot(x, id_vals, marker="o", linewidth=2.0, color=color, label=f"N={n}")
        axes[1].plot(x, value_vals, marker="s", linewidth=2.0, color=color, label=f"N={n}")

    axes[0].set_xlabel("Depth shell")
    axes[0].set_ylabel("ID exact rate")
    axes[0].set_ylim(0.0, 1.03)
    axes[0].set_title("ID by depth shell")
    axes[1].set_xlabel("Depth shell")
    axes[1].set_ylabel("Value accuracy")
    axes[1].set_ylim(0.0, 1.03)
    axes[1].set_title("Value by depth shell")
    axes[0].legend(frameon=False, loc="lower left")
    axes[1].legend(frameon=False, loc="lower left")

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "subproject5_depth_profile_compare.png", dpi=220)
    plt.close(fig)


def render_metric_table(rows: list[dict], output_path: Path) -> None:
    if plt is None or not rows:
        return
    _apply_style()
    sorted_rows = sorted(rows, key=lambda item: (item["N"], item["depth_shell"]))
    table_rows = [
        [
            f"{int(row['N'])}x{int(row['N'])}",
            str(int(row["P"])),
            str(int(row["M"])),
            str(row["depth_label"]),
            str(int(row["sample_count"])),
            f"{float(row['id_exact_rate']):.3f}",
            f"{float(row['value_accuracy']):.3f}",
        ]
        for row in sorted_rows
    ]
    headers = ["Topology", "P", "M", "Depth", "Samples", "ID exact", "Value"]

    fig_h = max(2.6, 0.55 * len(table_rows) + 1.3)
    fig, ax = plt.subplots(figsize=(9.0, fig_h), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root).resolve()
    data_roots = [Path(item).resolve() for item in args.data_roots]
    grid_filter = {int(item) for item in args.grid_list.split(",") if item.strip()}
    figure_dir = Path(args.figure_dir).resolve()

    rows = collect_rows(outputs_root, data_roots, grid_filter)
    if not rows:
        raise RuntimeError("No matching subproject-5 rows found. Check outputs root, grid list, and data roots.")

    summary_csv = outputs_root / "subproject5_depth_shell_summary.csv"
    write_csv(
        summary_csv,
        rows,
        [
            "run_dir",
            "dataset_stem",
            "study_protocol",
            "N",
            "P",
            "M",
            "depth_shell",
            "depth_label",
            "shell_edge_count",
            "sample_count",
            "id_exact_rate",
            "value_accuracy",
            "mae_changed",
        ],
    )
    plot_outer_inner_bars(rows, figure_dir)
    plot_depth_profile(rows, figure_dir)
    render_metric_table(rows, figure_dir / "subproject5_depth_metric_table.png")
    print(f"wrote_summary={summary_csv}")
    print(f"wrote_figure={figure_dir / 'subproject5_depth_outer_inner_compare.png'}")
    print(f"wrote_figure={figure_dir / 'subproject5_depth_profile_compare.png'}")
    print(f"wrote_figure={figure_dir / 'subproject5_depth_metric_table.png'}")


if __name__ == "__main__":
    main()
