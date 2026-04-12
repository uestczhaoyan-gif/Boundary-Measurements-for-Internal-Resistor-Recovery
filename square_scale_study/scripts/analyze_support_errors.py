from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
VENDOR_DIR = WORKSPACE_ROOT / ".vendor_torchpy311"
VENDOR_PLOT = PROJECT_ROOT / ".vendor_plot"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import prepend_vendor_dir

prepend_vendor_dir(VENDOR_DIR, required_version=(3, 11))
prepend_vendor_dir(VENDOR_PLOT, required_version=(3, 11))

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from models.modelv1 import ModelV1Regressor
from models.modelv1_1 import ModelV11Regressor
from models.modelv2 import ModelV2Regressor
from models.modelv3 import ModelV3Regressor
from models.modelo1_gnn import Modelo1GNNRegressor
from models.modelo1_mlp1 import Modelo1MLP1Regressor
from models.modelo1_mlp2 import Modelo1MLP2Regressor
from project_common import apply_standardization, compute_fixedk_metrics, load_json, load_split_from_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze support-recovery errors for a trained run.")
    parser.add_argument("--meta-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--model-type",
        choices=["modelv1", "modelv1_1", "modelv2", "modelv3", "modelo1_gnn", "modelo1_mlp1", "modelo1_mlp2"],
        required=True,
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def load_model(model_type: str, topology, input_dim: int, device: torch.device, train_metrics: dict):
    cfg = train_metrics.get("model_config", {})
    kwargs = {
        "topology": topology,
        "hidden_dim": int(cfg.get("hidden_dim", 128)),
        "edge_hidden": int(cfg.get("edge_hidden", 128)),
        "heads": int(cfg.get("gat_heads", 4)),
        "excitation_chunk_size": int(cfg.get("excitation_chunk_size", 4)),
        "dropout": float(cfg.get("dropout", 0.1)),
        "max_abs": float(cfg.get("max_abs", 250.0)),
    }
    if model_type == "modelv1":
        return ModelV1Regressor(**kwargs).to(device)
    if model_type == "modelv1_1":
        return ModelV11Regressor(**kwargs).to(device)
    if model_type == "modelv2":
        return ModelV2Regressor(**kwargs).to(device)
    if model_type == "modelv3":
        kwargs.update(
            {
                "num_layers": int(cfg.get("num_layers", 1)),
            }
        )
        return ModelV3Regressor(**kwargs).to(device)
    if model_type == "modelo1_gnn":
        kwargs.update(
            {
                "edge_hidden": int(cfg.get("edge_hidden", 512)),
                "heads": int(cfg.get("gat_heads", 8)),
                "num_layers": int(cfg.get("num_layers", 4)),
            }
        )
        return Modelo1GNNRegressor(**kwargs).to(device)
    if model_type == "modelo1_mlp1":
        return Modelo1MLP1Regressor(
            input_dim=input_dim,
            num_resistors=topology.num_resistors,
            hidden_dim=int(cfg.get("hidden_dim", 1536)),
            num_blocks=int(cfg.get("num_blocks", 8)),
            ff_multiplier=float(cfg.get("ff_multiplier", 2.0)),
            dropout=float(cfg.get("dropout", 0.02)),
            max_abs=float(cfg.get("max_abs", 250.0)),
        ).to(device)
    return Modelo1MLP2Regressor(
        input_dim=input_dim,
        num_resistors=topology.num_resistors,
        hidden_dim=int(cfg.get("hidden_dim", 1536)),
        num_blocks=int(cfg.get("num_blocks", 8)),
        ff_multiplier=float(cfg.get("ff_multiplier", 2.0)),
        dropout=float(cfg.get("dropout", 0.02)),
        max_abs=float(cfg.get("max_abs", 250.0)),
    ).to(device)


def run_predictions(model, model_type: str, x_split: np.ndarray, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    ranking_rows = []
    value_rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_split), batch_size):
            xb = torch.from_numpy(x_split[start:start + batch_size]).float().to(device)
            if model_type in {"modelv2", "modelv3", "modelo1_gnn", "modelo1_mlp1"}:
                score_logits, value_pred = model(xb)
                ranking_rows.append(score_logits.cpu().numpy())
                value_rows.append(value_pred.cpu().numpy())
            else:
                value_pred = model(xb)
                values = value_pred.cpu().numpy()
                ranking_rows.append(values.copy())
                value_rows.append(values)
    ranking_scores = np.concatenate(ranking_rows, axis=0) if ranking_rows else np.zeros((0, 0), dtype=np.float32)
    value_pred = np.concatenate(value_rows, axis=0) if value_rows else np.zeros((0, 0), dtype=np.float32)
    return ranking_scores, value_pred


def resistor_neighbor_map(resistor_edges: list[tuple[int, int]]) -> dict[int, set[int]]:
    mapping: dict[int, set[int]] = {idx: set() for idx in range(len(resistor_edges))}
    for i, (u1, v1) in enumerate(resistor_edges):
        for j, (u2, v2) in enumerate(resistor_edges):
            if i != j and len({u1, v1} & {u2, v2}) > 0:
                mapping[i].add(j)
    return mapping


def score_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def analyze_details(
    sample_ids: np.ndarray,
    y_true: np.ndarray,
    ranking_scores: np.ndarray,
    k: int,
    resistor_edges: list[tuple[int, int]],
) -> tuple[list[dict], dict]:
    neighbor_map = resistor_neighbor_map(resistor_edges)
    rows: list[dict] = []
    missed_counter = np.zeros(y_true.shape[1], dtype=np.int64)
    true_ranks_all: list[float] = []
    support_gaps: list[float] = []
    failure_gaps: list[float] = []
    failure_near_miss_flags: list[int] = []

    for sample_index in range(y_true.shape[0]):
        target = y_true[sample_index]
        scores = ranking_scores[sample_index]
        true_support = np.flatnonzero(np.abs(target) > 1e-9).astype(np.int64)
        pred_support = np.argpartition(scores, -k)[-k:]
        pred_support = pred_support[np.argsort(-scores[pred_support])].astype(np.int64)
        true_set = set(true_support.tolist())
        pred_set = set(pred_support.tolist())
        missed_true = sorted(true_set - pred_set)
        false_selected = sorted(pred_set - true_set)
        exact = int(true_set == pred_set)

        ranks = score_ranks(scores)
        true_ranks = ranks[true_support]
        true_ranks_all.extend(true_ranks.astype(float).tolist())

        other_idx = np.setdiff1d(np.arange(len(scores)), true_support, assume_unique=True)
        support_gap = float(scores[true_support].min() - scores[other_idx].max()) if len(true_support) and len(other_idx) else 0.0
        support_gaps.append(support_gap)
        if not exact:
            failure_gaps.append(support_gap)

        near_miss = 0
        if missed_true and false_selected:
            near_candidates = set()
            for rid in missed_true:
                near_candidates.update(neighbor_map[int(rid)])
                missed_counter[int(rid)] += 1
            near_miss = int(any(idx in near_candidates for idx in false_selected))
        elif missed_true:
            for rid in missed_true:
                missed_counter[int(rid)] += 1

        if not exact:
            failure_near_miss_flags.append(near_miss)

        rows.append(
            {
                "sample_id": int(sample_ids[sample_index]),
                "sample_index": sample_index,
                "support_exact": exact,
                "support_overlap": len(true_set & pred_set) / max(k, 1),
                "missed_true_count": len(missed_true),
                "false_selected_count": len(false_selected),
                "mean_true_rank": float(np.mean(true_ranks)) if len(true_ranks) else 0.0,
                "worst_true_rank": int(np.max(true_ranks)) if len(true_ranks) else 0,
                "support_gap": support_gap,
                "near_miss_flag": near_miss,
                "true_support": ";".join(str(v) for v in true_support.tolist()),
                "pred_support": ";".join(str(v) for v in pred_support.tolist()),
                "missed_true": ";".join(str(v) for v in missed_true),
                "false_selected": ";".join(str(v) for v in false_selected),
            }
        )

    top_ids = np.argsort(-missed_counter)[: min(12, len(missed_counter))]
    summary = {
        "failure_count": int(sum(1 for row in rows if row["support_exact"] == 0)),
        "mean_true_rank": float(np.mean(true_ranks_all)) if true_ranks_all else 0.0,
        "median_true_rank": float(np.median(true_ranks_all)) if true_ranks_all else 0.0,
        "p90_true_rank": float(np.percentile(true_ranks_all, 90)) if true_ranks_all else 0.0,
        "mean_support_gap": float(np.mean(support_gaps)) if support_gaps else 0.0,
        "mean_support_gap_failures": float(np.mean(failure_gaps)) if failure_gaps else 0.0,
        "near_miss_rate_on_failures": float(np.mean(failure_near_miss_flags)) if failure_near_miss_flags else 0.0,
        "top_missed_edges": [
            {"edge_id": int(idx), "miss_count": int(missed_counter[idx])}
            for idx in top_ids
            if missed_counter[idx] > 0
        ],
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_training_curves(history: list[dict], path: Path) -> None:
    if plt is None or not history:
        return
    epochs = [int(item["epoch"]) for item in history]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    axes[0].plot(epochs, [float(item["train_id_exact_rate"]) for item in history], label="Train ID", color="#1f77b4")
    axes[0].plot(epochs, [float(item["val_id_exact_rate"]) for item in history], label="Val ID", color="#d62728")
    axes[0].plot(epochs, [float(item["train_value_accuracy"]) for item in history], label="Train Value", color="#2ca02c", linestyle="--")
    axes[0].plot(epochs, [float(item["val_value_accuracy"]) for item in history], label="Val Value", color="#9467bd", linestyle="--")
    axes[0].set_title("Training Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Metric")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(epochs, [float(item["train_loss"]) for item in history], label="Train Loss", color="#1f77b4")
    if "loss_changed" in history[0]:
        axes[1].plot(epochs, [float(item["loss_changed"]) for item in history], label="Changed Loss", color="#ff7f0e")
    if "loss_unchanged" in history[0]:
        axes[1].plot(epochs, [float(item["loss_unchanged"]) for item in history], label="Unchanged Loss", color="#2ca02c")
    if "loss_ranking" in history[0]:
        axes[1].plot(epochs, [float(item["loss_ranking"]) for item in history], label="Ranking Loss", color="#d62728")
    axes[1].set_title("Loss Breakdown")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(frameon=False, fontsize=9)

    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_error_overview(rows: list[dict], summary: dict, path: Path) -> None:
    if plt is None or not rows:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True)

    missed_counts = [int(row["missed_true_count"]) for row in rows]
    mean_true_ranks = [float(row["mean_true_rank"]) for row in rows]
    support_gaps = [float(row["support_gap"]) for row in rows]
    top_missed = summary.get("top_missed_edges", [])

    axes[0, 0].hist(missed_counts, bins=np.arange(-0.5, max(missed_counts) + 1.5, 1), color="#1f77b4", edgecolor="white")
    axes[0, 0].set_title("Missed True Edges")
    axes[0, 0].set_xlabel("Missed count per sample")
    axes[0, 0].set_ylabel("Samples")

    axes[0, 1].hist(mean_true_ranks, bins=20, color="#ff7f0e", edgecolor="white")
    axes[0, 1].set_title("True-Edge Rank")
    axes[0, 1].set_xlabel("Mean rank of true changed edges")
    axes[0, 1].set_ylabel("Samples")

    axes[1, 0].hist(support_gaps, bins=20, color="#2ca02c", edgecolor="white")
    axes[1, 0].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("Support Gap")
    axes[1, 0].set_xlabel("min(true score) - max(false score)")
    axes[1, 0].set_ylabel("Samples")

    if top_missed:
        ids = [item["edge_id"] for item in top_missed]
        counts = [item["miss_count"] for item in top_missed]
        axes[1, 1].bar(range(len(ids)), counts, color="#d62728")
        axes[1, 1].set_xticks(range(len(ids)))
        axes[1, 1].set_xticklabels([str(v) for v in ids], rotation=45, ha="right")
        axes[1, 1].set_title("Most Missed Edge IDs")
        axes[1, 1].set_xlabel("Edge ID")
        axes[1, 1].set_ylabel("Miss count")
    else:
        axes[1, 1].text(0.5, 0.5, "No missed edges", ha="center", va="center")
        axes[1, 1].set_axis_off()

    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    meta_path = Path(args.meta_path).resolve()
    out_dir = Path(args.out_dir).resolve()

    meta = load_json(meta_path)
    train_metrics = load_json(out_dir / "train_metrics.json")
    x_split, y_split, sample_ids, topology, _ = load_split_from_meta(meta_path, args.split)
    standardization = np.load(out_dir / "standardization.npz")
    boundary_nodes = standardization["boundary_nodes"].astype(np.int64)
    x_split = apply_standardization(x_split, boundary_nodes, standardization["mean"], standardization["std"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_type, topology, int(np.prod(x_split.shape[1:])), device, train_metrics)
    model.load_state_dict(torch.load(out_dir / "model_last.pt", map_location=device))
    ranking_scores, value_pred = run_predictions(model, args.model_type, x_split, args.batch_size, device)

    ranking_arg = ranking_scores if args.model_type in {"modelv2", "modelv3", "modelo1_gnn", "modelo1_mlp1"} else None
    metrics = compute_fixedk_metrics(value_pred, y_split, k=int(meta["k"]), ranking_scores=ranking_arg)
    metrics.pop("per_sample", None)
    detail_rows, detail_summary = analyze_details(
        sample_ids=sample_ids,
        y_true=y_split,
        ranking_scores=ranking_scores if args.model_type in {"modelv2", "modelv3", "modelo1_gnn", "modelo1_mlp1"} else np.abs(value_pred),
        k=int(meta["k"]),
        resistor_edges=[tuple(edge) for edge in meta["topology"]["resistor_edges"]],
    )

    summary_payload = {
        "model_type": args.model_type,
        "meta_path": str(meta_path),
        "out_dir": str(out_dir),
        "split": args.split,
        "grid_size": int(meta["topology"]["grid_size"]),
        "k": int(meta["k"]),
        "port_count": int(meta["topology"]["port_count"]),
        **metrics,
        **detail_summary,
    }

    write_csv(out_dir / f"{args.split}_error_details.csv", detail_rows)
    (out_dir / f"{args.split}_error_summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_training_curves(train_metrics.get("history", []), out_dir / "training_curves.png")
    plot_error_overview(detail_rows, detail_summary, out_dir / "error_analysis_overview.png")

    print(
        f"[analysis] id_exact_rate={summary_payload['id_exact_rate']:.4f} "
        f"value_accuracy={summary_payload['value_accuracy']:.4f} "
        f"mean_true_rank={summary_payload['mean_true_rank']:.3f} "
        f"near_miss_rate_on_failures={summary_payload['near_miss_rate_on_failures']:.3f}"
    )


if __name__ == "__main__":
    main()
