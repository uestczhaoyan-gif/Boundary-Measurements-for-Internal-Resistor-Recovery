from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
VENDOR_DIR = WORKSPACE_ROOT / ".vendor_torchpy311"
if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

import numpy as np
import torch

from expand_common import (
    build_reg_dataset,
    confusion,
    dump_json,
    load_partial_state_dict,
    macro_f1,
    resolve_inference_runtime_paths,
    select_focus_indices,
    split_indices,
    standardize_graph_voltage,
)
from models import PhysicsInformedGNNRegressor
from topologies import get_topology


def parse_args(default_data_path: str):
    parser = argparse.ArgumentParser(description="Topology-expand GNN regression inference.")
    parser.add_argument("--data-path", default=default_data_path)
    parser.add_argument("--dataset-tag", default="")
    parser.add_argument("--cache-path", default="cache_dataset_reg_expand.npz")
    parser.add_argument("--model-path", default="./outputs/model_last.pt")
    parser.add_argument("--metrics-path", default="./outputs/metrics.json")
    parser.add_argument("--standardization", default="./outputs/standardization.npz")
    parser.set_defaults(dataset_subdir=True)
    parser.add_argument("--dataset-subdir", dest="dataset_subdir", action="store_true")
    parser.add_argument("--no-dataset-subdir", dest="dataset_subdir", action="store_false")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260325)
    parser.set_defaults(focus_high_change=True)
    parser.add_argument("--focus-high-change", dest="focus_high_change", action="store_true")
    parser.add_argument("--no-focus-high-change", dest="focus_high_change", action="store_false")
    parser.add_argument("--min-true-change", type=int, default=2)
    parser.add_argument("--count-threshold", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--edge-hidden", type=int, default=128)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--excitation-chunk-size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-abs", type=float, default=300.0)
    return parser.parse_args()


def main(stage_name: str, topology_key: str, default_data_path: str, runtime_dir: Path | None = None):
    args = parse_args(default_data_path)
    script_dir = Path(runtime_dir).resolve() if runtime_dir is not None else Path(__file__).resolve().parent
    resolve_inference_runtime_paths(args, script_dir, "cache_dataset_reg_expand.npz")
    topology = get_topology(topology_key)
    cache_path = Path(args.cache_path)
    if not cache_path.exists():
        build_reg_dataset(Path(args.data_path), cache_path, topology)
    if args.count_threshold is None and Path(args.metrics_path).exists():
        metrics = json.loads(Path(args.metrics_path).read_text(encoding="utf-8"))
        args.count_threshold = float(metrics.get("best_count_threshold", 50.0))
    if args.count_threshold is None:
        args.count_threshold = 50.0

    d = np.load(args.cache_path)
    x = d["x"].astype(np.float32)
    y_change = d["y_change"].astype(np.float32)
    y_delta = d["y_delta"].astype(np.float32)
    std = np.load(args.standardization)
    ext_nodes = std["ext_nodes"].astype(np.int64)
    x = standardize_graph_voltage(x, std["mean"], std["std"], ext_nodes)

    _, _, test_idx = split_indices(len(x), args.seed)
    x_test = x[test_idx]
    y_change_test = y_change[test_idx]
    y_delta_test = y_delta[test_idx]
    idx_to_local = {int(idx): pos for pos, idx in enumerate(test_idx)}
    sample_idx = select_focus_indices(test_idx, y_change.sum(axis=1), args.num_samples, args.seed, args.focus_high_change, args.min_true_change)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicsInformedGNNRegressor(
        topology=topology,
        in_dim=4,
        hidden_dim=args.hidden_dim,
        edge_hidden=args.edge_hidden,
        heads=args.gat_heads,
        excitation_chunk_size=args.excitation_chunk_size,
        dropout=args.dropout,
        max_abs=args.max_abs,
    ).to(device)
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    load_partial_state_dict(model, args.model_path, device, label=f"{stage_name}/reg_infer")
    model.eval()

    pred_all = []
    mask_prob_all = []
    with torch.no_grad():
        for start in range(0, len(x_test), 64):
            xb = torch.from_numpy(x_test[start:start + 64]).to(device)
            pred, aux = model(xb, return_aux=True)
            pred_all.append(pred.cpu().numpy())
            mask_prob_all.append(aux["mask_prob"].cpu().numpy())
    pred_all = np.concatenate(pred_all, axis=0)
    mask_prob_all = np.concatenate(mask_prob_all, axis=0)

    changed_mask = y_change_test > 0.5
    mae_all = float(np.abs(pred_all - y_delta_test).mean())
    mae_changed = float(np.abs(pred_all[changed_mask] - y_delta_test[changed_mask]).mean()) if changed_mask.any() else 0.0
    pred_counts_all = (np.abs(pred_all) > args.count_threshold).sum(axis=1).astype(np.int64)
    true_counts_all = changed_mask.sum(axis=1).astype(np.int64)
    cm = confusion(np.clip(pred_counts_all, 0, 3), np.clip(true_counts_all, 0, 3), 4)
    eval_metrics = {
        "stage_name": stage_name,
        "topology_key": topology.key,
        "dataset_tag": args.dataset_tag,
        "mae_all": mae_all,
        "mae_changed": mae_changed,
        "count_threshold": args.count_threshold,
        "count_macro_f1": macro_f1(cm),
        "confusion_matrix": cm.tolist(),
    }
    dump_json(Path(args.model_path).parent / "inference_eval.json", eval_metrics)

    results = []
    for idx in sample_idx:
        local_idx = idx_to_local[int(idx)]
        true_ids = np.where(y_change[idx] > 0.5)[0].astype(int).tolist()
        pred_ids = np.where(np.abs(pred_all[local_idx]) > args.count_threshold)[0].astype(int).tolist()
        results.append(
            {
                "sample_index": int(idx),
                "true_ids": true_ids,
                "pred_ids": pred_ids,
                "pred_deltas": [float(pred_all[local_idx][eid]) for eid in pred_ids],
                "true_deltas": [float(y_delta[idx][eid]) for eid in true_ids],
                "pred_count": int(pred_counts_all[local_idx]),
                "true_count": int(true_counts_all[local_idx]),
                "avg_mask_prob": float(mask_prob_all[local_idx].mean()),
            }
        )
    dump_json(Path(args.model_path).parent / "inference_samples.json", results)
    print("Derived Count Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print(f"mae_all={mae_all:.4f} | mae_changed={mae_changed:.4f}")


if __name__ == "__main__":
    raise RuntimeError("Use a stage-specific wrapper under gnn/GNN_EXPAND/<stage>/reg/inference.py")
