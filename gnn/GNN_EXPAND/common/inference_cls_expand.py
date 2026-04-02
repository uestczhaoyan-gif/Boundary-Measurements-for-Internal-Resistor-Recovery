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
    build_cls_dataset,
    confusion,
    dump_json,
    expit,
    load_partial_state_dict,
    macro_f1,
    resolve_inference_runtime_paths,
    select_focus_indices,
    split_indices,
    standardize_graph_voltage,
)
from models import PhysicsInformedGNNClassifier
from topologies import get_topology


def parse_args(default_data_path: str):
    parser = argparse.ArgumentParser(description="Topology-expand GNN classifier inference.")
    parser.add_argument("--data-path", default=default_data_path)
    parser.add_argument("--dataset-tag", default="")
    parser.add_argument("--cache-path", default="cache_dataset_cls_expand.npz")
    parser.add_argument("--model-path", default="./outputs/model_last.pt")
    parser.add_argument("--metrics-path", default="./outputs/metrics.json")
    parser.add_argument("--standardization", default="./outputs/standardization.npz")
    parser.set_defaults(dataset_subdir=True)
    parser.add_argument("--dataset-subdir", dest="dataset_subdir", action="store_true")
    parser.add_argument("--no-dataset-subdir", dest="dataset_subdir", action="store_false")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260325)
    parser.set_defaults(focus_high_change=True)
    parser.add_argument("--focus-high-change", dest="focus_high_change", action="store_true")
    parser.add_argument("--no-focus-high-change", dest="focus_high_change", action="store_false")
    parser.add_argument("--min-true-change", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--excitation-chunk-size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main(stage_name: str, topology_key: str, default_data_path: str, runtime_dir: Path | None = None):
    args = parse_args(default_data_path)
    script_dir = Path(runtime_dir).resolve() if runtime_dir is not None else Path(__file__).resolve().parent
    resolve_inference_runtime_paths(args, script_dir, "cache_dataset_cls_expand.npz")
    topology = get_topology(topology_key)
    cache_path = Path(args.cache_path)
    if not cache_path.exists():
        build_cls_dataset(Path(args.data_path), cache_path, topology)
    metrics = {}
    if Path(args.metrics_path).exists():
        metrics = json.loads(Path(args.metrics_path).read_text(encoding="utf-8"))
    thresholds = metrics.get("best_thresholds", [0.5, 0.5, 0.5])

    d = np.load(args.cache_path)
    x = d["x"].astype(np.float32)
    y = d["y"].astype(np.int64)
    std = np.load(args.standardization)
    ext_nodes = std["ext_nodes"].astype(np.int64)
    x = standardize_graph_voltage(x, std["mean"], std["std"], ext_nodes)
    _, _, test_idx = split_indices(len(x), args.seed)
    x_test = x[test_idx]
    y_test = y[test_idx]
    sample_idx = select_focus_indices(test_idx, y, args.num_samples, args.seed, args.focus_high_change, args.min_true_change)
    idx_to_local = {int(idx): pos for pos, idx in enumerate(test_idx)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicsInformedGNNClassifier(
        topology=topology,
        in_dim=4,
        hidden_dim=args.hidden_dim,
        proj_dim=args.proj_dim,
        out_dim=3,
        heads=args.gat_heads,
        excitation_chunk_size=args.excitation_chunk_size,
        dropout=args.dropout,
    ).to(device)
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    load_partial_state_dict(model, args.model_path, device, label=f"{stage_name}/cls_infer")
    model.eval()

    logits_all = []
    with torch.no_grad():
        for start in range(0, len(x_test), 64):
            xb = torch.from_numpy(x_test[start:start + 64]).to(device)
            logits = model(xb).cpu().numpy()
            logits_all.append(logits)
    logits_all = np.concatenate(logits_all, axis=0)
    probs_all = expit(logits_all)
    pred_all = (
        (probs_all[:, 0] > thresholds[0]).astype(np.int64)
        + (probs_all[:, 1] > thresholds[1]).astype(np.int64)
        + (probs_all[:, 2] > thresholds[2]).astype(np.int64)
    )
    cm = confusion(pred_all, y_test)
    final_f1 = macro_f1(cm)

    eval_metrics = {
        "stage_name": stage_name,
        "topology_key": topology.key,
        "dataset_tag": args.dataset_tag,
        "test_macro_f1": final_f1,
        "best_thresholds": thresholds,
        "confusion_matrix": cm.tolist(),
    }
    dump_json(Path(args.model_path).parent / "inference_eval.json", eval_metrics)

    results = []
    for idx in sample_idx:
        local_idx = idx_to_local[int(idx)]
        results.append(
            {
                "sample_index": int(idx),
                "true_count": int(y[idx]),
                "pred_count": int(pred_all[local_idx]),
                "probs": probs_all[local_idx].tolist(),
            }
        )
    dump_json(Path(args.model_path).parent / "inference_samples.json", results)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print(f"test_macro_f1={final_f1:.4f}")


if __name__ == "__main__":
    raise RuntimeError("Use a stage-specific wrapper under gnn/GNN_EXPAND/<stage>/cls/inference.py")
