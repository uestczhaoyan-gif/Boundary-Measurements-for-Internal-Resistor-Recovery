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
    apply_near_miss,
    build_cls_dataset,
    build_edge_adjacency,
    build_reg_dataset,
    confusion,
    dump_json,
    expit,
    extract_edge_values,
    load_partial_state_dict,
    macro_f1,
    predict_count_from_coral,
    resolve_input_data_path,
    split_indices,
    standardize_graph_voltage,
)
from models import PhysicsInformedGNNClassifier, PhysicsInformedGNNRegressor
from topologies import get_topology


def parse_args(default_data_path: str):
    parser = argparse.ArgumentParser(description="Topology-expand joint GNN CMEI inference.")
    parser.add_argument("--data-path", default=default_data_path)
    parser.add_argument("--dataset-tag", default="")
    parser.add_argument("--out-dir", default="./outputs")
    parser.add_argument("--cls-dir", default="../cls")
    parser.add_argument("--reg-dir", default="../reg")
    parser.add_argument("--split-seed", type=int, default=20260325)
    parser.add_argument("--seed", type=int, default=20260327)
    parser.add_argument("--mse-max", type=float, default=10000.0)
    parser.add_argument("--batch-size-cls", type=int, default=64)
    parser.add_argument("--batch-size-reg", type=int, default=64)
    parser.add_argument("--detail-samples", type=int, default=5)
    parser.add_argument("--near-miss-ratio", type=float, default=0.92)
    parser.set_defaults(enable_near_miss=True)
    parser.add_argument("--enable-near-miss", dest="enable_near_miss", action="store_true")
    parser.add_argument("--no-near-miss", dest="enable_near_miss", action="store_false")
    return parser.parse_args()


def main(stage_name: str, topology_key: str, default_data_path: str, runtime_dir: Path | None = None):
    args = parse_args(default_data_path)
    script_dir = Path(runtime_dir).resolve() if runtime_dir is not None else Path(__file__).resolve().parent
    data_path = resolve_input_data_path(args.data_path, script_dir)
    dataset_tag = args.dataset_tag or data_path.stem
    out_dir = (script_dir / "outputs" / dataset_tag) if args.out_dir == "./outputs" else Path(args.out_dir) / dataset_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    topology = get_topology(topology_key)

    cls_dir = resolve_input_data_path(args.cls_dir, script_dir)
    reg_dir = resolve_input_data_path(args.reg_dir, script_dir)
    cls_cache = cls_dir / "cache" / dataset_tag / "cache_dataset_cls_expand.npz"
    reg_cache = reg_dir / "cache" / dataset_tag / "cache_dataset_reg_expand.npz"
    if not cls_cache.exists():
        build_cls_dataset(data_path, cls_cache, topology)
    if not reg_cache.exists():
        build_reg_dataset(data_path, reg_cache, topology)

    cls_out = cls_dir / "outputs" / dataset_tag
    reg_out = reg_dir / "outputs" / dataset_tag
    cls_model_path = cls_out / "model_last.pt"
    cls_metrics_path = cls_out / "metrics.json"
    cls_std_path = cls_out / "standardization.npz"
    reg_model_path = reg_out / "model_last.pt"
    reg_metrics_path = reg_out / "metrics.json"
    reg_std_path = reg_out / "standardization.npz"

    cls_npz = np.load(cls_cache)
    x_cls = cls_npz["x"].astype(np.float32)
    y_cls = cls_npz["y"].astype(np.int64)
    cls_std = np.load(cls_std_path)
    cls_ext_nodes = cls_std["ext_nodes"].astype(np.int64)
    x_cls = standardize_graph_voltage(x_cls, cls_std["mean"], cls_std["std"], cls_ext_nodes)

    reg_npz = np.load(reg_cache)
    x_reg = reg_npz["x"].astype(np.float32)
    y_change = reg_npz["y_change"].astype(np.float32)
    y_delta = reg_npz["y_delta"].astype(np.float32)
    reg_std = np.load(reg_std_path)
    reg_ext_nodes = reg_std["ext_nodes"].astype(np.int64)
    x_reg = standardize_graph_voltage(x_reg, reg_std["mean"], reg_std["std"], reg_ext_nodes)

    _, _, test_idx = split_indices(len(y_cls), args.split_seed)
    x_cls_test = x_cls[test_idx]
    x_reg_test = x_reg[test_idx]
    y_cls_test = y_cls[test_idx]
    y_change_test = y_change[test_idx]
    y_delta_test = y_delta[test_idx]

    cls_metrics = json.loads(cls_metrics_path.read_text(encoding="utf-8"))
    reg_metrics = json.loads(reg_metrics_path.read_text(encoding="utf-8"))
    cls_thresholds = cls_metrics.get("best_thresholds", [0.5, 0.5, 0.5])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_model = PhysicsInformedGNNClassifier(
        topology=topology,
        in_dim=4,
        hidden_dim=int(cls_metrics.get("hidden_dim", 128)),
        proj_dim=int(cls_metrics.get("proj_dim", 128)),
        out_dim=3,
        heads=int(cls_metrics.get("gat_heads", 4)),
        excitation_chunk_size=int(cls_metrics.get("excitation_chunk_size", 4)),
        dropout=float(cls_metrics.get("dropout", 0.1)),
    ).to(device)
    if not cls_model_path.exists():
        raise FileNotFoundError(f"CLS model not found: {cls_model_path}")
    load_partial_state_dict(cls_model, str(cls_model_path), device, label=f"{stage_name}/joint_cls")
    cls_model.eval()
    reg_model = PhysicsInformedGNNRegressor(
        topology=topology,
        in_dim=4,
        hidden_dim=int(reg_metrics.get("hidden_dim", 128)),
        edge_hidden=int(reg_metrics.get("edge_hidden", 128)),
        heads=int(reg_metrics.get("gat_heads", 4)),
        excitation_chunk_size=int(reg_metrics.get("excitation_chunk_size", 4)),
        dropout=float(reg_metrics.get("dropout", 0.1)),
        max_abs=float(reg_metrics.get("max_abs", 300.0)),
        mask_init_prob=float(reg_metrics.get("mask_init_prob", 0.45)),
    ).to(device)
    if not reg_model_path.exists():
        raise FileNotFoundError(f"REG model not found: {reg_model_path}")
    load_partial_state_dict(reg_model, str(reg_model_path), device, label=f"{stage_name}/joint_reg")
    reg_model.eval()

    cls_probs_all = []
    with torch.no_grad():
        for start in range(0, len(x_cls_test), args.batch_size_cls):
            xb = torch.from_numpy(x_cls_test[start:start + args.batch_size_cls]).to(device)
            logits, _aux = cls_model(xb, return_aux=True)
            cls_probs_all.append(expit(logits.cpu().numpy()))
    cls_probs_all = np.concatenate(cls_probs_all, axis=0)
    pred_counts = predict_count_from_coral(cls_probs_all, cls_thresholds)

    reg_pred_all = []
    reg_mask_all = []
    with torch.no_grad():
        for start in range(0, len(x_reg_test), args.batch_size_reg):
            xb = torch.from_numpy(x_reg_test[start:start + args.batch_size_reg]).to(device)
            pred, aux = reg_model(xb, return_aux=True)
            reg_pred_all.append(pred.cpu().numpy())
            reg_mask_all.append(aux["mask_prob"].cpu().numpy())
    reg_pred_all = np.concatenate(reg_pred_all, axis=0)
    reg_mask_all = np.concatenate(reg_mask_all, axis=0)

    adjacency = build_edge_adjacency(topology.resistor_edges)
    final_delta_all = np.zeros_like(reg_pred_all, dtype=np.float32)
    correct_id_total = 0
    true_id_total = int(y_change_test.sum())

    for i in range(len(test_idx)):
        k_pred = int(pred_counts[i])
        reg_abs = np.abs(reg_pred_all[i])
        sorted_ids = np.argsort(-reg_abs)
        topk_ids = sorted_ids[:k_pred].astype(int).tolist()
        if args.enable_near_miss:
            topk_ids = apply_near_miss(
                topk_ids,
                sorted_ids.astype(int).tolist(),
                reg_abs,
                reg_mask_all[i],
                adjacency,
                k_pred,
                near_ratio=args.near_miss_ratio,
            )
        if k_pred > 0:
            final_delta_all[i, topk_ids] = reg_pred_all[i, topk_ids]
        true_ids = np.where(y_change_test[i] > 0.5)[0].astype(int).tolist()
        pred_ids = np.where(np.abs(final_delta_all[i]) > 0)[0].astype(int).tolist()
        correct_id_total += len(set(true_ids).intersection(pred_ids))

    num_accuracy = float((pred_counts == y_cls_test).mean())
    cm = confusion(pred_counts, y_cls_test, num_classes=4)
    f1 = macro_f1(cm)
    id_recall = correct_id_total / max(true_id_total, 1)
    mse = float(np.mean((final_delta_all - y_delta_test) ** 2))
    s_num = num_accuracy * 100.0
    s_f1 = f1 * 100.0
    s_id = id_recall * 100.0
    s_mse = 100.0 * max(0.0, 1.0 - mse / args.mse_max)
    cmei = 0.40 * s_id + 0.30 * s_mse + 0.15 * s_f1 + 0.15 * s_num

    metrics = {
        "stage_name": stage_name,
        "topology_key": topology.key,
        "topology_title": topology.title,
        "num_nodes": topology.num_nodes,
        "num_resistors": topology.num_resistors,
        "dataset_tag": dataset_tag,
        "data_path": str(data_path),
        "models": {"gnn_cls": str(cls_model_path), "gnn_reg": str(reg_model_path)},
        "score_weights": {"S_id": 0.40, "S_mse": 0.30, "S_F1": 0.15, "S_num": 0.15},
        "cls_thresholds": cls_thresholds,
        "near_miss": {"enabled": bool(args.enable_near_miss), "ratio": args.near_miss_ratio},
        "num_accuracy": num_accuracy,
        "macro_f1": f1,
        "id_recall": id_recall,
        "mse_all_edges": mse,
        "scores": {"S_num": s_num, "S_F1": s_f1, "S_id": s_id, "S_mse": s_mse, "CMEI": cmei},
        "confusion_matrix": cm.tolist(),
        "mse_max": args.mse_max,
        "test_size": len(test_idx),
    }
    dump_json(out_dir / "cmei_metrics.json", metrics)

    detail_rows = []
    detail_selected = [i for i in range(len(test_idx)) if int(y_cls_test[i]) in (2, 3)][: min(args.detail_samples, len(test_idx))]
    for local_idx in detail_selected:
        dataset_index = int(test_idx[local_idx])
        true_delta = y_delta_test[local_idx]
        pred_delta = final_delta_all[local_idx]
        true_ids = np.where(y_change_test[local_idx] > 0.5)[0].astype(int).tolist()
        pred_ids = np.where(np.abs(pred_delta) > 0)[0].astype(int).tolist()
        reg_top_ids = np.argsort(-np.abs(reg_pred_all[local_idx]))[:8].astype(int).tolist()
        detail_rows.append(
            {
                "sample_index": dataset_index,
                "true_k": int(y_cls_test[local_idx]),
                "pred_k": int(pred_counts[local_idx]),
                "true_ids": true_ids,
                "pred_ids": pred_ids,
                "true_deltas": extract_edge_values(true_delta, true_ids),
                "pred_deltas": extract_edge_values(pred_delta, pred_ids),
                "cls_probs": cls_probs_all[local_idx].tolist(),
                "reg_top_abs_ids": reg_top_ids,
                "reg_top_abs_values": [float(reg_pred_all[local_idx][eid]) for eid in reg_top_ids],
                "reg_top_mask_probs": [float(reg_mask_all[local_idx][eid]) for eid in reg_top_ids],
            }
        )
    dump_json(out_dir / "detail_samples.json", detail_rows)

    report_lines = [
        "# GNN Expand CMEI Report",
        "",
        f"- Stage: `{stage_name}`",
        f"- Topology: `{topology.title}`",
        f"- Dataset: `{dataset_tag}`",
        f"- Count accuracy: `{num_accuracy:.4f}`",
        f"- Macro-F1: `{f1:.4f}`",
        f"- ID recall: `{id_recall:.4f}`",
        f"- All-edge MSE: `{mse:.4f}`",
        f"- **CMEI={cmei:.2f}**",
    ]
    (out_dir / "cmei_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"CMEI={cmei:.2f} | num_accuracy={num_accuracy:.4f} | macro_f1={f1:.4f} | id_recall={id_recall:.4f}")


if __name__ == "__main__":
    raise RuntimeError("Use a stage-specific wrapper under gnn/GNN_EXPAND/<stage>/joint_inference/inference.py")
