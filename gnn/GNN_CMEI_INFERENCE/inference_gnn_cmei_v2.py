import argparse
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

GNN_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = GNN_ROOT.parent
if str(GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GNN_ROOT))

from vendor_bootstrap import bootstrap_vendor_paths, format_dependency_import_error

_BOOTSTRAP_RESULTS = bootstrap_vendor_paths(WORKSPACE_ROOT)

try:
    import numpy as np
except Exception as exc:
    raise ImportError(format_dependency_import_error("numpy", exc, _BOOTSTRAP_RESULTS)) from None

try:
    import torch
except Exception as exc:
    raise ImportError(format_dependency_import_error("torch", exc, _BOOTSTRAP_RESULTS)) from None

try:
    from scipy.special import expit
except Exception:
    def expit(x):
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x, dtype=np.float64)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[~pos])
        out[~pos] = exp_x / (1.0 + exp_x)
        return out


BASE_R = 1000.0
GRID = 8
DEFAULT_MAIN_DATA_PATH = "data/training_data64Nodes_2.csv"


def sanitize_dataset_tag(raw_tag):
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", raw_tag.strip())
    safe = safe.strip("._-")
    return safe or "dataset"


def resolve_input_data_path(raw_path, script_dir):
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    gnn_root = script_dir.parent
    workspace_root = gnn_root.parent
    candidates = [path, workspace_root / path, gnn_root / path, script_dir / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (workspace_root / path).resolve()


def resolve_output_dir(raw_path, script_dir):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    gnn_root = script_dir.parent
    workspace_root = gnn_root.parent
    candidates = [path, script_dir / path, gnn_root / path, workspace_root / path]
    for candidate in candidates:
        if candidate.exists() or candidate.parent.exists():
            return candidate.resolve()
    return (script_dir / path).resolve()


def candidate_dataset_tags(requested_tag, data_path):
    tags = []
    for tag in [sanitize_dataset_tag(requested_tag), sanitize_dataset_tag(data_path.stem)]:
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def resolve_default_artifact_path(root_dir, filename, requested_tag, data_path):
    root_dir = Path(root_dir)
    for tag in candidate_dataset_tags(requested_tag, data_path):
        candidate = root_dir / tag / filename
        if candidate.exists():
            return candidate
    legacy = root_dir / filename
    if legacy.exists():
        return legacy
    return root_dir / requested_tag / filename


def split_indices(n, seed):
    rng = random.Random(seed)
    ids = list(range(n))
    rng.shuffle(ids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def load_module_attr(module_name, file_path, attr_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, attr_name)


def standardize_gnn(x, mean, std, ext_nodes):
    x_std = x.copy()
    x_std[:, :, ext_nodes, 2] = (x_std[:, :, ext_nodes, 2] - mean) / std
    return x_std.astype(np.float32)


def inject_voltage_noise(x, ext_nodes, noise_std, seed):
    if noise_std <= 0:
        return x
    noisy = x.copy()
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, size=noisy[:, :, ext_nodes, 2].shape).astype(np.float32)
    noisy[:, :, ext_nodes, 2] += noise
    return noisy


def confusion(pred, true, num_classes=4):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(pred, true):
        cm[t, p] += 1
    return cm


def macro_f1(cm):
    vals = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        vals.append(0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(vals))


def build_resistor_edges(grid=GRID):
    edges = []
    for r in range(grid):
        for c in range(grid - 1):
            edges.append((r * grid + c, r * grid + c + 1))
        if r < grid - 1:
            for c in range(grid):
                edges.append((r * grid + c, (r + 1) * grid + c))
    return edges


def build_edge_adjacency(edges):
    adjacency = [set() for _ in edges]
    for i, (u1, v1) in enumerate(edges):
        nodes_i = {u1, v1}
        for j in range(i + 1, len(edges)):
            u2, v2 = edges[j]
            if nodes_i.intersection({u2, v2}):
                adjacency[i].add(j)
                adjacency[j].add(i)
    return adjacency


def predict_count_from_coral(probs, thresholds):
    thr = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    return (probs > thr).sum(axis=1).astype(np.int64)


def predict_count_from_reg_abs(reg_abs, count_threshold, max_k=3):
    return min(int(np.sum(np.asarray(reg_abs) >= float(count_threshold))), int(max_k))


def arbitrate_count_with_reg(
    k_cls,
    reg_abs,
    reg_prob,
    sorted_ids,
    reg_count_threshold,
    max_k=3,
    dynamic_up_prob=0.95,
    dynamic_down_prob=0.45,
):
    k_cls = max(0, min(int(k_cls), int(max_k)))
    k_reg = predict_count_from_reg_abs(reg_abs, reg_count_threshold, max_k=max_k)
    if k_cls == k_reg:
        return k_cls, k_reg, "agree"

    if k_reg > k_cls:
        if k_cls >= len(sorted_ids):
            return k_cls, k_reg, "cls_cap"
        challenger = int(sorted_ids[k_cls])
        if float(reg_prob[challenger]) >= float(dynamic_up_prob) or float(reg_abs[challenger]) >= float(reg_count_threshold):
            return min(k_reg, k_cls + 1), k_reg, "reg_dynamic_up"
        return k_cls, k_reg, "cls_keep_lower"

    if k_cls <= 0:
        return k_cls, k_reg, "cls_zero"
    weakest_selected = int(sorted_ids[k_cls - 1])
    if float(reg_prob[weakest_selected]) < float(dynamic_down_prob) and float(reg_abs[weakest_selected]) < float(reg_count_threshold):
        return k_reg, k_reg, "reg_dynamic_down"
    return k_cls, k_reg, "cls_keep_higher"


def apply_near_miss(topk_ids, sorted_ids, reg_abs, reg_prob, adjacency, k, near_ratio=0.92, protect_high_prob=0.85):
    if k <= 0 or len(sorted_ids) <= k or len(topk_ids) < 2:
        return topk_ids

    selected = list(topk_ids)
    selected_set = set(selected)
    adjacent_candidates = []
    for edge_id in selected:
        if adjacency[edge_id].intersection(selected_set - {edge_id}):
            adjacent_candidates.append(edge_id)
    if not adjacent_candidates:
        return selected

    weakest = min(adjacent_candidates, key=lambda eid: (float(reg_prob[eid]), float(reg_abs[eid])))
    challenger = int(sorted_ids[k])
    if challenger in selected_set:
        return selected

    weak_prob = float(reg_prob[weakest])
    weak_score = float(reg_abs[weakest])
    if weak_prob >= protect_high_prob:
        return selected
    chal_prob = float(reg_prob[challenger])
    chal_score = float(reg_abs[challenger])
    if chal_prob < near_ratio * weak_prob and chal_score < near_ratio * weak_score:
        return selected

    weak_adj = len(adjacency[weakest].intersection(selected_set - {weakest}))
    chal_adj = len(adjacency[challenger].intersection(selected_set - {weakest}))
    if chal_adj > weak_adj:
        return selected

    selected[selected.index(weakest)] = challenger
    selected = sorted(selected, key=lambda eid: -reg_abs[eid])
    return selected


def extract_edge_values(delta, ids):
    return [float(delta[int(eid)]) for eid in ids]


def main():
    parser = argparse.ArgumentParser(description="Unified GNN CLS+REG inference with CMEI v2 evaluation.")
    parser.add_argument("--data-path", default=DEFAULT_MAIN_DATA_PATH)
    parser.add_argument("--dataset-tag", default="")
    parser.add_argument("--out-dir", default="outputs/gnn_cmei_v2")
    parser.add_argument("--cls-dir", default="GNN_CLS/modelo3")
    parser.add_argument("--reg-dir", default="GNN_REG/o4a2")
    parser.add_argument("--split-seed", type=int, default=20260325)
    parser.add_argument("--seed", type=int, default=20260327)
    parser.add_argument("--mse-max", type=float, default=10000.0)
    parser.add_argument("--batch-size-cls", type=int, default=64)
    parser.add_argument("--batch-size-reg", type=int, default=64)
    parser.add_argument("--detail-samples", type=int, default=5)
    parser.add_argument("--near-miss-ratio", type=float, default=0.92)
    parser.add_argument("--near-miss-protect-prob", type=float, default=0.85)
    parser.add_argument("--noise-std", type=float, default=0.0, help="Standardized voltage noise std applied to test set only.")
    parser.add_argument("--noise-seed", type=int, default=20260331)
    parser.add_argument("--reg-count-threshold", type=float, default=-1.0)
    parser.add_argument("--reg-dynamic-up-prob", type=float, default=0.95)
    parser.add_argument("--reg-dynamic-down-prob", type=float, default=0.45)
    parser.add_argument("--max-k", type=int, default=3)
    parser.set_defaults(enable_near_miss=True)
    parser.add_argument("--enable-near-miss", dest="enable_near_miss", action="store_true")
    parser.add_argument("--no-near-miss", dest="enable_near_miss", action="store_false")
    parser.set_defaults(enable_reg_arbitration=False)
    parser.add_argument("--enable-reg-arbitration", dest="enable_reg_arbitration", action="store_true")
    parser.add_argument("--no-reg-arbitration", dest="enable_reg_arbitration", action="store_false")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_path = resolve_input_data_path(args.data_path, script_dir)
    dataset_tag = sanitize_dataset_tag(args.dataset_tag or data_path.stem)
    out_dir = resolve_output_dir(args.out_dir, script_dir) / dataset_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cls_dir = resolve_input_data_path(args.cls_dir, script_dir)
    reg_dir = resolve_input_data_path(args.reg_dir, script_dir)

    cls_cache = resolve_default_artifact_path(cls_dir / "cache", "cache_dataset_cls_graphattn.npz", dataset_tag, data_path)
    cls_out = resolve_default_artifact_path(cls_dir / "outputs", "model_last.pt", dataset_tag, data_path).parent
    cls_model_path = cls_out / "model_last.pt"
    cls_metrics_path = cls_out / "metrics.json"
    cls_std_path = cls_out / "standardization.npz"

    reg_cache = resolve_default_artifact_path(reg_dir / "cache", "cache_dataset_reg_graphattn.npz", dataset_tag, data_path)
    reg_out = resolve_default_artifact_path(reg_dir / "outputs", "model_last.pt", dataset_tag, data_path).parent
    reg_model_path = reg_out / "model_last.pt"
    reg_metrics_path = reg_out / "metrics.json"
    reg_std_path = reg_out / "standardization.npz"

    print(
        "[Runtime GNN CMEI] "
        f"dataset_tag={dataset_tag} | data_path={data_path} | "
        f"gnn_cls={cls_model_path} | gnn_reg={reg_model_path}"
    )

    cls_npz = np.load(cls_cache)
    x_cls = cls_npz["x"].astype(np.float32)
    y_cls = cls_npz["y"].astype(np.int64)
    cls_std = np.load(cls_std_path)
    cls_ext_nodes = cls_std["ext_nodes"].astype(np.int64)
    x_cls = standardize_gnn(x_cls, cls_std["mean"], cls_std["std"], cls_ext_nodes)

    reg_npz = np.load(reg_cache)
    x_reg = reg_npz["x"].astype(np.float32)
    y_change = reg_npz["y_change"].astype(np.float32)
    y_delta = reg_npz["y_delta"].astype(np.float32)
    reg_std = np.load(reg_std_path)
    reg_ext_nodes = reg_std["ext_nodes"].astype(np.int64)
    x_reg = standardize_gnn(x_reg, reg_std["mean"], reg_std["std"], reg_ext_nodes)

    if not np.array_equal(y_cls, y_change.sum(axis=1).astype(np.int64)):
        raise RuntimeError("GNN_CLS labels and GNN_REG change counts are inconsistent.")

    _, _, test_idx = split_indices(len(y_cls), args.split_seed)
    x_cls_test = inject_voltage_noise(x_cls[test_idx], cls_ext_nodes, args.noise_std, args.noise_seed)
    x_reg_test = inject_voltage_noise(x_reg[test_idx], reg_ext_nodes, args.noise_std, args.noise_seed + 1)
    y_cls_test = y_cls[test_idx]
    y_change_test = y_change[test_idx]
    y_delta_test = y_delta[test_idx]

    cls_metrics = json.loads(cls_metrics_path.read_text(encoding="utf-8"))
    reg_metrics = json.loads(reg_metrics_path.read_text(encoding="utf-8"))
    cls_thresholds = cls_metrics.get("best_thresholds", [0.5, 0.5, 0.5])
    reg_count_threshold = float(
        args.reg_count_threshold if args.reg_count_threshold > 0 else reg_metrics.get("best_count_threshold", 40.0)
    )

    PhysicsInformedGNNClassifier = load_module_attr(
        "gnn_cls_model",
        cls_dir / "model" / "model.py",
        "PhysicsInformedGNNClassifier",
    )
    PhysicsInformedGNNRegressor = load_module_attr(
        "gnn_reg_model",
        reg_dir / "model" / "model.py",
        "PhysicsInformedGNNRegressor",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cls_model = PhysicsInformedGNNClassifier(
        in_dim=4,
        hidden_dim=int(cls_metrics.get("hidden_dim", 128)),
        proj_dim=int(cls_metrics.get("proj_dim", 128)),
        out_dim=3,
        heads=int(cls_metrics.get("gat_heads", 4)),
        excitation_chunk_size=int(cls_metrics.get("excitation_chunk_size", 4)),
        dropout=float(cls_metrics.get("dropout", 0.1)),
    ).to(device)
    cls_model.load_state_dict(torch.load(cls_model_path, map_location=device))
    cls_model.eval()

    reg_model = PhysicsInformedGNNRegressor(
        in_dim=4,
        hidden_dim=int(reg_metrics.get("hidden_dim", 128)),
        edge_hidden=int(reg_metrics.get("edge_hidden", 128)),
        out_dim=112,
        heads=int(reg_metrics.get("gat_heads", 4)),
        excitation_chunk_size=int(reg_metrics.get("excitation_chunk_size", 4)),
        dropout=float(reg_metrics.get("dropout", 0.1)),
        max_abs=float(reg_metrics.get("max_abs", 300.0)),
    ).to(device)
    reg_model.load_state_dict(torch.load(reg_model_path, map_location=device))
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

    edges = build_resistor_edges()
    adjacency = build_edge_adjacency(edges)
    final_delta_all = np.zeros_like(reg_pred_all, dtype=np.float32)
    final_counts = np.zeros(len(test_idx), dtype=np.int64)
    reg_count_preds = np.zeros(len(test_idx), dtype=np.int64)
    count_rules = []
    correct_id_total = 0
    true_id_total = int(y_change_test.sum())

    for i in range(len(test_idx)):
        k_cls = int(pred_counts[i])
        reg_abs = np.abs(reg_pred_all[i])
        sorted_ids = np.argsort(-reg_abs)
        k_pred = max(0, min(k_cls, int(args.max_k)))
        k_reg = predict_count_from_reg_abs(reg_abs, reg_count_threshold, max_k=args.max_k)
        rule = "cls_only"
        if args.enable_reg_arbitration:
            k_pred, k_reg, rule = arbitrate_count_with_reg(
                k_cls,
                reg_abs,
                reg_mask_all[i],
                sorted_ids.astype(int).tolist(),
                reg_count_threshold,
                max_k=args.max_k,
                dynamic_up_prob=args.reg_dynamic_up_prob,
                dynamic_down_prob=args.reg_dynamic_down_prob,
            )
        final_counts[i] = int(k_pred)
        reg_count_preds[i] = int(k_reg)
        count_rules.append(rule)
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
                protect_high_prob=args.near_miss_protect_prob,
            )
        if k_pred > 0:
            final_delta_all[i, topk_ids] = reg_pred_all[i, topk_ids]

        true_ids = np.where(y_change_test[i] > 0.5)[0].astype(int).tolist()
        pred_ids = np.where(np.abs(final_delta_all[i]) > 0)[0].astype(int).tolist()
        correct_id_total += len(set(true_ids).intersection(pred_ids))

    num_accuracy = float((final_counts == y_cls_test).mean())
    cm = confusion(final_counts, y_cls_test, num_classes=4)
    f1 = macro_f1(cm)
    id_recall = correct_id_total / max(true_id_total, 1)
    mse = float(np.mean((final_delta_all - y_delta_test) ** 2))
    s_num = num_accuracy * 100.0
    s_f1 = f1 * 100.0
    s_id = id_recall * 100.0
    s_mse = 100.0 * max(0.0, 1.0 - mse / args.mse_max)
    cmei = 0.40 * s_id + 0.30 * s_mse + 0.15 * s_f1 + 0.15 * s_num

    metrics = {
        "version": "cmei_v2",
        "dataset_tag": dataset_tag,
        "data_path": str(data_path),
        "models": {
            "gnn_cls": str(cls_model_path),
            "gnn_reg": str(reg_model_path),
        },
        "score_weights": {"S_id": 0.40, "S_mse": 0.30, "S_F1": 0.15, "S_num": 0.15},
        "cls_thresholds": cls_thresholds,
        "near_miss": {
            "enabled": bool(args.enable_near_miss),
            "ratio": args.near_miss_ratio,
            "protect_high_prob": args.near_miss_protect_prob,
        },
        "reg_arbitration": {
            "enabled": bool(args.enable_reg_arbitration),
            "reg_count_threshold": reg_count_threshold,
            "dynamic_up_prob": args.reg_dynamic_up_prob,
            "dynamic_down_prob": args.reg_dynamic_down_prob,
            "max_k": args.max_k,
        },
        "noise": {
            "std": args.noise_std,
            "seed": args.noise_seed,
        },
        "num_accuracy": num_accuracy,
        "macro_f1": f1,
        "id_recall": id_recall,
        "mse_all_edges": mse,
        "scores": {
            "S_num": s_num,
            "S_F1": s_f1,
            "S_id": s_id,
            "S_mse": s_mse,
            "CMEI": cmei,
        },
        "confusion_matrix": cm.tolist(),
        "mse_max": args.mse_max,
        "test_size": len(test_idx),
    }
    (out_dir / "cmei_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    rng = random.Random(args.seed)
    detail_candidates = [i for i in range(len(test_idx)) if int(y_cls_test[i]) in (2, 3)]
    if not detail_candidates:
        detail_candidates = list(range(len(test_idx)))
    rng.shuffle(detail_candidates)
    detail_selected = detail_candidates[: min(args.detail_samples, len(detail_candidates))]

    detail_rows = []
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
                "pred_k": int(final_counts[local_idx]),
                "cls_k": int(pred_counts[local_idx]),
                "reg_k": int(reg_count_preds[local_idx]),
                "count_rule": count_rules[local_idx],
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
    (out_dir / "detail_samples.json").write_text(json.dumps(detail_rows, indent=2), encoding="utf-8")

    report_lines = [
        "# GNN CMEI Report",
        "",
        f"- Dataset: `{dataset_tag}`",
        f"- Count accuracy: `{num_accuracy:.4f}`",
        f"- Macro-F1: `{f1:.4f}`",
        f"- ID recall: `{id_recall:.4f}`",
        f"- All-edge MSE: `{mse:.4f}`",
        f"- Sub-scores: `S_num={s_num:.2f}`, `S_F1={s_f1:.2f}`, `S_id={s_id:.2f}`, `S_mse={s_mse:.2f}`",
        f"- Noise: `std={args.noise_std:.4f}`, `seed={args.noise_seed}`",
        f"- Near-Miss: `enabled={args.enable_near_miss}`, `ratio={args.near_miss_ratio:.2f}`, `protect_high_prob={args.near_miss_protect_prob:.2f}`",
        f"- REG Arbitration: `enabled={args.enable_reg_arbitration}`, `reg_count_threshold={reg_count_threshold:.2f}`, `up_prob={args.reg_dynamic_up_prob:.2f}`, `down_prob={args.reg_dynamic_down_prob:.2f}`",
        f"- **CMEI={cmei:.2f}**",
        "",
        "## Confusion Matrix",
        "```text",
        np.array2string(cm),
        "```",
        "",
        "## Detail Samples",
    ]
    for row in detail_rows:
        report_lines.append(
            f"- Sample `{row['sample_index']}`: true_k=`{row['true_k']}` pred_k=`{row['pred_k']}` "
            f"true_ids=`{row['true_ids']}` pred_ids=`{row['pred_ids']}`"
        )
    (out_dir / "cmei_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("[CMEI Sub-scores]")
    print(f"S_num={s_num:.2f}  (weight=0.15)")
    print(f"S_F1={s_f1:.2f}  (weight=0.15)")
    print(f"S_id={s_id:.2f}  (weight=0.40)")
    print(f"S_mse={s_mse:.2f}  (weight=0.30)")
    print("[CMEI Total]")
    print(f"CMEI = 0.40*S_id + 0.30*S_mse + 0.15*S_F1 + 0.15*S_num = {cmei:.2f}")
    print("[Raw Metrics]")
    print(f"num_accuracy={num_accuracy:.4f} | macro_f1={f1:.4f} | id_recall={id_recall:.4f} | mse_all_edges={mse:.4f}")
    print(f"noise_std={args.noise_std:.4f} | noise_seed={args.noise_seed}")
    print(
        "[REG Arbitration] "
        f"enabled={args.enable_reg_arbitration} | reg_count_threshold={reg_count_threshold:.2f} | "
        f"up_prob={args.reg_dynamic_up_prob:.2f} | down_prob={args.reg_dynamic_down_prob:.2f}"
    )
    print("Confusion Matrix (rows=true, cols=pred):")
    print(np.array2string(cm))
    print("[Detail Samples]")
    for row in detail_rows:
        print(
            f"Sample index={row['sample_index']} | pred_k={row['pred_k']} | true_k={row['true_k']} | "
            f"cls_k={row['cls_k']} | reg_k={row['reg_k']} | rule={row['count_rule']}"
        )
        print(f"  pred_ids={row['pred_ids']}")
        print(f"  true_ids={row['true_ids']}")
        print(f"  pred_deltas={[round(v, 2) for v in row['pred_deltas']]}")
        print(f"  true_deltas={[round(v, 2) for v in row['true_deltas']]}")
    print(f"Saved summary to {out_dir}")


if __name__ == "__main__":
    main()
