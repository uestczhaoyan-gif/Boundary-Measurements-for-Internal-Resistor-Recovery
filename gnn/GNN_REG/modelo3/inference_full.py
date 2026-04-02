import argparse
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

_VENDOR_DIR = Path(__file__).resolve().parents[3] / ".vendor_torchpy311"
if _VENDOR_DIR.exists() and str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

import numpy as np
import torch
from scipy.special import expit

from model.model import PhysicsInformedGNNRegressor


BASE_R = 1000.0
DEFAULT_MAIN_DATA_PATH = "../../../data/training_data64Nodes_2.csv"


def sanitize_dataset_tag(raw_tag):
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", raw_tag.strip())
    safe = safe.strip("._-")
    return safe or "dataset"


def resolve_input_data_path(raw_path, script_dir):
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    project_root = script_dir.parents[2]
    candidates = [path, script_dir / path, project_root / path]
    for candidate in candidates:
        if candidate.exists():
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


def select_focus_indices(test_idx, true_counts, num_samples, seed, focus_high_change=True, min_true_change=2):
    rng = random.Random(seed)
    pool = list(test_idx)
    if not focus_high_change:
        return rng.sample(pool, k=min(num_samples, len(pool)))

    high = [idx for idx in pool if int(true_counts[idx]) >= min_true_change]
    low = [idx for idx in pool if int(true_counts[idx]) < min_true_change]
    rng.shuffle(high)
    rng.shuffle(low)
    selected = high[: min(num_samples, len(high))]
    if len(selected) < num_samples:
        selected.extend(low[: num_samples - len(selected)])
    return selected


def load_module_attr(module_name, file_path, attr_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, attr_name)


def standardize_graph_voltage(x, mean, std, ext_nodes):
    x_std = x.copy()
    x_std[:, :, ext_nodes, 2] = (x_std[:, :, ext_nodes, 2] - mean) / std
    return x_std.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Combined inference for GNN CLS+REG using top-K from REG, where K comes from CLS.")
    parser.add_argument("--data-path", default=DEFAULT_MAIN_DATA_PATH)
    parser.add_argument("--dataset-tag", default="", help="数据集标签；默认取 data-path 文件名。")
    parser.add_argument("--reg-cache-path", default="./cache_dataset_reg_graphattn.npz")
    parser.add_argument("--reg-model-path", default="")
    parser.add_argument("--reg-metrics-path", default="")
    parser.add_argument("--reg-standardization", default="")
    parser.add_argument("--cls-dir", default="../../GNN_CLS/modelo3")
    parser.add_argument("--cls-cache-path", default="")
    parser.add_argument("--cls-model-path", default="")
    parser.add_argument("--cls-metrics-path", default="")
    parser.add_argument("--cls-standardization", default="")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260325)
    parser.set_defaults(focus_high_change=True)
    parser.add_argument("--focus-high-change", dest="focus_high_change", action="store_true")
    parser.add_argument("--no-focus-high-change", dest="focus_high_change", action="store_false")
    parser.add_argument("--min-true-change", type=int, default=2)
    parser.add_argument("--reg-hidden-dim", type=int, default=128)
    parser.add_argument("--reg-edge-hidden", type=int, default=128)
    parser.add_argument("--reg-gat-heads", type=int, default=4)
    parser.add_argument("--reg-excitation-chunk-size", type=int, default=4)
    parser.add_argument("--reg-dropout", type=float, default=0.1)
    parser.add_argument("--reg-max-abs", type=float, default=300.0)
    parser.add_argument("--cls-hidden-dim", type=int, default=128)
    parser.add_argument("--cls-proj-dim", type=int, default=128)
    parser.add_argument("--cls-gat-heads", type=int, default=4)
    parser.add_argument("--cls-excitation-chunk-size", type=int, default=4)
    parser.add_argument("--cls-dropout", type=float, default=0.1)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_path = resolve_input_data_path(args.data_path, script_dir)
    dataset_tag = sanitize_dataset_tag(args.dataset_tag or data_path.stem)

    reg_cache = resolve_default_artifact_path(script_dir / "cache", Path(args.reg_cache_path).name, dataset_tag, data_path) if args.reg_cache_path == "./cache_dataset_reg_graphattn.npz" else Path(args.reg_cache_path)
    reg_out_dir = resolve_default_artifact_path(script_dir / "outputs", "model_last.pt", dataset_tag, data_path).parent
    reg_model = reg_out_dir / "model_last.pt" if not args.reg_model_path else Path(args.reg_model_path)
    reg_metrics = reg_out_dir / "metrics.json" if not args.reg_metrics_path else Path(args.reg_metrics_path)
    reg_std = reg_out_dir / "standardization.npz" if not args.reg_standardization else Path(args.reg_standardization)

    cls_dir = resolve_input_data_path(args.cls_dir, script_dir)
    cls_cache = resolve_default_artifact_path(cls_dir / "cache", "cache_dataset_cls_graphattn.npz", dataset_tag, data_path) if not args.cls_cache_path else Path(args.cls_cache_path)
    cls_out_dir = resolve_default_artifact_path(cls_dir / "outputs", "model_last.pt", dataset_tag, data_path).parent
    cls_model = cls_out_dir / "model_last.pt" if not args.cls_model_path else Path(args.cls_model_path)
    cls_metrics = cls_out_dir / "metrics.json" if not args.cls_metrics_path else Path(args.cls_metrics_path)
    cls_std = cls_out_dir / "standardization.npz" if not args.cls_standardization else Path(args.cls_standardization)

    print(
        "[Runtime Full] "
        f"dataset_tag={dataset_tag} | data_path={data_path} | "
        f"reg_model={reg_model} | cls_model={cls_model}"
    )

    reg_npz = np.load(reg_cache)
    x_reg = reg_npz["x"].astype(np.float32)
    y_change = reg_npz["y_change"].astype(np.float32)
    y_delta = reg_npz["y_delta"].astype(np.float32)
    reg_norm = np.load(reg_std)
    x_reg = standardize_graph_voltage(x_reg, reg_norm["mean"], reg_norm["std"], reg_norm["ext_nodes"].astype(np.int64))

    cls_npz = np.load(cls_cache)
    x_cls = cls_npz["x"].astype(np.float32)
    y_cls = cls_npz["y"].astype(np.int64)
    cls_norm = np.load(cls_std)
    x_cls = standardize_graph_voltage(x_cls, cls_norm["mean"], cls_norm["std"], cls_norm["ext_nodes"].astype(np.int64))

    _, _, test_idx = split_indices(len(x_reg), args.seed)
    sample_idx = select_focus_indices(
        test_idx,
        y_change.sum(axis=1),
        args.num_samples,
        args.seed,
        focus_high_change=args.focus_high_change,
        min_true_change=args.min_true_change,
    )

    cls_metrics_json = json.loads(Path(cls_metrics).read_text(encoding="utf-8")) if Path(cls_metrics).exists() else {}
    thresholds = cls_metrics_json.get("best_thresholds", [0.5, 0.5, 0.5])

    PhysicsInformedGNNClassifier = load_module_attr(
        "gnn_cls_modelo3_model",
        cls_dir / "model" / "model.py",
        "PhysicsInformedGNNClassifier",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reg_model_obj = PhysicsInformedGNNRegressor(
        in_dim=4,
        hidden_dim=args.reg_hidden_dim,
        edge_hidden=args.reg_edge_hidden,
        out_dim=112,
        heads=args.reg_gat_heads,
        excitation_chunk_size=args.reg_excitation_chunk_size,
        dropout=args.reg_dropout,
        max_abs=args.reg_max_abs,
    ).to(device)
    reg_model_obj.load_state_dict(torch.load(reg_model, map_location=device))
    reg_model_obj.eval()

    cls_model_obj = PhysicsInformedGNNClassifier(
        in_dim=4,
        hidden_dim=args.cls_hidden_dim,
        proj_dim=args.cls_proj_dim,
        out_dim=3,
        heads=args.cls_gat_heads,
        excitation_chunk_size=args.cls_excitation_chunk_size,
        dropout=args.cls_dropout,
    ).to(device)
    cls_model_obj.load_state_dict(torch.load(cls_model, map_location=device))
    cls_model_obj.eval()

    results = []
    for idx in sample_idx:
        xb_reg = torch.from_numpy(x_reg[idx:idx + 1]).to(device)
        xb_cls = torch.from_numpy(x_cls[idx:idx + 1]).to(device)
        with torch.no_grad():
            reg_pred, reg_aux = reg_model_obj(xb_reg, return_aux=True)
            cls_logits, cls_aux = cls_model_obj(xb_cls, return_aux=True)

        reg_pred = reg_pred.cpu().numpy().flatten()
        cls_logits = cls_logits.cpu().numpy()[0]
        cls_probs = expit(cls_logits)
        k_pred = int((cls_probs > np.array(thresholds, dtype=np.float32)).sum())
        true_count = int(y_change[idx].sum())
        if int(y_cls[idx]) != true_count:
            raise RuntimeError(f"CLS/REG label mismatch at index {idx}: cls={int(y_cls[idx])}, reg={true_count}")

        topk_ids = np.argsort(-np.abs(reg_pred))[:k_pred].astype(int).tolist()
        final_delta = np.zeros_like(reg_pred)
        if k_pred > 0:
            final_delta[topk_ids] = reg_pred[topk_ids]

        true_delta = y_delta[idx].astype(np.float32)
        true_ids = np.where(y_change[idx] > 0.5)[0].astype(int).tolist()
        final_ids = np.where(np.abs(final_delta) > 0)[0].astype(int).tolist()

        results.append(
            {
                "index": int(idx),
                "true_change_count": true_count,
                "cls_pred_count": k_pred,
                "cls_thresholds": thresholds,
                "cls_logits": cls_logits.tolist(),
                "cls_probs": cls_probs.tolist(),
                "reg_top_abs_ids": np.argsort(-np.abs(reg_pred))[: min(8, len(reg_pred))].astype(int).tolist(),
                "final_change_ids": final_ids,
                "final_change_deltas": [float(final_delta[i]) for i in final_ids],
                "true_change_ids": true_ids,
                "true_change_deltas": [float(true_delta[i]) for i in true_ids],
                "pred_mask_prob": reg_aux["mask_prob"].cpu().numpy().flatten().tolist(),
                "pred_gate_value": reg_aux["value"].cpu().numpy().flatten().tolist(),
                "final_deltas": final_delta.tolist(),
                "true_deltas": true_delta.tolist(),
                "final_resistances": (BASE_R + final_delta).tolist(),
                "true_resistances": (BASE_R + true_delta).tolist(),
                "abs_error_resistance": np.abs(final_delta - true_delta).tolist(),
                "contrast_feat": cls_aux["contrast_feat"].cpu().numpy()[0].tolist(),
            }
        )
        print(f"Sample index={idx} | cls_pred_k={k_pred} | true_k={true_count} | final_ids={final_ids} | true_ids={true_ids}")

    out_path = reg_out_dir / "inference_full_samples.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
