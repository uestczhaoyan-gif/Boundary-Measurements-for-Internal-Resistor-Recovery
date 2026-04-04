import argparse
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

from model.model import PhysicsInformedGNNRegressor


BASE_R = 1000.0
DEFAULT_MAIN_DATA_PATH = "../../../data/training_data64Nodes_2.csv"


def ensure_cli_option_has_value(argv, option_name):
    for idx, token in enumerate(argv[1:], start=1):
        if token != option_name:
            continue
        next_idx = idx + 1
        if next_idx >= len(argv):
            raise SystemExit(
                f"{Path(argv[0]).name}: {option_name} is missing a value. "
                "If you used TAG, make sure it has been set before running this command."
            )
        next_token = argv[next_idx]
        if next_token == "" or next_token.startswith("-"):
            raise SystemExit(
                f"{Path(argv[0]).name}: {option_name} did not receive a usable value. "
                "If you used ${TAG}, it likely expanded to an empty string."
            )


def validate_dataset_tag_arg(raw_tag):
    if raw_tag and any(ch in raw_tag for ch in "$ {}"):
        raise SystemExit(
            f"{Path(sys.argv[0]).name}: --dataset-tag looks like an unexpanded placeholder: {raw_tag!r}. "
            "Please replace it with the real run tag."
        )


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
            if tag != requested_tag:
                print(f"[Fallback] dataset_tag='{requested_tag}' not found, using '{tag}' -> {candidate}")
            return candidate
    legacy = root_dir / filename
    if legacy.exists():
        print(f"[Fallback] using legacy root artifact -> {legacy}")
        return legacy
    return root_dir / requested_tag / filename


def resolve_inference_runtime_paths(args, script_dir, default_cache_path):
    data_path = resolve_input_data_path(args.data_path, script_dir)
    validate_dataset_tag_arg(args.dataset_tag)
    dataset_tag = sanitize_dataset_tag(args.dataset_tag or data_path.stem)

    if args.cache_path == default_cache_path:
        cache_path = resolve_default_artifact_path(script_dir / "cache", Path(default_cache_path).name, dataset_tag, data_path)
    else:
        cache_path = Path(args.cache_path)

    outputs_root = script_dir / "outputs"
    outputs_dir = outputs_root
    if args.dataset_subdir:
        outputs_dir = resolve_default_artifact_path(outputs_root, "model_last.pt", dataset_tag, data_path).parent

    model_path = outputs_dir / "model_last.pt" if args.model_path == "./outputs/model_last.pt" else Path(args.model_path)
    metrics_path = outputs_dir / "metrics.json" if args.metrics_path == "./outputs/metrics.json" else Path(args.metrics_path)
    standardization = outputs_dir / "standardization.npz" if args.standardization == "./outputs/standardization.npz" else Path(args.standardization)

    args.data_path = str(data_path)
    args.dataset_tag = dataset_tag
    args.cache_path = str(cache_path)
    args.model_path = str(model_path)
    args.metrics_path = str(metrics_path)
    args.standardization = str(standardization)


def standardize_graph_voltage(x, mean, std, ext_nodes):
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


def main():
    ensure_cli_option_has_value(sys.argv, "--dataset-tag")
    parser = argparse.ArgumentParser(description="Inference for 64Nodes physics-informed GNN regression o4a2.")
    parser.add_argument("--data-path", default=DEFAULT_MAIN_DATA_PATH)
    parser.add_argument("--dataset-tag", default="", help="数据集标签；默认取 data-path 文件名。")
    parser.add_argument("--cache-path", default="./cache_dataset_reg_graphattn.npz")
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
    parser.add_argument("--per-class", action="store_true")
    parser.add_argument("--noise-std", type=float, default=0.0, help="Standardized voltage noise std applied to test set only.")
    parser.add_argument("--noise-seed", type=int, default=20260331)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    resolve_inference_runtime_paths(args, script_dir, "./cache_dataset_reg_graphattn.npz")
    print(
        "[Runtime] "
        f"dataset_tag={args.dataset_tag} | data_path={Path(args.data_path)} | cache_path={Path(args.cache_path)} | "
        f"model_path={Path(args.model_path)} | metrics_path={Path(args.metrics_path)} | std_path={Path(args.standardization)}"
    )

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
    x_test = inject_voltage_noise(x[test_idx], ext_nodes, args.noise_std, args.noise_seed)
    y_change_test = y_change[test_idx]
    y_delta_test = y_delta[test_idx]
    idx_to_local = {int(idx): pos for pos, idx in enumerate(test_idx)}
    rng = random.Random(args.seed)
    if args.per_class:
        buckets = {i: [] for i in range(4)}
        for idx in test_idx:
            local_idx = idx_to_local[int(idx)]
            buckets[min(3, int(y_change_test[local_idx].sum()))].append(idx)
        sample_idx = []
        for k in range(4):
            if buckets[k]:
                sample_idx.append(rng.choice(buckets[k]))
        rest = [i for i in test_idx if i not in sample_idx]
        rng.shuffle(rest)
        sample_idx.extend(rest[: max(0, args.num_samples - len(sample_idx))])
    else:
        sample_idx = select_focus_indices(
            test_idx,
            y_change.sum(axis=1),
            args.num_samples,
            args.seed,
            focus_high_change=args.focus_high_change,
            min_true_change=args.min_true_change,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicsInformedGNNRegressor(
        in_dim=4,
        hidden_dim=args.hidden_dim,
        edge_hidden=args.edge_hidden,
        out_dim=112,
        heads=args.gat_heads,
        excitation_chunk_size=args.excitation_chunk_size,
        dropout=args.dropout,
        max_abs=args.max_abs,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    pred_all = []
    mask_prob_all = []
    value_all = []
    with torch.no_grad():
        for start in range(0, len(x_test), 64):
            xb = torch.from_numpy(x_test[start:start + 64]).to(device)
            pred_delta, aux = model(xb, return_aux=True)
            pred_all.append(pred_delta.cpu().numpy())
            mask_prob_all.append(aux["mask_prob"].cpu().numpy())
            value_all.append(aux["value"].cpu().numpy())
    pred_all = np.concatenate(pred_all, axis=0)
    mask_prob_all = np.concatenate(mask_prob_all, axis=0)
    value_all = np.concatenate(value_all, axis=0)

    mae_all = float(np.abs(pred_all - y_delta_test).mean())
    changed_mask = y_change_test > 0.5
    mae_changed = float(np.abs(pred_all[changed_mask] - y_delta_test[changed_mask]).mean()) if changed_mask.any() else 0.0
    pred_counts_all = (np.abs(pred_all) > args.count_threshold).sum(axis=1).astype(np.int64)
    true_counts_all = changed_mask.sum(axis=1).astype(np.int64)
    avg_gt = float(pred_counts_all.mean())
    avg_mask_prob = float(mask_prob_all.mean())
    cm = np.zeros((4, 4), dtype=np.int64)
    for pred_k, true_k in zip(pred_counts_all, true_counts_all):
        cm[min(3, int(true_k)), min(3, int(pred_k))] += 1
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1s.append(0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall))
    count_macro_f1 = float(np.mean(f1s))

    eval_metrics = {
        "dataset_tag": args.dataset_tag,
        "noise_std": args.noise_std,
        "noise_seed": args.noise_seed,
        "count_threshold": args.count_threshold,
        "mae_all": mae_all,
        "mae_changed": mae_changed,
        "avg_abs_gt_threshold": avg_gt,
        "avg_mask_prob": avg_mask_prob,
        "count_macro_f1": count_macro_f1,
        "confusion_matrix": cm.tolist(),
    }
    metrics_name = "noise_eval.json" if args.noise_std > 0 else "inference_eval.json"
    (Path(args.model_path).parent / metrics_name).write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")

    results = []
    for idx in sample_idx:
        local_idx = idx_to_local[int(idx)]
        pred_delta = pred_all[local_idx].astype(np.float32)
        mask_prob = mask_prob_all[local_idx].astype(np.float32)
        gate_value = value_all[local_idx].astype(np.float32)
        true_delta = y_delta_test[local_idx].astype(np.float32)

        pred_r = BASE_R + pred_delta
        true_r = BASE_R + true_delta
        abs_err = np.abs(pred_r - true_r)
        pred_count = int((np.abs(pred_delta) > args.count_threshold).sum())
        true_count = int(y_change_test[local_idx].sum())
        pred_change_ids = np.where(np.abs(pred_delta) > args.count_threshold)[0].astype(int).tolist()
        true_change_ids = np.where(y_change_test[local_idx] > 0.5)[0].astype(int).tolist()
        pred_change_deltas = [float(pred_delta[i]) for i in pred_change_ids]
        true_change_deltas = [float(true_delta[i]) for i in true_change_ids]

        results.append(
            {
                "index": int(idx),
                "pred_gt_threshold": pred_count,
                "true_change_count": true_count,
                "pred_change_ids": pred_change_ids,
                "pred_change_deltas": pred_change_deltas,
                "true_change_ids": true_change_ids,
                "true_change_deltas": true_change_deltas,
                "pred_mask_prob": mask_prob.tolist(),
                "pred_gate_value": gate_value.tolist(),
                "pred_deltas": pred_delta.tolist(),
                "true_deltas": true_delta.tolist(),
                "pred_resistances": pred_r.tolist(),
                "true_resistances": true_r.tolist(),
                "abs_error_resistance": abs_err.tolist(),
            }
        )
        print(f"\nSample index={idx}")
        print(f"Pred |dR|>{args.count_threshold:.1f}: {pred_count} | True changes: {true_count}")
        print(f"Pred ids: {pred_change_ids}")
        print(f"True ids: {true_change_ids}")

    out_path = Path(args.model_path).parent / "inference_samples.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nEvaluation Metrics:")
    print(f"noise_std={args.noise_std:.4f} | mae_all={mae_all:.4f} | mae_changed={mae_changed:.4f}")
    print(f"count_threshold={args.count_threshold:.1f} | count_macro_f1={count_macro_f1:.4f}")
    print(f"avg(|dR|>{args.count_threshold:.1f})={avg_gt:.2f} | avg(mask_prob)={avg_mask_prob:.4f}")
    print("Derived Count Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
