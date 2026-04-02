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

from model.model import PhysicsInformedGNNClassifier


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
    parser = argparse.ArgumentParser(description="Inference for 64Nodes physics-informed GNN classifier modelo3.")
    parser.add_argument("--data-path", default=DEFAULT_MAIN_DATA_PATH)
    parser.add_argument("--dataset-tag", default="", help="数据集标签；默认取 data-path 文件名。")
    parser.add_argument("--cache-path", default="./cache_dataset_cls_graphattn.npz")
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
    parser.add_argument("--noise-std", type=float, default=0.0, help="Standardized voltage noise std applied to test set only.")
    parser.add_argument("--noise-seed", type=int, default=20260331)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    resolve_inference_runtime_paths(args, script_dir, "./cache_dataset_cls_graphattn.npz")
    print(
        "[Runtime] "
        f"dataset_tag={args.dataset_tag} | data_path={Path(args.data_path)} | cache_path={Path(args.cache_path)} | "
        f"model_path={Path(args.model_path)} | metrics_path={Path(args.metrics_path)} | std_path={Path(args.standardization)}"
    )

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
    x_test = inject_voltage_noise(x[test_idx], ext_nodes, args.noise_std, args.noise_seed)
    y_test = y[test_idx]
    idx_to_local = {int(idx): pos for pos, idx in enumerate(test_idx)}
    sample_idx = select_focus_indices(
        test_idx,
        y,
        args.num_samples,
        args.seed,
        focus_high_change=args.focus_high_change,
        min_true_change=args.min_true_change,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicsInformedGNNClassifier(
        in_dim=4,
        hidden_dim=args.hidden_dim,
        proj_dim=args.proj_dim,
        out_dim=3,
        heads=args.gat_heads,
        excitation_chunk_size=args.excitation_chunk_size,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    logits_all = []
    with torch.no_grad():
        for start in range(0, len(x_test), 64):
            xb = torch.from_numpy(x_test[start:start + 64]).to(device)
            logits = model(xb).cpu().numpy()
            logits_all.append(logits)
    logits_all = np.concatenate(logits_all, axis=0)
    probs_all = expit(logits_all)
    threshold_array = np.array(thresholds, dtype=np.float32)
    pred_all = (probs_all > threshold_array.reshape(1, -1)).sum(axis=1).astype(np.int64)
    cm = np.zeros((4, 4), dtype=np.int64)
    for pred, true in zip(pred_all, y_test):
        cm[int(true), int(pred)] += 1
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1s.append(0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall))
    macro_f1 = float(np.mean(f1s))

    eval_metrics = {
        "dataset_tag": args.dataset_tag,
        "noise_std": args.noise_std,
        "noise_seed": args.noise_seed,
        "thresholds": thresholds,
        "test_macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
    }
    metrics_name = "noise_eval.json" if args.noise_std > 0 else "inference_eval.json"
    (Path(args.model_path).parent / metrics_name).write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")

    results = []
    with torch.no_grad():
        for idx in sample_idx:
            local_idx = idx_to_local[int(idx)]
            xb = torch.from_numpy(x_test[local_idx:local_idx + 1]).to(device)
            logits, aux = model(xb, return_aux=True)
            logits_np = logits.cpu().numpy()[0]
            probs = expit(logits_np)
            pred = int((probs > threshold_array).sum())
            results.append(
                {
                    "index": int(idx),
                    "true_label": int(y_test[local_idx]),
                    "pred_label": pred,
                    "raw_logits": logits_np.tolist(),
                    "coral_probs": probs.tolist(),
                    "thresholds": thresholds,
                    "contrast_feat": aux["contrast_feat"].cpu().numpy()[0].tolist(),
                }
            )
            print(f"Sample index={idx} | pred={pred} | true={int(y_test[local_idx])}")

    out_path = Path(args.model_path).parent / "inference_samples.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nEvaluation Metrics:")
    print(f"noise_std={args.noise_std:.4f} | test_macro_f1={macro_f1:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
