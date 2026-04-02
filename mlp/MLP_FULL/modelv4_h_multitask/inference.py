import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from model.model import HMultiTaskMLP


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


def resolve_inference_runtime_paths(args, script_dir, default_cache_path):
    data_path = resolve_input_data_path(args.data_path, script_dir)

    dataset_tag = sanitize_dataset_tag(args.dataset_tag or data_path.stem)

    if args.cache_path == default_cache_path:
        cache_path = script_dir / "cache" / dataset_tag / Path(default_cache_path).name
    else:
        cache_path = Path(args.cache_path)

    outputs_root = script_dir / "outputs"
    outputs_dir = outputs_root / dataset_tag if args.dataset_subdir else outputs_root

    if args.model_path == "./outputs/model_last.pt":
        model_path = outputs_dir / "model_last.pt"
    else:
        model_path = Path(args.model_path)
    if args.metrics_path == "./outputs/metrics.json":
        metrics_path = outputs_dir / "metrics.json"
    else:
        metrics_path = Path(args.metrics_path)
    if args.standardization == "./outputs/standardization.npz":
        standardization_path = outputs_dir / "standardization.npz"
    else:
        standardization_path = Path(args.standardization)

    args.data_path = str(data_path)
    args.dataset_tag = dataset_tag
    args.cache_path = str(cache_path)
    args.model_path = str(model_path)
    args.metrics_path = str(metrics_path)
    args.standardization = str(standardization_path)


def split_indices(n, seed):
    rng = random.Random(seed)
    ids = list(range(n))
    rng.shuffle(ids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def main():
    parser = argparse.ArgumentParser(description="Inference for 64Nodes MLP full modelv4_h_multitask.")
    parser.add_argument("--data-path", default=DEFAULT_MAIN_DATA_PATH)
    parser.add_argument("--dataset-tag", default="", help="数据集标签；默认取 data-path 文件名。")
    parser.add_argument("--cache-path", default="./cache_dataset_full.npz")
    parser.add_argument("--model-path", default="./outputs/model_last.pt")
    parser.add_argument("--metrics-path", default="./outputs/metrics.json")
    parser.add_argument("--standardization", default="./outputs/standardization.npz")
    parser.set_defaults(dataset_subdir=True)
    parser.add_argument("--dataset-subdir", dest="dataset_subdir", action="store_true", help="按数据集标签查找 outputs/cache 子目录。")
    parser.add_argument("--no-dataset-subdir", dest="dataset_subdir", action="store_false", help="关闭按数据集标签查找子目录。")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260319)
    parser.add_argument("--per-class", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    resolve_inference_runtime_paths(args, script_dir, "./cache_dataset_full.npz")

    d = np.load(args.cache_path)
    x = d["x"].astype(np.float32)
    y_change = d["y_change"].astype(np.float32)
    y_delta = d["y_delta"].astype(np.float32)
    std = np.load(args.standardization)
    x = ((x - std["mean"]) / std["std"]).astype(np.float32)

    metrics = json.loads(Path(args.metrics_path).read_text(encoding="utf-8"))
    temp = float(metrics.get("temperature", 2.0))
    coral_thrs = np.array(metrics.get("coral_thresholds", [0.5, 0.5, 0.5]), dtype=np.float32)
    reg_count_thr = float(metrics.get("reg_count_threshold", 50.0))

    _, _, test_idx = split_indices(len(x), args.seed)
    rng = random.Random(args.seed)
    if args.per_class:
        buckets = {i: [] for i in range(4)}
        for idx in test_idx:
            k = int(y_change[idx].sum())
            buckets[min(3, k)].append(idx)
        sample_idx = []
        for k in range(4):
            if buckets[k]:
                sample_idx.append(rng.choice(buckets[k]))
        rest = [i for i in test_idx if i not in sample_idx]
        rng.shuffle(rest)
        sample_idx.extend(rest[: max(0, args.num_samples - len(sample_idx))])
    else:
        sample_idx = rng.sample(test_idx, k=min(args.num_samples, len(test_idx)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HMultiTaskMLP(in_dim=x.shape[1], out_dim=112).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    results = []
    for idx in sample_idx:
        xb = torch.from_numpy(x[idx:idx + 1]).to(device)
        with torch.no_grad():
            logits, pred = model(xb)
            probs = torch.sigmoid(logits / temp).cpu().numpy().flatten()
            pred_delta = pred.cpu().numpy().flatten()

        true_delta = y_delta[idx].astype(np.float32)
        true_change_ids = np.where(y_change[idx] > 0.5)[0].astype(int).tolist()
        pred_count_head = int((probs > coral_thrs).sum())
        pred_count_reg = int(np.clip((np.abs(pred_delta) > reg_count_thr).sum(), 0, 3))

        # IDs from count head: use top-k |dR| where k is predicted by count head.
        top_order = np.argsort(-np.abs(pred_delta))
        pred_change_ids_head = top_order[: max(pred_count_head, 0)].astype(int).tolist()
        # IDs from regression threshold:
        pred_change_ids_reg = np.where(np.abs(pred_delta) > reg_count_thr)[0].astype(int).tolist()

        pred_r = BASE_R + pred_delta
        true_r = BASE_R + true_delta
        abs_err_r = np.abs(pred_r - true_r)

        results.append(
            {
                "index": int(idx),
                "true_count": int(len(true_change_ids)),
                "true_change_ids": true_change_ids,
                "pred_count_head": pred_count_head,
                "pred_change_ids_head": pred_change_ids_head,
                "pred_count_reg_threshold": pred_count_reg,
                "pred_change_ids_reg_threshold": pred_change_ids_reg,
                "coral_probs": probs.tolist(),
                "pred_deltas": pred_delta.tolist(),
                "true_deltas": true_delta.tolist(),
                "pred_resistances": pred_r.tolist(),
                "true_resistances": true_r.tolist(),
                "abs_error_resistance": abs_err_r.tolist(),
            }
        )

        print(f"\nSample index={idx}")
        print(
            f"Pred count (head)={pred_count_head}, ids={pred_change_ids_head} | "
            f"Pred count (reg-thr)={pred_count_reg}, ids={pred_change_ids_reg}"
        )
        print(f"True count={len(true_change_ids)}, ids={true_change_ids}")
        print("All 112 resistor delta comparison: pred_deltas vs true_deltas (saved to JSON).")

    out_path = Path(args.model_path).parent / "inference_samples.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
