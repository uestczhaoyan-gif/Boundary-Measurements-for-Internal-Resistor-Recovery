import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from model.model import CNN2DHMultiTask


def split_indices(n, seed):
    rng = random.Random(seed)
    ids = list(range(n))
    rng.shuffle(ids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def main():
    parser = argparse.ArgumentParser(description="Inference for 64Nodes CNN2D-MLP full h-multitask.")
    parser.add_argument("--cache-path", default="./cache_dataset_full_v2.npz")
    parser.add_argument("--model-path", default="./outputs/model_last.pt")
    parser.add_argument("--metrics-path", default="./outputs/metrics.json")
    parser.add_argument("--standardization", default="./outputs/standardization.npz")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260319)
    parser.add_argument("--per-class", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.cache_path == "./cache_dataset_full_v2.npz":
        args.cache_path = str(script_dir / "cache_dataset_full_v2.npz")
    if args.model_path == "./outputs/model_last.pt":
        args.model_path = str(script_dir / "outputs" / "model_last.pt")
    if args.metrics_path == "./outputs/metrics.json":
        args.metrics_path = str(script_dir / "outputs" / "metrics.json")
    if args.standardization == "./outputs/standardization.npz":
        args.standardization = str(script_dir / "outputs" / "standardization.npz")

    d = np.load(args.cache_path)
    x = d["x"].astype(np.float32)
    y_change = d["y_change"].astype(np.float32)
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
    model = CNN2DHMultiTask(in_ch=x.shape[1], out_dim=112).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    results = []
    for idx in sample_idx:
        xb = torch.from_numpy(x[idx:idx + 1]).to(device)
        with torch.no_grad():
            logits, pred = model(xb)
            probs = torch.sigmoid(logits / temp).cpu().numpy().flatten()
            pred_delta = pred.cpu().numpy().flatten()

        pred_count_head = int((probs > coral_thrs).sum())
        pred_count_reg = int(np.clip((np.abs(pred_delta) > reg_count_thr).sum(), 0, 3))
        true_count = int(y_change[idx].sum())
        top_idx = np.argsort(-np.abs(pred_delta))[:5].tolist()
        results.append(
            {
                "index": int(idx),
                "true_count": true_count,
                "pred_count_head": pred_count_head,
                "pred_count_reg": pred_count_reg,
                "coral_probs": probs.tolist(),
                "top5_abs_delta_idx": top_idx,
                "top5_abs_delta_val": [float(pred_delta[i]) for i in top_idx],
                "pred_deltas": pred_delta.tolist(),
            }
        )
        print(f"\nSample index={idx}")
        print(f"True={true_count} | Pred(count head)={pred_count_head} | Pred(reg-thr)={pred_count_reg}")

    out_path = Path(args.model_path).parent / "inference_samples.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
