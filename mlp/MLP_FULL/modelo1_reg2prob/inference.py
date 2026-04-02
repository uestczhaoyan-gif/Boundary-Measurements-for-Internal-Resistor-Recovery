import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from model.model import MLPReg2Prob


def split_indices(n, seed):
    rng = random.Random(seed)
    ids = list(range(n))
    rng.shuffle(ids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def main():
    parser = argparse.ArgumentParser(description="Inference for 64Nodes MLP full reg2prob.")
    parser.add_argument("--cache-path", default="./cache_dataset_full.npz")
    parser.add_argument("--model-path", default="./outputs/model_last.pt")
    parser.add_argument("--standardization", default="./outputs/standardization.npz")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260319)
    parser.add_argument("--prob-threshold", type=float, default=50.0)
    parser.add_argument("--prob-tau", type=float, default=12.0)
    parser.add_argument("--per-class", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.cache_path == "./cache_dataset_full.npz":
        args.cache_path = str(script_dir / "cache_dataset_full.npz")
    if args.model_path == "./outputs/model_last.pt":
        args.model_path = str(script_dir / "outputs" / "model_last.pt")
    if args.standardization == "./outputs/standardization.npz":
        args.standardization = str(script_dir / "outputs" / "standardization.npz")

    d = np.load(args.cache_path)
    x = d["x"].astype(np.float32)
    y_change = d["y_change"].astype(np.float32)
    std = np.load(args.standardization)
    x = ((x - std["mean"]) / std["std"]).astype(np.float32)

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
    model = MLPReg2Prob(in_dim=x.shape[1], out_dim=112).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    results = []
    for idx in sample_idx:
        xb = torch.from_numpy(x[idx:idx + 1]).to(device)
        with torch.no_grad():
            pred = model(xb).cpu().numpy().flatten()
        p = 1.0 / (1.0 + np.exp(-(np.abs(pred) - args.prob_threshold) / args.prob_tau))
        k_pred = int(np.clip(np.rint(p.sum()), 0, 3))
        k_true = int(np.clip(y_change[idx].sum(), 0, 3))
        results.append(
            {
                "index": int(idx),
                "pred_count": k_pred,
                "true_count": k_true,
                "pred_prob": p.tolist(),
                "pred_deltas": pred.tolist(),
            }
        )
        print(f"\nSample index={idx}")
        print(f"Count pred={k_pred} | Count true={k_true}")

    out_path = Path(args.model_path).parent / "inference_samples.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
