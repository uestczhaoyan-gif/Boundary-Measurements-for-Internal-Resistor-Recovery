import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from model.model import MLPClassifier


NUM_CLASSES = 4


def split_indices(n, seed):
    rng = random.Random(seed)
    ids = list(range(n))
    rng.shuffle(ids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def main():
    parser = argparse.ArgumentParser(description="Inference for 64Nodes MLP classification.")
    parser.add_argument("--cache-path", default="./cache_dataset_cls.npz")
    parser.add_argument("--model-path", default="./outputs/model_last.pt")
    parser.add_argument("--metrics-path", default="./outputs/metrics.json")
    parser.add_argument("--standardization", default="./outputs/standardization.npz")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260319)
    parser.add_argument("--per-class", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.cache_path == "./cache_dataset_cls.npz":
        args.cache_path = str(script_dir / "cache_dataset_cls.npz")
    if args.model_path == "./outputs/model_last.pt":
        args.model_path = str(script_dir / "outputs" / "model_last.pt")
    if args.metrics_path == "./outputs/metrics.json":
        args.metrics_path = str(script_dir / "outputs" / "metrics.json")
    if args.standardization == "./outputs/standardization.npz":
        args.standardization = str(script_dir / "outputs" / "standardization.npz")

    d = np.load(args.cache_path)
    x = d["x"].astype(np.float32)
    y = d["y"].astype(np.int64)
    std = np.load(args.standardization)
    x = ((x - std["mean"]) / std["std"]).astype(np.float32)

    _, _, test_idx = split_indices(len(y), args.seed)
    rng = random.Random(args.seed)
    if args.per_class:
        buckets = {i: [] for i in range(NUM_CLASSES)}
        for idx in test_idx:
            buckets[int(y[idx])].append(idx)
        pick = []
        for k in range(NUM_CLASSES):
            if buckets[k]:
                pick.append(rng.choice(buckets[k]))
        remain = [i for i in test_idx if i not in pick]
        rng.shuffle(remain)
        sample_idx = pick + remain[: max(0, args.num_samples - len(pick))]
    else:
        sample_idx = rng.sample(test_idx, k=min(args.num_samples, len(test_idx)))

    metrics = json.loads(Path(args.metrics_path).read_text(encoding="utf-8"))
    thrs = np.array(metrics["thresholds"], dtype=np.float32)
    temp = float(metrics["temperature"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(in_dim=x.shape[1], out_dim=NUM_CLASSES - 1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    for idx in sample_idx:
        xb = torch.from_numpy(x[idx:idx + 1]).to(device)
        with torch.no_grad():
            logits = model(xb)
            probs = torch.sigmoid(logits / temp).cpu().numpy().flatten()
        pred = int((probs > thrs).sum())
        print(f"\nSample index={idx}")
        print(f"True label: {int(y[idx])} | Pred label: {pred}")
        print(f"CORAL probs: {np.round(probs, 4).tolist()}")


if __name__ == "__main__":
    main()
