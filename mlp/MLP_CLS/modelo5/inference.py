import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from model.model import MLPClassifierMultiHead


NUM_CLASSES = 4
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
    parser = argparse.ArgumentParser(description="Inference for 64Nodes MLP classification modelo5.")
    parser.add_argument("--data-path", default=DEFAULT_MAIN_DATA_PATH)
    parser.add_argument("--dataset-tag", default="", help="数据集标签；默认取 data-path 文件名。")
    parser.add_argument("--cache-path", default="./cache_dataset_cls.npz")
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
    resolve_inference_runtime_paths(args, script_dir, "./cache_dataset_cls.npz")

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
    use_aux_refiner = bool(metrics.get("use_aux_refiner", True))
    aux_thr = float(metrics.get("aux23_threshold", 0.5)) if metrics.get("aux23_threshold", None) is not None else 0.5
    temp = float(metrics["temperature"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifierMultiHead(in_dim=x.shape[1], out_dim=NUM_CLASSES - 1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    for idx in sample_idx:
        xb = torch.from_numpy(x[idx:idx + 1]).to(device)
        with torch.no_grad():
            main_logits, aux_logit = model(xb)
            main_probs = torch.sigmoid(main_logits / temp).cpu().numpy().flatten()
            aux_prob = torch.sigmoid(aux_logit).cpu().numpy().item()

        pred_main = int((main_probs > thrs).sum())
        if use_aux_refiner and pred_main in (2, 3):
            pred = 3 if aux_prob > aux_thr else 2
        else:
            pred = pred_main

        print(f"\nSample index={idx}")
        print(f"True label: {int(y[idx])} | Pred label: {pred}")
        print(f"CORAL probs: {np.round(main_probs, 4).tolist()} | aux_2v3_prob={aux_prob:.4f} | use_aux_refiner={use_aux_refiner}")


if __name__ == "__main__":
    main()
