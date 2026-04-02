import argparse
import json
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
from train import (
    build_dataset,
    confusion,
    macro_f1,
    resolve_input_data_path,
    split_indices,
    standardize_graph_voltage,
    weighted_score,
)


DEFAULT_MAIN_DATA_PATH = "../../../data/training_data64Nodes_2.csv"
DEFAULT_CACHE_NAME = "cache_dataset_cls_graphattn.npz"


def sanitize_dataset_tag(raw_tag):
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", raw_tag.strip())
    safe = safe.strip("._-")
    return safe or "dataset"


def resolve_default_artifact_path(root_dir, filename, requested_tag, data_path):
    root_dir = Path(root_dir)
    tags = []
    for tag in [sanitize_dataset_tag(requested_tag), sanitize_dataset_tag(data_path.stem)]:
        if tag and tag not in tags:
            tags.append(tag)
    for tag in tags:
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


def resolve_output_path(raw_path, script_dir):
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    project_root = script_dir.parents[2]
    return (project_root / path).resolve()


def build_runtime_paths(args, script_dir):
    data_path = resolve_input_data_path(args.data_path, script_dir)
    dataset_tag = sanitize_dataset_tag(args.dataset_tag or data_path.stem)
    artifacts_root = resolve_input_data_path(args.artifacts_root, script_dir)

    if args.cache_path:
        cache_path = resolve_input_data_path(args.cache_path, script_dir)
    else:
        cache_path = resolve_default_artifact_path(artifacts_root / "cache", DEFAULT_CACHE_NAME, dataset_tag, data_path)

    if args.model_path:
        model_path = resolve_input_data_path(args.model_path, script_dir)
    else:
        model_path = resolve_default_artifact_path(artifacts_root / "outputs", "model_last.pt", dataset_tag, data_path)

    if args.metrics_path:
        metrics_path = resolve_input_data_path(args.metrics_path, script_dir)
    else:
        metrics_path = model_path.parent / "metrics.json"

    if args.standardization:
        standardization = resolve_input_data_path(args.standardization, script_dir)
    else:
        standardization = model_path.parent / "standardization.npz"

    if args.out_report:
        out_report = resolve_output_path(args.out_report, script_dir)
    else:
        out_report = model_path.parent / "two_stage_threshold_report.json"

    if args.out_metrics_path:
        out_metrics_path = resolve_output_path(args.out_metrics_path, script_dir)
    else:
        out_metrics_path = model_path.parent / "metrics.two_stage_thresholds.json"

    return {
        "data_path": data_path,
        "dataset_tag": dataset_tag,
        "artifacts_root": artifacts_root,
        "cache_path": cache_path,
        "model_path": model_path,
        "metrics_path": metrics_path,
        "standardization": standardization,
        "out_report": out_report,
        "out_metrics_path": out_metrics_path,
    }


def thresholds_to_pred(probs, thresholds):
    thr = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    return (probs > thr).sum(axis=1).astype(np.int64)


def evaluate_thresholds(probs, true_labels, thresholds, penalty_32, bonus_r3, bonus_r2):
    pred = thresholds_to_pred(probs, thresholds)
    cm = confusion(pred, true_labels)
    return {
        "thresholds": [float(t) for t in thresholds],
        "macro_f1": float(macro_f1(cm)),
        "weighted_score": float(weighted_score(cm, penalty_32=penalty_32, bonus_r3=bonus_r3, bonus_r2=bonus_r2)),
        "confusion_matrix": cm.tolist(),
    }


def global_search(val_probs, val_true, step, penalty_32, bonus_r3, bonus_r2):
    grid = np.arange(0.05, 0.951, step)
    best = None
    for t1 in grid:
        m1 = val_probs[:, 0] > t1
        for t2 in grid[grid >= t1]:
            m2 = val_probs[:, 1] > t2
            for t3 in grid[grid >= t2]:
                pred = m1.astype(np.int64) + m2.astype(np.int64) + (val_probs[:, 2] > t3).astype(np.int64)
                cm = confusion(pred, val_true)
                score = weighted_score(cm, penalty_32=penalty_32, bonus_r3=bonus_r3, bonus_r2=bonus_r2)
                if best is None or score > best["weighted_score"]:
                    best = {
                        "thresholds": [float(t1), float(t2), float(t3)],
                        "macro_f1": float(macro_f1(cm)),
                        "weighted_score": float(score),
                        "confusion_matrix": cm.tolist(),
                    }
    return best


def local_grid(center, radius, step, low=0.05, high=0.95):
    start = max(low, center - radius)
    end = min(high, center + radius)
    values = np.arange(start, end + 0.5 * step, step, dtype=np.float64)
    values = np.concatenate([values, np.array([center], dtype=np.float64)])
    values = np.clip(values, low, high)
    values = np.unique(np.round(values, 6))
    return values


def local_refine_search(val_probs, val_true, coarse_thresholds, radius, step, penalty_32, bonus_r3, bonus_r2):
    g1 = local_grid(coarse_thresholds[0], radius, step)
    g2 = local_grid(coarse_thresholds[1], radius, step)
    g3 = local_grid(coarse_thresholds[2], radius, step)
    best = None
    for t1 in g1:
        m1 = val_probs[:, 0] > t1
        valid_g2 = g2[g2 >= t1]
        for t2 in valid_g2:
            m2 = val_probs[:, 1] > t2
            valid_g3 = g3[g3 >= t2]
            for t3 in valid_g3:
                pred = m1.astype(np.int64) + m2.astype(np.int64) + (val_probs[:, 2] > t3).astype(np.int64)
                cm = confusion(pred, val_true)
                score = weighted_score(cm, penalty_32=penalty_32, bonus_r3=bonus_r3, bonus_r2=bonus_r2)
                if best is None or score > best["weighted_score"]:
                    best = {
                        "thresholds": [float(t1), float(t2), float(t3)],
                        "macro_f1": float(macro_f1(cm)),
                        "weighted_score": float(score),
                        "confusion_matrix": cm.tolist(),
                    }
    return best


def collect_probs(model, x, batch_size, device):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            logits = model(xb).cpu().numpy()
            outputs.append(expit(logits))
    return np.concatenate(outputs, axis=0)


def maybe_load_metrics(metrics_path):
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return {}


def maybe_load_or_compute_standardization(x, tr_idx, ext_nodes, standardization_path):
    if standardization_path.exists():
        d = np.load(standardization_path)
        return d["mean"].astype(np.float32), d["std"].astype(np.float32)
    mean = x[tr_idx][:, :, ext_nodes, 2].mean(axis=0, keepdims=True)
    std = x[tr_idx][:, :, ext_nodes, 2].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage threshold refinement for modelo3-compatible CLS checkpoints.")
    parser.add_argument("--data-path", default=DEFAULT_MAIN_DATA_PATH)
    parser.add_argument("--dataset-tag", default="", help="Data tag used to resolve cache/outputs.")
    parser.add_argument("--artifacts-root", default=".", help="Classifier root directory containing cache/ and outputs/.")
    parser.add_argument("--cache-path", default="", help="Optional explicit cache path.")
    parser.add_argument("--model-path", default="", help="Optional explicit model_last.pt path.")
    parser.add_argument("--metrics-path", default="", help="Optional explicit metrics.json path.")
    parser.add_argument("--standardization", default="", help="Optional explicit standardization.npz path.")
    parser.add_argument("--out-report", default="", help="Optional output JSON report path.")
    parser.add_argument("--out-metrics-path", default="", help="Optional refined metrics JSON path.")
    parser.add_argument("--write-back-metrics", action="store_true", help="Overwrite metrics.json after creating a backup.")
    parser.add_argument("--seed", type=int, default=20260325, help="Must match the original train/val/test split seed.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--coarse-step", type=float, default=0.01)
    parser.add_argument("--fine-step", type=float, default=0.002)
    parser.add_argument("--fine-radius", type=float, default=0.03)
    parser.add_argument("--penalty-32", type=float, default=0.12)
    parser.add_argument("--bonus-r3", type=float, default=0.06)
    parser.add_argument("--bonus-r2", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=-1)
    parser.add_argument("--proj-dim", type=int, default=-1)
    parser.add_argument("--gat-heads", type=int, default=-1)
    parser.add_argument("--excitation-chunk-size", type=int, default=-1)
    parser.add_argument("--dropout", type=float, default=-1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    rt = build_runtime_paths(args, script_dir)
    metrics = maybe_load_metrics(rt["metrics_path"])

    hidden_dim = int(metrics.get("hidden_dim", 128) if args.hidden_dim <= 0 else args.hidden_dim)
    proj_dim = int(metrics.get("proj_dim", 128) if args.proj_dim <= 0 else args.proj_dim)
    gat_heads = int(metrics.get("gat_heads", 4) if args.gat_heads <= 0 else args.gat_heads)
    excitation_chunk_size = int(metrics.get("excitation_chunk_size", 4) if args.excitation_chunk_size <= 0 else args.excitation_chunk_size)
    dropout = float(metrics.get("dropout", 0.1) if args.dropout < 0 else args.dropout)

    print(
        "[Two-Stage Threshold Search] "
        f"dataset_tag={rt['dataset_tag']} | data_path={rt['data_path']} | cache_path={rt['cache_path']} | "
        f"model_path={rt['model_path']} | metrics_path={rt['metrics_path']} | std_path={rt['standardization']}"
    )

    if rt["cache_path"].exists():
        d = np.load(rt["cache_path"])
        x = d["x"].astype(np.float32)
        y = d["y"].astype(np.int64)
        ext_nodes = d["ext_nodes"].astype(np.int64)
    else:
        x, y, ext_nodes = build_dataset(rt["data_path"], rt["cache_path"])

    tr_idx, va_idx, te_idx = split_indices(len(x), args.seed)
    mean, std = maybe_load_or_compute_standardization(x, tr_idx, ext_nodes, rt["standardization"])
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    x_std = standardize_graph_voltage(x, mean, std, ext_nodes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicsInformedGNNClassifier(
        in_dim=4,
        hidden_dim=hidden_dim,
        proj_dim=proj_dim,
        out_dim=3,
        heads=gat_heads,
        excitation_chunk_size=excitation_chunk_size,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(torch.load(rt["model_path"], map_location=device))
    model.eval()

    val_probs = collect_probs(model, x_std[va_idx], args.batch_size, device)
    test_probs = collect_probs(model, x_std[te_idx], args.batch_size, device)
    val_true = y[va_idx]
    test_true = y[te_idx]

    original_thresholds = metrics.get("best_thresholds", [0.5, 0.5, 0.5])
    original_val = evaluate_thresholds(val_probs, val_true, original_thresholds, args.penalty_32, args.bonus_r3, args.bonus_r2)
    original_test = evaluate_thresholds(test_probs, test_true, original_thresholds, args.penalty_32, args.bonus_r3, args.bonus_r2)

    coarse = global_search(val_probs, val_true, args.coarse_step, args.penalty_32, args.bonus_r3, args.bonus_r2)
    fine = local_refine_search(
        val_probs,
        val_true,
        coarse["thresholds"],
        args.fine_radius,
        args.fine_step,
        args.penalty_32,
        args.bonus_r3,
        args.bonus_r2,
    )
    refined_test = evaluate_thresholds(test_probs, test_true, fine["thresholds"], args.penalty_32, args.bonus_r3, args.bonus_r2)

    report = {
        "dataset_tag": rt["dataset_tag"],
        "data_path": str(rt["data_path"]),
        "cache_path": str(rt["cache_path"]),
        "model_path": str(rt["model_path"]),
        "metrics_path": str(rt["metrics_path"]),
        "standardization": str(rt["standardization"]),
        "seed": args.seed,
        "coarse_step": args.coarse_step,
        "fine_step": args.fine_step,
        "fine_radius": args.fine_radius,
        "penalty_32": args.penalty_32,
        "bonus_r3": args.bonus_r3,
        "bonus_r2": args.bonus_r2,
        "original": {
            "val": original_val,
            "test": original_test,
        },
        "coarse_search": coarse,
        "fine_search": fine,
        "refined_test": refined_test,
    }
    rt["out_report"].parent.mkdir(parents=True, exist_ok=True)
    rt["out_report"].write_text(json.dumps(report, indent=2), encoding="utf-8")

    refined_metrics = dict(metrics)
    refined_metrics["best_thresholds"] = fine["thresholds"]
    refined_metrics["two_stage_threshold_search"] = {
        "report_path": str(rt["out_report"]),
        "coarse_step": args.coarse_step,
        "fine_step": args.fine_step,
        "fine_radius": args.fine_radius,
        "original_val_macro_f1": original_val["macro_f1"],
        "refined_val_macro_f1": fine["macro_f1"],
        "original_test_macro_f1": original_test["macro_f1"],
        "refined_test_macro_f1": refined_test["macro_f1"],
    }
    rt["out_metrics_path"].parent.mkdir(parents=True, exist_ok=True)
    rt["out_metrics_path"].write_text(json.dumps(refined_metrics, indent=2), encoding="utf-8")

    if args.write_back_metrics:
        backup_path = rt["metrics_path"].with_name("metrics.before_two_stage_threshold.json")
        if rt["metrics_path"].exists() and not backup_path.exists():
            backup_path.write_text(rt["metrics_path"].read_text(encoding="utf-8"), encoding="utf-8")
        rt["metrics_path"].write_text(json.dumps(refined_metrics, indent=2), encoding="utf-8")

    print(f"Original thresholds: {original_thresholds}")
    print(f"Coarse best thresholds: {coarse['thresholds']}")
    print(f"Fine best thresholds: {fine['thresholds']}")
    print(
        "Val macro_f1: "
        f"{original_val['macro_f1']:.4f} -> {fine['macro_f1']:.4f} | "
        "Test macro_f1: "
        f"{original_test['macro_f1']:.4f} -> {refined_test['macro_f1']:.4f}"
    )
    print(f"Saved report to {rt['out_report']}")
    print(f"Saved refined metrics to {rt['out_metrics_path']}")
    if args.write_back_metrics:
        print(f"Updated metrics.json in place: {rt['metrics_path']}")


if __name__ == "__main__":
    main()
