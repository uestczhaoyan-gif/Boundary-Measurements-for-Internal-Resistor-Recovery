import argparse
import importlib.util
import json
import random
import re
from pathlib import Path

from vendor_bootstrap import bootstrap_vendor_paths, format_dependency_import_error

_BOOTSTRAP_RESULTS = bootstrap_vendor_paths(Path(__file__).resolve().parents[3])

try:
    import numpy as np
except Exception as exc:
    raise ImportError(
        format_dependency_import_error("numpy", exc, _BOOTSTRAP_RESULTS)
    ) from None

try:
    import torch
except Exception as exc:
    raise ImportError(
        format_dependency_import_error("torch", exc, _BOOTSTRAP_RESULTS)
    ) from None


BASE_R = 1000.0
DEFAULT_MAIN_DATA_PATH = "../../../data/training_data64Nodes_2.csv"
DEFAULT_SOURCE_MODEL_DIR = "../modelo3"


def sanitize_dataset_tag(raw_tag):
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", raw_tag.strip())
    safe = safe.strip("._-")
    return safe or "dataset"


def resolve_input_path(raw_path, script_dir):
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


def resolve_runtime_paths(args, script_dir, default_cache_path):
    data_path = resolve_input_path(args.data_path, script_dir)
    dataset_tag = sanitize_dataset_tag(args.dataset_tag or data_path.stem)
    source_model_dir = resolve_input_path(args.source_model_dir, script_dir)

    if args.cache_path == default_cache_path:
        cache_path = resolve_default_artifact_path(
            source_model_dir / "cache",
            Path(default_cache_path).name,
            dataset_tag,
            data_path,
        )
    else:
        cache_path = Path(args.cache_path)

    source_outputs_root = source_model_dir / "outputs"
    if args.dataset_subdir:
        source_outputs_dir = resolve_default_artifact_path(
            source_outputs_root,
            "model_last.pt",
            dataset_tag,
            data_path,
        ).parent
    else:
        source_outputs_dir = source_outputs_root

    model_path = source_outputs_dir / "model_last.pt" if args.model_path == "./outputs/model_last.pt" else Path(args.model_path)
    metrics_path = source_outputs_dir / "metrics.json" if args.metrics_path == "./outputs/metrics.json" else Path(args.metrics_path)
    standardization = source_outputs_dir / "standardization.npz" if args.standardization == "./outputs/standardization.npz" else Path(args.standardization)

    out_base = script_dir / "outputs" if args.out_dir == "./outputs" else Path(args.out_dir)
    out_dir = out_base / dataset_tag if args.dataset_subdir else out_base

    args.data_path = str(data_path)
    args.dataset_tag = dataset_tag
    args.source_model_dir = str(source_model_dir)
    args.cache_path = str(cache_path)
    args.model_path = str(model_path)
    args.metrics_path = str(metrics_path)
    args.standardization = str(standardization)
    args.out_dir = str(out_dir)


def standardize_graph_voltage(x, mean, std, ext_nodes):
    x_std = x.copy()
    x_std[:, :, ext_nodes, 2] = (x_std[:, :, ext_nodes, 2] - mean) / std
    return x_std.astype(np.float32)


def split_indices(n, seed):
    rng = random.Random(seed)
    ids = list(range(n))
    rng.shuffle(ids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def select_focus_local_indices(true_counts, num_samples, seed, focus_high_change=True, min_true_change=2):
    rng = random.Random(seed)
    pool = list(range(len(true_counts)))
    if not focus_high_change:
        rng.shuffle(pool)
        return pool[: min(num_samples, len(pool))]

    high = [idx for idx, count in enumerate(true_counts) if int(count) >= min_true_change]
    low = [idx for idx, count in enumerate(true_counts) if int(count) < min_true_change]
    rng.shuffle(high)
    rng.shuffle(low)
    selected = high[: min(num_samples, len(high))]
    if len(selected) < num_samples:
        selected.extend(low[: num_samples - len(selected)])
    return selected


def parse_candidate_sizes(raw_sizes):
    values = []
    for part in str(raw_sizes).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            continue
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("candidate_sizes must contain at least one positive integer.")
    return tuple(values)


def topm_ids(delta, m):
    m = max(1, int(m))
    return np.argsort(-np.abs(delta))[: min(m, len(delta))].astype(int).tolist()


def build_candidate_metrics(pred_all, y_change, candidate_sizes):
    metrics = {}
    for m in candidate_sizes:
        hits_all = []
        hits_changed_only = []
        hits_by_k = {1: [], 2: [], 3: []}
        for idx in range(len(pred_all)):
            true_ids = np.where(y_change[idx] > 0.5)[0].astype(int).tolist()
            hit = set(true_ids).issubset(set(topm_ids(pred_all[idx], m)))
            hit_value = 1.0 if hit else 0.0
            hits_all.append(hit_value)
            true_k = min(3, int(y_change[idx].sum()))
            if true_k > 0:
                hits_changed_only.append(hit_value)
                hits_by_k[true_k].append(hit_value)
        metrics[f"top{m}_candidate_cover"] = float(np.mean(hits_all)) if hits_all else 0.0
        metrics[f"top{m}_candidate_cover_changed_only"] = (
            float(np.mean(hits_changed_only)) if hits_changed_only else 0.0
        )
        for true_k in (1, 2, 3):
            values = hits_by_k[true_k]
            metrics[f"top{m}_candidate_cover_k{true_k}"] = float(np.mean(values)) if values else 0.0
    return metrics


def load_module_attr(module_name, file_path, attr_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, attr_name)


def load_source_metrics(metrics_path):
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(
        description="Candidate-set inference for 64Nodes physics-informed GNN regression modelo3b."
    )
    parser.add_argument("--data-path", default=DEFAULT_MAIN_DATA_PATH)
    parser.add_argument("--dataset-tag", default="", help="数据集标签；默认取 data-path 文件名。")
    parser.add_argument("--source-model-dir", default=DEFAULT_SOURCE_MODEL_DIR)
    parser.add_argument("--cache-path", default="./cache_dataset_reg_graphattn.npz")
    parser.add_argument("--model-path", default="./outputs/model_last.pt")
    parser.add_argument("--metrics-path", default="./outputs/metrics.json")
    parser.add_argument("--standardization", default="./outputs/standardization.npz")
    parser.add_argument("--out-dir", default="./outputs")
    parser.set_defaults(dataset_subdir=True)
    parser.add_argument("--dataset-subdir", dest="dataset_subdir", action="store_true")
    parser.add_argument("--no-dataset-subdir", dest="dataset_subdir", action="store_false")
    parser.add_argument("--candidate-sizes", default="3,4,5")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=20260325)
    parser.add_argument("--seed", type=int, default=20260325)
    parser.set_defaults(focus_high_change=True)
    parser.add_argument("--focus-high-change", dest="focus_high_change", action="store_true")
    parser.add_argument("--no-focus-high-change", dest="focus_high_change", action="store_false")
    parser.add_argument("--min-true-change", type=int, default=2)
    parser.add_argument("--count-threshold", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--edge-hidden", type=int, default=None)
    parser.add_argument("--gat-heads", type=int, default=None)
    parser.add_argument("--excitation-chunk-size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-abs", type=float, default=300.0)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    resolve_runtime_paths(args, script_dir, "./cache_dataset_reg_graphattn.npz")
    candidate_sizes = parse_candidate_sizes(args.candidate_sizes)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[Runtime modelo3b] "
        f"dataset_tag={args.dataset_tag} | data_path={Path(args.data_path)} | "
        f"source_model_dir={Path(args.source_model_dir)} | cache_path={Path(args.cache_path)} | "
        f"model_path={Path(args.model_path)} | metrics_path={Path(args.metrics_path)} | "
        f"std_path={Path(args.standardization)} | out_dir={out_dir}"
    )

    source_metrics = load_source_metrics(args.metrics_path)
    if args.count_threshold is None:
        args.count_threshold = float(source_metrics.get("best_count_threshold", 45.0))
    if args.hidden_dim is None:
        args.hidden_dim = int(source_metrics.get("hidden_dim", 128))
    if args.edge_hidden is None:
        args.edge_hidden = int(source_metrics.get("edge_hidden", 128))
    if args.gat_heads is None:
        args.gat_heads = int(source_metrics.get("gat_heads", 4))
    if args.excitation_chunk_size is None:
        args.excitation_chunk_size = int(source_metrics.get("excitation_chunk_size", 4))

    d = np.load(args.cache_path)
    x = d["x"].astype(np.float32)
    y_change = d["y_change"].astype(np.float32)
    y_delta = d["y_delta"].astype(np.float32)
    std = np.load(args.standardization)
    x = standardize_graph_voltage(x, std["mean"], std["std"], std["ext_nodes"].astype(np.int64))

    _, _, test_idx = split_indices(len(x), args.split_seed)
    x_test = x[test_idx]
    y_change_test = y_change[test_idx]
    y_delta_test = y_delta[test_idx]
    test_true_counts = y_change_test.sum(axis=1)

    model_cls = load_module_attr(
        "gnn_reg_modelo3_source",
        Path(args.source_model_dir) / "model" / "model.py",
        "PhysicsInformedGNNRegressor",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_cls(
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
    mask_all = []
    with torch.no_grad():
        for idx in range(len(x_test)):
            xb = torch.from_numpy(x_test[idx:idx + 1]).to(device)
            pred_delta, aux = model(xb, return_aux=True)
            pred_all.append(pred_delta.cpu().numpy()[0])
            mask_all.append(aux["mask_prob"].cpu().numpy()[0])
    pred_all = np.asarray(pred_all, dtype=np.float32)
    mask_all = np.asarray(mask_all, dtype=np.float32)

    candidate_metrics = build_candidate_metrics(pred_all, y_change_test, candidate_sizes)
    print("[Candidate Cover]")
    for candidate_size in candidate_sizes:
        print(
            f"top{candidate_size}_candidate_cover={candidate_metrics[f'top{candidate_size}_candidate_cover']:.4f}"
        )
        print(
            f"top{candidate_size}_candidate_cover_changed_only="
            f"{candidate_metrics[f'top{candidate_size}_candidate_cover_changed_only']:.4f}"
        )

    detail_selected = select_focus_local_indices(
        test_true_counts,
        args.num_samples,
        args.seed,
        focus_high_change=args.focus_high_change,
        min_true_change=args.min_true_change,
    )

    detail_rows = []
    for order, local_idx in enumerate(detail_selected, start=1):
        dataset_index = int(test_idx[local_idx])
        pred_delta = pred_all[local_idx]
        true_delta = y_delta_test[local_idx]
        true_change_ids = np.where(y_change_test[local_idx] > 0.5)[0].astype(int).tolist()
        pred_change_ids = np.where(np.abs(pred_delta) > args.count_threshold)[0].astype(int).tolist()

        row = {
            "order": int(order),
            "index": dataset_index,
            "true_change_count": int(y_change_test[local_idx].sum()),
            "pred_gt_threshold": int((np.abs(pred_delta) > args.count_threshold).sum()),
            "true_change_ids": true_change_ids,
            "pred_change_ids": pred_change_ids,
            "pred_change_deltas": [float(pred_delta[eid]) for eid in pred_change_ids],
            "true_change_deltas": [float(true_delta[eid]) for eid in true_change_ids],
            "pred_mask_prob": mask_all[local_idx].tolist(),
            "pred_deltas": pred_delta.tolist(),
            "true_deltas": true_delta.tolist(),
        }
        for candidate_size in candidate_sizes:
            ids = topm_ids(pred_delta, candidate_size)
            row[f"top{candidate_size}_ids"] = ids
            row[f"top{candidate_size}_hit"] = set(true_change_ids).issubset(set(ids))
        detail_rows.append(row)

        print(f"\nSample index={dataset_index}")
        print(f"Pred |dR|>{args.count_threshold:.1f}: {row['pred_gt_threshold']} | True changes: {row['true_change_count']}")
        print(f"Pred ids: {pred_change_ids}")
        print(f"True ids: {true_change_ids}")
        for candidate_size in candidate_sizes:
            print(
                f"Top{candidate_size} ids: {row[f'top{candidate_size}_ids']} | "
                f"hit={row[f'top{candidate_size}_hit']}"
            )

    metrics_payload = {
        "dataset_tag": args.dataset_tag,
        "data_path": str(args.data_path),
        "source_model_dir": str(args.source_model_dir),
        "model_path": str(args.model_path),
        "metrics_path": str(args.metrics_path),
        "standardization": str(args.standardization),
        "count_threshold": float(args.count_threshold),
        "candidate_sizes": list(candidate_sizes),
        "split_seed": int(args.split_seed),
        "seed": int(args.seed),
        "test_size": int(len(test_idx)),
    }
    metrics_payload.update(candidate_metrics)

    (out_dir / "candidate_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )
    (out_dir / "candidate_samples.json").write_text(
        json.dumps(detail_rows, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved: {out_dir / 'candidate_metrics.json'}")
    print(f"Saved: {out_dir / 'candidate_samples.json'}")


if __name__ == "__main__":
    main()
