from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
VENDOR_DIR = WORKSPACE_ROOT / ".vendor_torchpy311"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import prepend_vendor_dir

prepend_vendor_dir(VENDOR_DIR, required_version=(3, 11))

import numpy as np
import torch

from models.modelo1_mlp1 import Modelo1MLP1Regressor
from project_common import (
    apply_standardization,
    build_examples,
    compute_fixedk_metrics,
    dump_json,
    load_json,
    load_split_from_meta,
    run_dir_name,
    write_predictions_csv,
)


def default_out_dir(meta_path: Path) -> Path:
    meta = load_json(meta_path)
    return PROJECT_ROOT / "outputs_modelo1_mlp1" / run_dir_name(
        grid_size=int(meta["topology"]["grid_size"]),
        k=int(meta["k"]),
        port_count=int(meta["topology"]["port_count"]),
        excitation_count=len(meta["excitations"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for square_scale_study modelo1_mlp1.")
    parser.add_argument("--meta-path", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--examples-limit", type=int, default=12)
    parser.add_argument("--id-pass-threshold", type=float, default=0.98)
    parser.add_argument("--value-pass-threshold", type=float, default=0.90)
    parser.add_argument("--hidden-dim", type=int, default=1536)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--ff-multiplier", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.02)
    parser.add_argument("--max-abs", type=float, default=250.0)
    return parser.parse_args()


def infer_from_meta(args: argparse.Namespace) -> dict:
    meta_path = Path(args.meta_path).resolve()
    meta = load_json(meta_path)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else default_out_dir(meta_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_split, y_split, sample_ids, topology, _ = load_split_from_meta(meta_path, args.split)
    standardization = np.load(out_dir / "standardization.npz")
    boundary_nodes = standardization["boundary_nodes"].astype(np.int64)
    x_split = apply_standardization(x_split, boundary_nodes, standardization["mean"], standardization["std"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Modelo1MLP1Regressor(
        input_dim=int(np.prod(x_split.shape[1:])),
        num_resistors=topology.num_resistors,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        ff_multiplier=args.ff_multiplier,
        dropout=args.dropout,
        max_abs=args.max_abs,
    ).to(device)
    model.load_state_dict(torch.load(out_dir / "model_last.pt", map_location=device))
    model.eval()

    score_rows = []
    value_rows = []
    with torch.no_grad():
        for start in range(0, len(x_split), args.batch_size):
            xb = torch.from_numpy(x_split[start:start + args.batch_size]).float().to(device)
            score_logits, value_pred = model(xb)
            score_rows.append(score_logits.cpu().numpy())
            value_rows.append(value_pred.cpu().numpy())
    score_logits = np.concatenate(score_rows, axis=0) if score_rows else np.zeros_like(y_split)
    value_pred = np.concatenate(value_rows, axis=0) if value_rows else np.zeros_like(y_split)

    metrics = compute_fixedk_metrics(
        value_pred,
        y_split,
        k=int(meta["k"]),
        ranking_scores=score_logits,
        id_pass_threshold=args.id_pass_threshold,
        value_pass_threshold=args.value_pass_threshold,
    )
    prediction_rows = metrics.pop("per_sample")
    write_predictions_csv(out_dir / "predictions.csv", sample_ids, {**metrics, "per_sample": prediction_rows})
    examples = build_examples({**metrics, "per_sample": prediction_rows}, sample_ids, limit=args.examples_limit)
    dump_json(out_dir / "examples.json", {"split": args.split, "examples": examples})

    payload = {
        "meta_path": str(meta_path),
        "dataset_stem": meta["dataset_stem"],
        "out_dir": str(out_dir),
        "split": args.split,
        "grid_size": int(meta["topology"]["grid_size"]),
        "num_nodes": int(meta["topology"]["num_nodes"]),
        "num_resistors": int(meta["topology"]["num_resistors"]),
        "port_count": int(meta["topology"]["port_count"]),
        **metrics,
    }
    dump_json(out_dir / "inference_metrics.json", payload)
    print(
        f"[{args.split}] id_exact_rate={payload['id_exact_rate']:.4f} "
        f"value_accuracy={payload['value_accuracy']:.4f} "
        f"mae_changed={payload['mae_changed']:.4f} pass={payload['pass_flag']}"
    )
    return payload


def main() -> None:
    args = parse_args()
    infer_from_meta(args)


if __name__ == "__main__":
    main()
