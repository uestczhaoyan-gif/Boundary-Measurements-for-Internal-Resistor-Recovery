from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.modelv1.inference import infer_from_meta
from models.modelv1.train import train_from_meta
from summarize_scale_sweep import summarize_outputs
from generate_square_fixedk_data import generate_fixedk_dataset_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep N and K to estimate K_max for square grids.")
    parser.add_argument("--n-values", nargs="+", type=int, default=[3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument("--max-k", type=int, default=12)
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--outputs-root", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--figure-path", default=str(PROJECT_ROOT / "Figure" / "scale_kmax_summary.png"))
    parser.add_argument("--train-size", type=int, default=12000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--seed-base", type=int, default=20260410)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--force-regenerate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    outputs_root = Path(args.outputs_root).resolve()
    outputs_root.mkdir(parents=True, exist_ok=True)

    for n in args.n_values:
        consecutive_failures = 0
        for k in range(1, args.max_k + 1):
            stem = f"square_N{n}x{n}_K{k}"
            meta_path = data_root / f"N{n}x{n}" / f"{stem}_meta.json"
            if args.force_regenerate or not meta_path.exists():
                meta_path = generate_fixedk_dataset_bundle(
                    grid_size=n,
                    k=k,
                    output_root=data_root,
                    train_size=args.train_size,
                    val_size=args.val_size,
                    test_size=args.test_size,
                    seed=args.seed_base + n * 100 + k,
                )
            run_out_dir = outputs_root / f"N{n}x{n}_K{k}"
            train_args = argparse.Namespace(
                meta_path=str(meta_path),
                out_dir=str(run_out_dir),
                epochs=args.epochs,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                lr=3e-4,
                weight_decay=1e-4,
                dropout=0.1,
                hidden_dim=128,
                edge_hidden=128,
                gat_heads=4,
                excitation_chunk_size=4,
                max_abs=250.0,
                smooth_l1_beta=25.0,
                grad_clip=1.0,
                patience=args.patience,
                seed=args.seed_base + n * 100 + k,
            )
            infer_args = argparse.Namespace(
                meta_path=str(meta_path),
                out_dir=str(run_out_dir),
                split="test",
                batch_size=args.eval_batch_size,
                examples_limit=12,
                id_pass_threshold=0.98,
                value_pass_threshold=0.90,
                hidden_dim=128,
                edge_hidden=128,
                gat_heads=4,
                excitation_chunk_size=4,
                dropout=0.1,
                max_abs=250.0,
            )
            train_from_meta(train_args)
            infer_metrics = infer_from_meta(infer_args)

            if infer_metrics["pass_flag"]:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    break

    summarize_outputs(outputs_root, Path(args.figure_path).resolve())


if __name__ == "__main__":
    main()
