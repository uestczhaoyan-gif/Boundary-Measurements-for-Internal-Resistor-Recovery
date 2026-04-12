from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_common import DEFAULT_CHANGE_LIMIT, DEFAULT_CURRENT_A, generate_fixedk_dataset_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a grid of fixed-K square datasets.")
    parser.add_argument("--n-values", nargs="+", type=int, required=True)
    parser.add_argument("--k-values", nargs="+", type=int, required=True)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed-base", type=int, default=20260410)
    parser.add_argument("--current-a", type=float, default=DEFAULT_CURRENT_A)
    parser.add_argument("--change-limit", type=float, default=DEFAULT_CHANGE_LIMIT)
    parser.add_argument("--float-decimals", type=int, default=6)
    parser.add_argument("--excitation-count", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    generated = []
    for n in args.n_values:
        for k in args.k_values:
            seed = args.seed_base + n * 100 + k
            meta_path = generate_fixedk_dataset_bundle(
                grid_size=n,
                k=k,
                output_root=output_root,
                train_size=args.train_size,
                val_size=args.val_size,
                test_size=args.test_size,
                seed=seed,
                current_a=args.current_a,
                change_limit=args.change_limit,
                float_decimals=args.float_decimals,
                excitation_count=args.excitation_count,
            )
            generated.append(str(meta_path))

    print("Generated dataset bundles:")
    for item in generated:
        print(f"- {item}")


if __name__ == "__main__":
    main()
