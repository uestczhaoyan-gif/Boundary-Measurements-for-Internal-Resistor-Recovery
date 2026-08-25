from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_common import DEFAULT_CHANGE_LIMIT, DEFAULT_CURRENT_A, generate_fixedk_dataset_bundle


def parse_int_list(raw: str) -> list[int]:
    values = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        values.append(int(piece))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multiple fixed-K square-grid dataset bundles.")
    parser.add_argument("--n-values", required=True, help="Comma-separated grid sizes, e.g. 6,7,8,9,10")
    parser.add_argument("--k-values", required=True, help="Comma-separated K values, e.g. 1,2,3,4,5,6")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260414)
    parser.add_argument("--current-a", type=float, default=DEFAULT_CURRENT_A)
    parser.add_argument("--change-limit", type=float, default=DEFAULT_CHANGE_LIMIT)
    parser.add_argument("--float-decimals", type=int, default=6)
    parser.add_argument("--excitation-count", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_values = parse_int_list(args.n_values)
    k_values = parse_int_list(args.k_values)

    for n in n_values:
        for k in k_values:
            meta_path = generate_fixedk_dataset_bundle(
                grid_size=n,
                k=k,
                output_root=Path(args.output_root).resolve(),
                train_size=args.train_size,
                val_size=args.val_size,
                test_size=args.test_size,
                seed=args.seed + n * 100 + k,
                current_a=args.current_a,
                change_limit=args.change_limit,
                float_decimals=args.float_decimals,
                excitation_count=args.excitation_count,
            )
            print(f"Generated dataset meta: {meta_path}")


if __name__ == "__main__":
    main()
