from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = PROJECT_ROOT / "analysis"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from models.modelv1.inference import infer_from_meta
from models.modelv1.train import train_from_meta
from generate_square_fixedk_data import generate_fixedk_dataset_bundle
from physics_rank_study import analyze_excitation_counts


def read_kmax_summary(path: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mapping[int(row["N"])] = int(row["K_max"])
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-mainline excitation ablation experiments.")
    parser.add_argument("--n-values", nargs="+", type=int, default=[4, 7, 10])
    parser.add_argument("--excitation-counts", nargs="+", default=["1", "2", "4", "8", "full"])
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--outputs-root", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--kmax-summary", default=str(PROJECT_ROOT / "outputs" / "port_vs_kmax_summary.csv"))
    parser.add_argument("--train-size", type=int, default=12000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--seed-base", type=int, default=20260410)
    return parser.parse_args()


def parse_excitation(raw: str) -> int | None:
    return None if raw.lower() == "full" else int(raw)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    outputs_root = Path(args.outputs_root).resolve()
    kmax_summary = read_kmax_summary(Path(args.kmax_summary).resolve())
    excitation_counts = [parse_excitation(raw) for raw in args.excitation_counts]

    rows: list[dict] = []
    for n in args.n_values:
        kmax = kmax_summary.get(n, 0)
        candidate_k = sorted({value for value in [max(1, kmax - 1), kmax] if value > 0})
        for k in candidate_k:
            for excitation_count in excitation_counts:
                meta_path = generate_fixedk_dataset_bundle(
                    grid_size=n,
                    k=k,
                    output_root=data_root,
                    train_size=args.train_size,
                    val_size=args.val_size,
                    test_size=args.test_size,
                    seed=args.seed_base + n * 100 + k * 10 + (0 if excitation_count is None else excitation_count),
                    excitation_count=excitation_count,
                )
                excitation_tag = "full" if excitation_count is None else f"E{excitation_count}"
                run_out_dir = outputs_root / f"N{n}x{n}_K{k}_{excitation_tag}"
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
                    patience=10,
                    seed=args.seed_base + n * 100 + k * 10 + (0 if excitation_count is None else excitation_count),
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
                rows.append(
                    {
                        "N": n,
                        "K": k,
                        "excitation_count": "full" if excitation_count is None else excitation_count,
                        "test_id_exact_rate": infer_metrics["id_exact_rate"],
                        "test_value_accuracy": infer_metrics["value_accuracy"],
                        "test_mae_changed": infer_metrics["mae_changed"],
                        "pass_flag": int(bool(infer_metrics["pass_flag"])),
                    }
                )

        physics_rows = analyze_excitation_counts(n, excitation_counts)
        physics_csv = outputs_root / f"physics_rank_N{n}x{n}.csv"
        with physics_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(physics_rows[0].keys()))
            writer.writeheader()
            for row in physics_rows:
                writer.writerow(row)

    summary_csv = outputs_root / "excitation_ablation_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["N", "K", "excitation_count", "test_id_exact_rate", "test_value_accuracy", "test_mae_changed", "pass_flag"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
