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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.modelv1 import ModelV1Regressor
from project_common import (
    apply_standardization,
    compute_fixedk_metrics,
    compute_standardization,
    dump_json,
    load_json,
    load_split_from_meta,
    run_dir_name,
)


class RegressionDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def predict_array(model: ModelV1Regressor, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).float().to(device)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.zeros((0, model.num_resistors), dtype=np.float32)


def evaluate_split(
    model: ModelV1Regressor,
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    pred = predict_array(model, x, batch_size=batch_size, device=device)
    metrics = compute_fixedk_metrics(pred, y, k=k)
    metrics.pop("per_sample", None)
    return metrics


def default_out_dir(meta_path: Path) -> Path:
    meta = load_json(meta_path)
    return PROJECT_ROOT / "outputs" / run_dir_name(
        grid_size=int(meta["topology"]["grid_size"]),
        k=int(meta["k"]),
        port_count=int(meta["topology"]["port_count"]),
        excitation_count=len(meta["excitations"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train square_scale_study modelv1.")
    parser.add_argument("--meta-path", required=True, help="Path to square_N..._meta.json")
    parser.add_argument("--out-dir", default="", help="Default: square_scale_study/outputs/<dataset_stem>")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--edge-hidden", type=int, default=128)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--excitation-chunk-size", type=int, default=4)
    parser.add_argument("--max-abs", type=float, default=250.0)
    parser.add_argument("--smooth-l1-beta", type=float, default=25.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260410)
    return parser.parse_args()


def train_from_meta(args: argparse.Namespace) -> dict:
    meta_path = Path(args.meta_path).resolve()
    meta = load_json(meta_path)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else default_out_dir(meta_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, _train_ids, topology, _ = load_split_from_meta(meta_path, "train")
    x_val, y_val, _val_ids, _topology2, _ = load_split_from_meta(meta_path, "val")
    boundary_nodes = np.asarray(topology.boundary_nodes_clockwise, dtype=np.int64)
    mean, std = compute_standardization(x_train, boundary_nodes)
    x_train = apply_standardization(x_train, boundary_nodes, mean, std)
    x_val = apply_standardization(x_val, boundary_nodes, mean, std)

    np.savez_compressed(
        out_dir / "standardization.npz",
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        boundary_nodes=boundary_nodes.astype(np.int64),
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_ds = RegressionDataset(x_train, y_train)
    val_ds = RegressionDataset(x_val, y_val)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelV1Regressor(
        topology=topology,
        hidden_dim=args.hidden_dim,
        edge_hidden=args.edge_hidden,
        heads=args.gat_heads,
        excitation_chunk_size=args.excitation_chunk_size,
        dropout=args.dropout,
        max_abs=args.max_abs,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_key = None
    best_epoch = 0
    best_state = None
    best_train_metrics = None
    best_val_metrics = None
    bad_epochs = 0
    history: list[dict] = []
    k = int(meta["k"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = F.smooth_l1_loss(pred, yb, beta=args.smooth_l1_beta)
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)

        train_loss = total_loss / max(len(train_ds), 1)
        train_metrics = evaluate_split(model, x_train, y_train, k=k, batch_size=args.eval_batch_size, device=device)
        val_metrics = evaluate_split(model, x_val, y_val, k=k, batch_size=args.eval_batch_size, device=device)
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_id_exact_rate": train_metrics["id_exact_rate"],
            "train_value_accuracy": train_metrics["value_accuracy"],
            "train_mae_changed": train_metrics["mae_changed"],
            "val_id_exact_rate": val_metrics["id_exact_rate"],
            "val_value_accuracy": val_metrics["value_accuracy"],
            "val_mae_changed": val_metrics["mae_changed"],
        }
        history.append(entry)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} "
            f"| train_id={train_metrics['id_exact_rate']:.4f} | train_value={train_metrics['value_accuracy']:.4f} "
            f"| val_id={val_metrics['id_exact_rate']:.4f} | val_value={val_metrics['value_accuracy']:.4f} "
            f"| val_mae={val_metrics['mae_changed']:.4f}"
        )

        current_key = (
            float(val_metrics["id_exact_rate"]),
            float(val_metrics["value_accuracy"]),
            -float(val_metrics["mae_changed"]),
        )
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_train_metrics = train_metrics
            best_val_metrics = val_metrics
            bad_epochs = 0
        else:
            bad_epochs += 1
            if args.patience > 0 and bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch} (best_epoch={best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "model_last.pt")

    payload = {
        "meta_path": str(meta_path),
        "dataset_stem": meta["dataset_stem"],
        "out_dir": str(out_dir),
        "k": int(meta["k"]),
        "grid_size": int(meta["topology"]["grid_size"]),
        "num_nodes": int(meta["topology"]["num_nodes"]),
        "num_resistors": int(meta["topology"]["num_resistors"]),
        "port_count": int(meta["topology"]["port_count"]),
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "model_config": {
            "hidden_dim": args.hidden_dim,
            "edge_hidden": args.edge_hidden,
            "gat_heads": args.gat_heads,
            "excitation_chunk_size": args.excitation_chunk_size,
            "dropout": args.dropout,
            "max_abs": args.max_abs,
            "smooth_l1_beta": args.smooth_l1_beta,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
        },
        "history": history,
    }
    dump_json(out_dir / "train_metrics.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    train_from_meta(args)


if __name__ == "__main__":
    main()
