from __future__ import annotations

import argparse
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
VENDOR_DIR = WORKSPACE_ROOT / ".vendor_torchpy311"
if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader

from expand_common import (
    ClsDataset,
    class_weights,
    build_cls_dataset,
    confusion,
    coral_loss,
    dump_json,
    expit,
    load_partial_state_dict,
    macro_f1,
    resolve_dataset_runtime_paths,
    sanitize_dataset_tag,
    search_thresholds_constrained_weighted,
    set_global_seed,
    split_indices,
    standardize_graph_voltage,
    supervised_contrastive_loss,
)
from models import PhysicsInformedGNNClassifier
from topologies import get_topology


def parse_args(default_data_path: str, default_pretrained_model_path: str):
    parser = argparse.ArgumentParser(description="Topology-expand GNN classifier training.")
    parser.add_argument("--data-path", default=default_data_path)
    parser.add_argument("--cache-path", default="cache_dataset_cls_expand.npz")
    parser.add_argument("--out-dir", default="./outputs")
    parser.add_argument("--dataset-tag", default="")
    parser.set_defaults(dataset_subdir=True)
    parser.add_argument("--dataset-subdir", dest="dataset_subdir", action="store_true")
    parser.add_argument("--no-dataset-subdir", dest="dataset_subdir", action="store_false")
    parser.add_argument("--pretrained-model-path", default=default_pretrained_model_path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--excitation-chunk-size", type=int, default=4)
    parser.add_argument("--lambda-supcon", type=float, default=0.15)
    parser.add_argument("--contrast-temp", type=float, default=0.12)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--penalty-32", type=float, default=0.12)
    parser.add_argument("--bonus-r3", type=float, default=0.06)
    parser.add_argument("--bonus-r2", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260325)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args()


def main(
    stage_name: str,
    topology_key: str,
    default_data_path: str,
    default_pretrained_model_path: str = "",
    runtime_dir: Path | None = None,
):
    args = parse_args(default_data_path, default_pretrained_model_path)
    script_dir = Path(runtime_dir).resolve() if runtime_dir is not None else Path(__file__).resolve().parent
    resolve_dataset_runtime_paths(args, script_dir, "cache_dataset_cls_expand.npz")
    topology = get_topology(topology_key)
    out_dir = Path(args.out_dir)
    cache_path = Path(args.cache_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)

    x, y, ext_nodes = build_cls_dataset(Path(args.data_path), cache_path, topology)
    tr, va, te = split_indices(len(y), args.seed)
    mean = x[tr][:, :, ext_nodes, 2].mean(axis=0, keepdims=True)
    std = x[tr][:, :, ext_nodes, 2].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x = standardize_graph_voltage(x, mean, std, ext_nodes)
    np.savez_compressed(
        out_dir / "standardization.npz",
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        ext_nodes=ext_nodes.astype(np.int64),
    )

    train_ds = ClsDataset(x[tr], y[tr])
    val_ds = ClsDataset(x[va], y[va])
    test_ds = ClsDataset(x[te], y[te])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicsInformedGNNClassifier(
        topology=topology,
        in_dim=4,
        hidden_dim=args.hidden_dim,
        proj_dim=args.proj_dim,
        out_dim=3,
        heads=args.gat_heads,
        excitation_chunk_size=args.excitation_chunk_size,
        dropout=args.dropout,
    ).to(device)
    warm_info = load_partial_state_dict(model, args.pretrained_model_path, device, label=f"{stage_name}/cls")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    w_cls = class_weights(y[tr]).to(device)

    best_state = None
    best_epoch = 0
    best_score = -1e9
    best_thrs = [0.5, 0.5, 0.5]
    bad_epochs = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, aux = model(xb, return_aux=True)
            loss_coral = coral_loss(logits, yb, sample_w=w_cls[yb])
            loss_supcon = supervised_contrastive_loss(aux["contrast_feat"], yb, temperature=args.contrast_temp)
            loss = loss_coral + args.lambda_supcon * loss_supcon
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        if ep % args.log_every == 0 or ep == 1:
            model.eval()
            va_loss = 0.0
            val_logits = []
            val_true = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits, aux = model(xb, return_aux=True)
                    loss_coral = coral_loss(logits, yb, sample_w=w_cls[yb])
                    loss_supcon = supervised_contrastive_loss(aux["contrast_feat"], yb, temperature=args.contrast_temp)
                    va_loss += (loss_coral + args.lambda_supcon * loss_supcon).item() * xb.size(0)
                    val_logits.append(logits.cpu().numpy())
                    val_true.append(yb.cpu().numpy())
            va_loss /= len(val_loader.dataset)
            val_logits = np.concatenate(val_logits, axis=0)
            val_true = np.concatenate(val_true, axis=0)
            val_probs = expit(val_logits)
            thrs, va_f1, va_score = search_thresholds_constrained_weighted(
                val_probs,
                val_true,
                step=args.threshold_step,
                penalty_32=args.penalty_32,
                bonus_r3=args.bonus_r3,
                bonus_r2=args.bonus_r2,
            )
            print(f"Epoch {ep:03d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f} | val_macro_f1={va_f1:.4f}")
            if va_score > best_score + 1e-6:
                best_score = va_score
                best_epoch = ep
                best_thrs = thrs
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if args.patience > 0 and bad_epochs >= args.patience:
                    print(f"Early stopping at epoch {ep} (best_epoch={best_epoch}, best_val_score={best_score:.4f})")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "model_last.pt")

    model.eval()
    test_logits = []
    test_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device)).cpu().numpy()
            test_logits.append(logits)
            test_true.append(yb.numpy())
    test_logits = np.concatenate(test_logits, axis=0)
    test_true = np.concatenate(test_true, axis=0)
    test_probs = expit(test_logits)
    test_pred = (
        (test_probs[:, 0] > best_thrs[0]).astype(np.int64)
        + (test_probs[:, 1] > best_thrs[1]).astype(np.int64)
        + (test_probs[:, 2] > best_thrs[2]).astype(np.int64)
    )
    cm = confusion(test_pred, test_true)
    final_f1 = macro_f1(cm)

    (out_dir / "confusion_matrix.txt").write_text(np.array2string(cm), encoding="utf-8")
    metrics = {
        "stage_name": stage_name,
        "topology_key": topology.key,
        "topology_title": topology.title,
        "num_nodes": topology.num_nodes,
        "num_resistors": topology.num_resistors,
        "dataset_tag": args.dataset_tag,
        "data_path": str(Path(args.data_path)),
        "cache_path": str(cache_path),
        "out_dir": str(out_dir),
        "best_thresholds": best_thrs,
        "test_macro_f1": final_f1,
        "lambda_supcon": args.lambda_supcon,
        "contrast_temp": args.contrast_temp,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "proj_dim": args.proj_dim,
        "gat_heads": args.gat_heads,
        "excitation_chunk_size": args.excitation_chunk_size,
        "pretrained_model_path": args.pretrained_model_path,
        "warm_start": warm_info,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
    }
    dump_json(out_dir / "metrics.json", metrics)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)
    print(f"val_best_thresholds={best_thrs} | test_macro_f1={final_f1:.4f}")


if __name__ == "__main__":
    raise RuntimeError("Use a stage-specific wrapper under gnn/GNN_EXPAND/<stage>/cls/train.py")
