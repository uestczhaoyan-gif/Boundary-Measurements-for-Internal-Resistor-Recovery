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
    RegDataset,
    build_reg_dataset,
    compute_o4a2_loss,
    confusion,
    dump_json,
    evaluate_reg_val,
    load_partial_state_dict,
    macro_f1,
    mask_l1_weight_at_epoch,
    resolve_dataset_runtime_paths,
    search_best_count_threshold,
    set_global_seed,
    split_indices,
    standardize_graph_voltage,
)
from models import PhysicsInformedGNNRegressor
from topologies import get_topology


def parse_args(default_data_path: str, default_pretrained_model_path: str):
    parser = argparse.ArgumentParser(description="Topology-expand GNN regression training.")
    parser.add_argument("--data-path", default=default_data_path)
    parser.add_argument("--cache-path", default="cache_dataset_reg_expand.npz")
    parser.add_argument("--out-dir", default="./outputs")
    parser.add_argument("--dataset-tag", default="")
    parser.set_defaults(dataset_subdir=True)
    parser.add_argument("--dataset-subdir", dest="dataset_subdir", action="store_true")
    parser.add_argument("--no-dataset-subdir", dest="dataset_subdir", action="store_false")
    parser.add_argument("--pretrained-model-path", default=default_pretrained_model_path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-abs", type=float, default=300.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--edge-hidden", type=int, default=128)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--excitation-chunk-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260325)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-mask-l1", type=float, default=0.002)
    parser.add_argument("--lambda-mask-l1-start", type=float, default=0.0)
    parser.add_argument("--lambda-mask-warmup-epochs", type=int, default=20)
    parser.add_argument("--mask-init-prob", type=float, default=0.45)
    parser.add_argument("--mask-pos-weight", type=float, default=10.0)
    parser.add_argument("--mask-bce-weight", type=float, default=25.0)
    parser.add_argument("--reg-beta", type=float, default=25.0)
    parser.add_argument("--eval-sparse-threshold", type=float, default=50.0)
    parser.add_argument("--val-mae-all-alpha", type=float, default=0.12)
    parser.add_argument("--val-sparse-alpha", type=float, default=0.05)
    parser.add_argument("--count-thr-min", type=float, default=40.0)
    parser.add_argument("--count-thr-max", type=float, default=80.0)
    parser.add_argument("--count-thr-step", type=float, default=1.0)
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
    resolve_dataset_runtime_paths(args, script_dir, "cache_dataset_reg_expand.npz")
    topology = get_topology(topology_key)
    out_dir = Path(args.out_dir)
    cache_path = Path(args.cache_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)

    x, y_change, y_delta, ext_nodes = build_reg_dataset(Path(args.data_path), cache_path, topology)
    tr, va, te = split_indices(len(x), args.seed)
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

    train_ds = RegDataset(x[tr], y_change[tr], y_delta[tr])
    val_ds = RegDataset(x[va], y_change[va], y_delta[va])
    test_ds = RegDataset(x[te], y_change[te], y_delta[te])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicsInformedGNNRegressor(
        topology=topology,
        in_dim=4,
        hidden_dim=args.hidden_dim,
        edge_hidden=args.edge_hidden,
        heads=args.gat_heads,
        excitation_chunk_size=args.excitation_chunk_size,
        dropout=args.dropout,
        max_abs=args.max_abs,
        mask_init_prob=args.mask_init_prob,
    ).to(device)
    warm_info = load_partial_state_dict(model, args.pretrained_model_path, device, label=f"{stage_name}/reg")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = torch.tensor([args.mask_pos_weight], dtype=torch.float32, device=device)

    best_state = None
    best_epoch = 0
    best_score = float("inf")
    bad_epochs = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        current_mask_l1 = mask_l1_weight_at_epoch(ep, args)
        for xb, ycb, ydb in train_loader:
            xb, ycb, ydb = xb.to(device), ycb.to(device), ydb.to(device)
            pred, aux = model(xb, return_aux=True)
            loss = compute_o4a2_loss(pred, aux, ycb, ydb, current_mask_l1, pos_weight, args)
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        if ep % args.log_every == 0 or ep == 1:
            va_loss, va_mae_all, va_mae_changed, va_avg_gt, va_mask_mean, va_score = evaluate_reg_val(
                model,
                val_loader,
                device,
                args,
                current_mask_l1,
                pos_weight,
            )
            print(
                f"Epoch {ep:03d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f} "
                f"| val_mae_all={va_mae_all:.4f} | val_mae_changed={va_mae_changed:.4f} "
                f"| val_avg(|dR|>{args.eval_sparse_threshold:.0f})={va_avg_gt:.2f} | val_mask_mean={va_mask_mean:.4f}"
            )
            if va_score < best_score - 1e-6:
                best_score = va_score
                best_epoch = ep
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

    best_count_thr, val_count_f1 = search_best_count_threshold(
        model,
        val_loader,
        device,
        t_min=args.count_thr_min,
        t_max=args.count_thr_max,
        t_step=args.count_thr_step,
    )

    model.eval()
    mae_all = 0.0
    mae_changed = 0.0
    n_all = 0
    n_changed = 0
    pred_counts = []
    true_counts = []
    mask_probs = []
    with torch.no_grad():
        for xb, ycb, ydb in test_loader:
            xb, ycb, ydb = xb.to(device), ycb.to(device), ydb.to(device)
            pred, aux = model(xb, return_aux=True)
            mae_all += torch.abs(pred - ydb).sum().item()
            n_all += ydb.numel()
            mask = ycb > 0.5
            if mask.any():
                mae_changed += torch.abs(pred[mask] - ydb[mask]).sum().item()
                n_changed += int(mask.sum().item())
            pred_counts.extend((pred.abs() > best_count_thr).sum(dim=1).cpu().tolist())
            true_counts.extend((ycb > 0.5).sum(dim=1).cpu().tolist())
            mask_probs.append(aux["mask_prob"].mean().item())

    mae_all /= max(n_all, 1)
    mae_changed /= max(n_changed, 1)
    avg_gt = float(np.mean(pred_counts)) if pred_counts else 0.0
    avg_mask_prob = float(np.mean(mask_probs)) if mask_probs else 0.0
    cm = confusion([min(3, int(x)) for x in pred_counts], [min(3, int(x)) for x in true_counts], 4)

    (out_dir / "confusion_matrix_count_test.txt").write_text(np.array2string(cm), encoding="utf-8")
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
        "mae_all": mae_all,
        "mae_changed": mae_changed,
        "avg_abs_gt_threshold": avg_gt,
        "avg_mask_prob": avg_mask_prob,
        "best_count_threshold": best_count_thr,
        "val_count_macro_f1": val_count_f1,
        "lambda_mask_l1": args.lambda_mask_l1,
        "lambda_mask_l1_start": args.lambda_mask_l1_start,
        "lambda_mask_warmup_epochs": args.lambda_mask_warmup_epochs,
        "mask_init_prob": args.mask_init_prob,
        "mask_pos_weight": args.mask_pos_weight,
        "mask_bce_weight": args.mask_bce_weight,
        "reg_beta": args.reg_beta,
        "hidden_dim": args.hidden_dim,
        "edge_hidden": args.edge_hidden,
        "gat_heads": args.gat_heads,
        "excitation_chunk_size": args.excitation_chunk_size,
        "pretrained_model_path": args.pretrained_model_path,
        "warm_start": warm_info,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
    }
    dump_json(out_dir / "metrics.json", metrics)
    print("\nTest Metrics (Regression):")
    print(f"mae_all={mae_all:.4f}")
    print(f"mae_changed={mae_changed:.4f}")
    print(f"best_count_threshold(val)={best_count_thr:.1f} | val_macro_f1={val_count_f1:.4f}")
    print(cm)


if __name__ == "__main__":
    raise RuntimeError("Use a stage-specific wrapper under gnn/GNN_EXPAND/<stage>/reg/train.py")
