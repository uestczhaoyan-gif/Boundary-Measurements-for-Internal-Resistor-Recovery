import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from model.model import MLPRegressor


NUM_RESISTORS = 112
EXCITATIONS = 32
BASE_R = 1000.0


class RegDataset(Dataset):
    def __init__(self, x, y_change, y_delta):
        self.x = torch.from_numpy(x).float()
        self.y_change = torch.from_numpy(y_change).float()
        self.y_delta = torch.from_numpy(y_delta).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_change[idx], self.y_delta[idx]


def build_dataset(csv_path, cache_path):
    if cache_path.exists():
        d = np.load(cache_path)
        return d["x"], d["y_change"], d["y_delta"]

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        v_cols = [c for c in reader.fieldnames if c.startswith("v_node")]
        v_num = len(v_cols)
        sums = np.zeros((EXCITATIONS, v_num), dtype=np.float64)
        cnts = np.zeros(EXCITATIONS, dtype=np.int64)

        prev_combo = None
        ex_idx = 0
        for row in reader:
            cid = int(row["combo_id"])
            if cid != prev_combo:
                prev_combo = cid
                ex_idx = 0
            if int(row["change_count"]) == 0:
                v = np.array([float(row[c]) for c in v_cols], dtype=np.float64)
                sums[ex_idx] += v
                cnts[ex_idx] += 1
            ex_idx += 1

        if np.any(cnts == 0):
            raise RuntimeError("0-change samples are insufficient to compute base mean.")
        base_mean = (sums / cnts[:, None]).astype(np.float32)

    x_list, yc_list, yd_list = [], [], []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        v_cols = [c for c in reader.fieldnames if c.startswith("v_node")]
        prev_combo = None
        combo_rows = []
        y_change = None
        y_delta = None

        for row in reader:
            cid = int(row["combo_id"])
            if cid != prev_combo:
                if prev_combo is not None:
                    arr = np.stack(combo_rows, axis=0).astype(np.float32)
                    x_list.append(arr - base_mean)
                    yc_list.append(y_change)
                    yd_list.append(y_delta)
                prev_combo = cid
                combo_rows = []
                y_change = np.zeros(NUM_RESISTORS, dtype=np.float32)
                y_delta = np.zeros(NUM_RESISTORS, dtype=np.float32)
                for i in (1, 2, 3):
                    rid = int(row[f"r{i}_id"])
                    if rid >= 0:
                        val = float(row[f"r{i}_value"])
                        y_change[rid] = 1.0
                        y_delta[rid] = val - BASE_R

            v = np.array([float(row[c]) for c in v_cols], dtype=np.float32)
            combo_rows.append(v)

        if combo_rows:
            arr = np.stack(combo_rows, axis=0).astype(np.float32)
            x_list.append(arr - base_mean)
            yc_list.append(y_change)
            yd_list.append(y_delta)

    x = np.stack(x_list, axis=0).reshape(len(x_list), -1).astype(np.float32)
    y_change = np.stack(yc_list, axis=0).astype(np.float32)
    y_delta = np.stack(yd_list, axis=0).astype(np.float32)
    np.savez_compressed(cache_path, x=x, y_change=y_change, y_delta=y_delta)
    return x, y_change, y_delta


def split_indices(n, seed):
    rng = random.Random(seed)
    ids = list(range(n))
    rng.shuffle(ids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def confusion(pred, true, num_classes=4):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    return cm


def macro_f1(cm):
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 0.0 if (p + r) == 0 else (2 * p * r / (p + r))
        f1s.append(f1)
    return float(np.mean(f1s))


def staged_sparse_weights(epoch, args):
    # Epoch 1..warmup: weak sparsity; then linearly ramp to target.
    if epoch <= args.sparse_warmup_epochs:
        return args.lambda_hinge_warm, args.lambda_sparse_warm
    if epoch >= args.sparse_ramp_end:
        return args.lambda_hinge_target, args.lambda_sparse_target
    ratio = (epoch - args.sparse_warmup_epochs) / max(args.sparse_ramp_end - args.sparse_warmup_epochs, 1)
    lam_h = args.lambda_hinge_warm + ratio * (args.lambda_hinge_target - args.lambda_hinge_warm)
    lam_s = args.lambda_sparse_warm + ratio * (args.lambda_sparse_target - args.lambda_sparse_warm)
    return lam_h, lam_s


def staged_w_change(epoch, args):
    if epoch <= args.change_warmup_epochs:
        return args.w_change_start
    if epoch >= args.change_ramp_end:
        return args.w_change_target
    ratio = (epoch - args.change_warmup_epochs) / max(args.change_ramp_end - args.change_warmup_epochs, 1)
    return args.w_change_start + ratio * (args.w_change_target - args.w_change_start)


def search_best_count_threshold(model, loader, device, t_min=40, t_max=70, t_step=1):
    all_pred = []
    all_true = []
    with torch.no_grad():
        for xb, ycb, _ in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            all_pred.append(np.abs(pred))
            all_true.append((ycb.numpy() > 0.5).sum(axis=1))
    pred_abs = np.concatenate(all_pred, axis=0)
    true_k = np.concatenate(all_true, axis=0).astype(np.int64)

    best_t = float(t_min)
    best_f = -1.0
    for t in np.arange(t_min, t_max + 1e-9, t_step):
        pred_k = np.clip((pred_abs > t).sum(axis=1), 0, 3).astype(np.int64)
        cm = confusion(pred_k, true_k, 4)
        f = macro_f1(cm)
        if f > best_f:
            best_f = float(f)
            best_t = float(t)
    return best_t, best_f


def evaluate_val(model, loader, device, huber, l1, args, ep):
    model.eval()
    val_loss = 0.0
    sum_changed_abs = 0.0
    n_changed = 0
    pred_gt_list = []
    lam_hinge, lam_sparse = staged_sparse_weights(ep, args)
    w_change = staged_w_change(ep, args)
    with torch.no_grad():
        for xb, ycb, ydb in loader:
            xb, ycb, ydb = xb.to(device), ycb.to(device), ydb.to(device)
            pred = model(xb)
            mask = ycb > 0.5

            if mask.any():
                loss_change = huber(pred[mask], ydb[mask]).mean()
                sum_changed_abs += torch.abs(pred[mask] - ydb[mask]).sum().item()
                n_changed += int(mask.sum().item())
            else:
                loss_change = huber(pred, ydb).mean()

            if (~mask).any():
                abs_u = pred[~mask].abs()
                loss_unchange = l1(pred[~mask], torch.zeros_like(pred[~mask])).mean()
                hinge = torch.relu(abs_u - args.hinge_threshold)
                hard_w = 1.0 + args.hardneg_alpha * (abs_u > args.hardneg_threshold).float()
                loss_hinge = (hard_w * hinge * hinge).mean()
            else:
                loss_unchange = torch.tensor(0.0, device=device)
                loss_hinge = torch.tensor(0.0, device=device)

            loss_sparse = pred.abs().mean()
            p = torch.sigmoid((pred.abs() - args.count_prob_threshold) / args.count_prob_tau)
            loss_count = torch.abs(p.sum(dim=1) - ycb.sum(dim=1)).mean()

            loss = (
                w_change * loss_change
                + args.w_unchange * loss_unchange
                + lam_hinge * loss_hinge
                + lam_sparse * loss_sparse
                + args.lambda_count * loss_count
            )
            val_loss += loss.item() * xb.size(0)
            pred_gt_list.extend((pred.abs() > args.eval_sparse_threshold).sum(dim=1).cpu().tolist())

    val_loss /= len(loader.dataset)
    val_mae_changed = sum_changed_abs / max(n_changed, 1)
    val_avg_gt = float(np.mean(pred_gt_list)) if pred_gt_list else 0.0
    val_score = val_mae_changed + args.val_sparse_alpha * val_avg_gt
    return val_loss, val_mae_changed, val_avg_gt, val_score


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y_change, y_delta = build_dataset(Path(args.data_path), Path(args.cache_path))
    tr, va, te = split_indices(len(x), args.seed)

    mean = x[tr].mean(axis=0, keepdims=True)
    std = x[tr].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x = ((x - mean) / std).astype(np.float32)
    np.savez_compressed(out_dir / "standardization.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))

    ds = RegDataset(x, y_change, y_delta)
    train_loader = DataLoader(Subset(ds, tr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(ds, va), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(ds, te), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressor(in_dim=x.shape[1], out_dim=NUM_RESISTORS, dropout=args.dropout, max_abs=args.max_abs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    huber = nn.SmoothL1Loss(reduction="none")
    l1 = nn.L1Loss(reduction="none")

    best_state = None
    best_epoch = 0
    best_score = float("inf")
    bad_epochs = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        lam_hinge, lam_sparse = staged_sparse_weights(ep, args)
        w_change = staged_w_change(ep, args)

        for xb, ycb, ydb in train_loader:
            xb, ycb, ydb = xb.to(device), ycb.to(device), ydb.to(device)
            pred = model(xb)
            mask = ycb > 0.5

            if mask.any():
                loss_change = huber(pred[mask], ydb[mask]).mean()
            else:
                loss_change = huber(pred, ydb).mean()

            if (~mask).any():
                abs_u = pred[~mask].abs()
                loss_unchange = l1(pred[~mask], torch.zeros_like(pred[~mask])).mean()
                hinge = torch.relu(abs_u - args.hinge_threshold)
                # Hard negatives: penalize large false alarms more heavily on unchanged positions.
                hard_w = 1.0 + args.hardneg_alpha * (abs_u > args.hardneg_threshold).float()
                loss_hinge = (hard_w * hinge * hinge).mean()
            else:
                loss_unchange = torch.tensor(0.0, device=device)
                loss_hinge = torch.tensor(0.0, device=device)

            loss_sparse = pred.abs().mean()
            p = torch.sigmoid((pred.abs() - args.count_prob_threshold) / args.count_prob_tau)
            loss_count = torch.abs(p.sum(dim=1) - ycb.sum(dim=1)).mean()

            loss = (
                w_change * loss_change
                + args.w_unchange * loss_unchange
                + lam_hinge * loss_hinge
                + lam_sparse * loss_sparse
                + args.lambda_count * loss_count
            )
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        if ep % args.log_every == 0 or ep == 1:
            va_loss, va_mae_changed, va_avg_gt, va_score = evaluate_val(model, val_loader, device, huber, l1, args, ep)
            print(
                f"Epoch {ep:03d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f} "
                f"| val_mae_changed={va_mae_changed:.4f} | val_avg(|dR|>{args.eval_sparse_threshold:.0f})={va_avg_gt:.2f}"
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

    model.eval()
    best_count_thr, val_count_f1 = search_best_count_threshold(
        model,
        val_loader,
        device,
        t_min=args.count_thr_min,
        t_max=args.count_thr_max,
        t_step=args.count_thr_step,
    )

    mae_all = 0.0
    mae_changed = 0.0
    n_all = 0
    n_changed = 0
    pred_counts = []
    true_counts = []

    with torch.no_grad():
        for xb, ycb, ydb in test_loader:
            xb, ycb, ydb = xb.to(device), ycb.to(device), ydb.to(device)
            pred = model(xb)
            mae_all += torch.abs(pred - ydb).sum().item()
            n_all += ydb.numel()
            mask = ycb > 0.5
            if mask.any():
                mae_changed += torch.abs(pred[mask] - ydb[mask]).sum().item()
                n_changed += mask.sum().item()
            pred_counts.extend((pred.abs() > best_count_thr).sum(dim=1).cpu().tolist())
            true_counts.extend((ycb > 0.5).sum(dim=1).cpu().tolist())

    mae_all /= max(n_all, 1)
    mae_changed = mae_changed / max(n_changed, 1)
    avg_gt = float(np.mean(pred_counts)) if pred_counts else 0.0
    cm = confusion([min(3, int(x)) for x in pred_counts], [min(3, int(x)) for x in true_counts], 4)

    print("\nTest Metrics (Regression):")
    print(f"mae_all={mae_all:.4f}")
    print(f"mae_changed={mae_changed:.4f}")
    print(f"best_count_threshold(val)={best_count_thr:.1f} | val_macro_f1={val_count_f1:.4f}")
    print(f"avg(|dR|>{best_count_thr:.1f})={avg_gt:.2f}")
    print("Derived Count Confusion Matrix (from regression threshold):")
    print("(rows=true, cols=pred)")
    print(cm)

    (out_dir / "confusion_matrix_count_test.txt").write_text(np.array2string(cm), encoding="utf-8")
    metrics = {
        "mae_all": mae_all,
        "mae_changed": mae_changed,
        "avg_abs_gt_threshold": avg_gt,
        "best_count_threshold": best_count_thr,
        "val_count_macro_f1": val_count_f1,
        "w_change_start": args.w_change_start,
        "w_change_target": args.w_change_target,
        "w_unchange": args.w_unchange,
        "lambda_hinge_warm": args.lambda_hinge_warm,
        "lambda_sparse_warm": args.lambda_sparse_warm,
        "lambda_hinge_target": args.lambda_hinge_target,
        "lambda_sparse_target": args.lambda_sparse_target,
        "lambda_count": args.lambda_count,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="64Nodes MLP regression modelo3.")
    parser.add_argument("--data-path", default="../../../data/training_data64.csv")
    parser.add_argument("--cache-path", default="./cache_dataset_reg.npz")
    parser.add_argument("--out-dir", default="./outputs")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-abs", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260320)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--w-change-start", type=float, default=1.6)
    parser.add_argument("--w-change-target", type=float, default=2.3)
    parser.add_argument("--change-warmup-epochs", type=int, default=15)
    parser.add_argument("--change-ramp-end", type=int, default=60)
    parser.add_argument("--w-unchange", type=float, default=1.4)

    parser.add_argument("--lambda-hinge-warm", type=float, default=0.08)
    parser.add_argument("--lambda-sparse-warm", type=float, default=0.004)
    parser.add_argument("--lambda-hinge-target", type=float, default=0.35)
    parser.add_argument("--lambda-sparse-target", type=float, default=0.020)
    parser.add_argument("--sparse-warmup-epochs", type=int, default=20)
    parser.add_argument("--sparse-ramp-end", type=int, default=70)

    parser.add_argument("--hinge-threshold", type=float, default=50.0)
    parser.add_argument("--hardneg-threshold", type=float, default=80.0)
    parser.add_argument("--hardneg-alpha", type=float, default=1.0)

    parser.add_argument("--lambda-count", type=float, default=0.10)
    parser.add_argument("--count-prob-threshold", type=float, default=50.0)
    parser.add_argument("--count-prob-tau", type=float, default=10.0)

    parser.add_argument("--eval-sparse-threshold", type=float, default=50.0)
    parser.add_argument("--val-sparse-alpha", type=float, default=0.35)

    parser.add_argument("--count-thr-min", type=float, default=40.0)
    parser.add_argument("--count-thr-max", type=float, default=70.0)
    parser.add_argument("--count-thr-step", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    if args.data_path == "../../../data/training_data64.csv":
        args.data_path = str(script_dir.parents[3] / "data" / "training_data64.csv")
    if args.cache_path == "./cache_dataset_reg.npz":
        args.cache_path = str(script_dir / "cache_dataset_reg.npz")
    if args.out_dir == "./outputs":
        args.out_dir = str(script_dir / "outputs")
    run(args)
