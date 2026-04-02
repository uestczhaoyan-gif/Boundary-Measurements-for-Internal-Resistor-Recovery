import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from model.model import HMultiTaskMLP


NUM_RESISTORS = 112
NUM_CLASSES = 4
EXCITATIONS = 32
BASE_R = 1000.0


class FullDataset(Dataset):
    def __init__(self, x, y_change, y_delta):
        self.x = torch.from_numpy(x).float()
        self.y_change = torch.from_numpy(y_change).float()
        self.y_delta = torch.from_numpy(y_delta).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_change[idx], self.y_delta[idx]


def coral_targets(labels, num_classes=NUM_CLASSES):
    thr = torch.arange(num_classes - 1, device=labels.device).view(1, -1)
    return (labels.view(-1, 1) > thr).float()


def coral_loss(logits, labels, sample_w=None):
    tgt = coral_targets(labels)
    loss = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
    if sample_w is not None:
        loss = loss * sample_w.view(-1, 1)
    return loss.mean()


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


def confusion(pred, true, num_classes=NUM_CLASSES):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    return cm


def class_weights(y_count):
    cnt = np.bincount(y_count, minlength=NUM_CLASSES).astype(np.float32)
    total = cnt.sum()
    w = total / (NUM_CLASSES * np.maximum(cnt, 1.0))
    return torch.tensor(w, dtype=torch.float32)


def search_thresholds_constrained(val_probs, val_true, step=0.01):
    grid = np.arange(0.05, 0.951, step)
    best_f = -1.0
    best_t = [0.5, 0.5, 0.5]
    for t1 in grid:
        m1 = val_probs[:, 0] > t1
        for t2 in grid[grid >= t1]:
            m2 = val_probs[:, 1] > t2
            for t3 in grid[grid >= t2]:
                pred = m1.astype(np.int64) + m2.astype(np.int64) + (val_probs[:, 2] > t3).astype(np.int64)
                f = macro_f1(confusion(pred, val_true))
                if f > best_f:
                    best_f = float(f)
                    best_t = [float(t1), float(t2), float(t3)]
    return best_t, best_f


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


def search_best_count_threshold_from_delta(model, loader, device, t_min=40, t_max=70, t_step=1):
    all_pred = []
    all_true = []
    with torch.no_grad():
        for xb, ycb, _ in loader:
            xb = xb.to(device)
            _, pred_delta = model(xb)
            all_pred.append(np.abs(pred_delta.cpu().numpy()))
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


def run(args):
    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y_change, y_delta = build_dataset(Path(args.data_path), Path(args.cache_path))
    tr, va, te = split_indices(len(x), args.seed)

    mean = x[tr].mean(axis=0, keepdims=True)
    std = x[tr].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x = ((x - mean) / std).astype(np.float32)
    np.savez_compressed(out_dir / "standardization.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))

    ds = FullDataset(x, y_change, y_delta)
    train_loader = DataLoader(Subset(ds, tr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(ds, va), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(ds, te), batch_size=args.batch_size, shuffle=False)

    y_count = np.clip(y_change.sum(axis=1).astype(np.int64), 0, 3)
    w_cls = class_weights(y_count[tr])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HMultiTaskMLP(
        in_dim=x.shape[1],
        out_dim=NUM_RESISTORS,
        dropout=args.dropout,
        max_abs=args.max_abs,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    huber = nn.SmoothL1Loss(reduction="none")
    l1 = nn.L1Loss(reduction="none")
    w_cls = w_cls.to(device)

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, ycb, ydb in train_loader:
            xb, ycb, ydb = xb.to(device), ycb.to(device), ydb.to(device)
            yk = torch.clamp(ycb.sum(dim=1), 0, 3).long()
            logits, pred = model(xb)

            loss_cls = coral_loss(logits, yk, sample_w=w_cls[yk])

            mask = ycb > 0.5
            if mask.any():
                loss_change = huber(pred[mask], ydb[mask]).mean()
            else:
                loss_change = huber(pred, ydb).mean()
            if (~mask).any():
                loss_unchange = l1(pred[~mask], torch.zeros_like(pred[~mask])).mean()
                hinge = torch.relu(pred[~mask].abs() - args.hinge_threshold)
                loss_hinge = (hinge * hinge).mean()
            else:
                loss_unchange = torch.tensor(0.0, device=device)
                loss_hinge = torch.tensor(0.0, device=device)
            loss_sparse = pred.abs().mean()

            p = torch.sigmoid((pred.abs() - args.prob_threshold) / args.prob_tau)
            count_pred = p.sum(dim=1)
            count_true = ycb.sum(dim=1)
            loss_count = F.l1_loss(count_pred, count_true)

            loss = (
                args.lambda_cls * loss_cls
                + args.w_change * loss_change
                + args.w_unchange * loss_unchange
                + args.lambda_hinge * loss_hinge
                + args.lambda_sparse * loss_sparse
                + args.lambda_count * loss_count
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        if ep % args.log_every == 0 or ep == 1:
            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for xb, ycb, ydb in val_loader:
                    xb, ycb, ydb = xb.to(device), ycb.to(device), ydb.to(device)
                    yk = torch.clamp(ycb.sum(dim=1), 0, 3).long()
                    logits, pred = model(xb)
                    loss_cls = coral_loss(logits, yk, sample_w=w_cls[yk])

                    mask = ycb > 0.5
                    if mask.any():
                        loss_change = huber(pred[mask], ydb[mask]).mean()
                    else:
                        loss_change = huber(pred, ydb).mean()
                    if (~mask).any():
                        loss_unchange = l1(pred[~mask], torch.zeros_like(pred[~mask])).mean()
                        hinge = torch.relu(pred[~mask].abs() - args.hinge_threshold)
                        loss_hinge = (hinge * hinge).mean()
                    else:
                        loss_unchange = torch.tensor(0.0, device=device)
                        loss_hinge = torch.tensor(0.0, device=device)
                    loss_sparse = pred.abs().mean()

                    p = torch.sigmoid((pred.abs() - args.prob_threshold) / args.prob_tau)
                    count_pred = p.sum(dim=1)
                    count_true = ycb.sum(dim=1)
                    loss_count = F.l1_loss(count_pred, count_true)

                    loss = (
                        args.lambda_cls * loss_cls
                        + args.w_change * loss_change
                        + args.w_unchange * loss_unchange
                        + args.lambda_hinge * loss_hinge
                        + args.lambda_sparse * loss_sparse
                        + args.lambda_count * loss_count
                    )
                    va_loss += loss.item() * xb.size(0)
            va_loss /= len(val_loader.dataset)
            print(f"Epoch {ep:03d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f}")

    torch.save(model.state_dict(), out_dir / "model_last.pt")

    # Validation calibrations
    model.eval()
    val_logits = []
    val_true_k = []
    with torch.no_grad():
        for xb, ycb, _ in val_loader:
            xb = xb.to(device)
            logits, _ = model(xb)
            val_logits.append(logits.cpu().numpy())
            val_true_k.append(np.clip(ycb.numpy().sum(axis=1), 0, 3).astype(np.int64))
    val_logits = np.concatenate(val_logits, axis=0)
    val_true_k = np.concatenate(val_true_k, axis=0)
    val_probs = 1.0 / (1.0 + np.exp(-val_logits / args.temp))
    coral_thrs, coral_val_f1 = search_thresholds_constrained(val_probs, val_true_k, step=args.thr_step)
    reg_count_thr, reg_val_f1 = search_best_count_threshold_from_delta(
        model,
        val_loader,
        device,
        t_min=args.count_thr_min,
        t_max=args.count_thr_max,
        t_step=args.count_thr_step,
    )

    # Test metrics
    mae_all = 0.0
    mae_changed = 0.0
    n_all = 0
    n_changed = 0
    pred_counts_head = []
    pred_counts_reg = []
    true_counts = []
    gt50_list = []

    with torch.no_grad():
        for xb, ycb, ydb in test_loader:
            xb, ycb, ydb = xb.to(device), ycb.to(device), ydb.to(device)
            logits, pred = model(xb)
            mae_all += torch.abs(pred - ydb).sum().item()
            n_all += ydb.numel()
            mask = ycb > 0.5
            if mask.any():
                mae_changed += torch.abs(pred[mask] - ydb[mask]).sum().item()
                n_changed += mask.sum().item()

            probs = torch.sigmoid(logits / args.temp).cpu().numpy()
            pred_k_head = (probs > np.array(coral_thrs, dtype=np.float32).reshape(1, -1)).sum(axis=1).astype(np.int64)
            pred_k_reg = np.clip((pred.abs().cpu().numpy() > reg_count_thr).sum(axis=1), 0, 3).astype(np.int64)
            true_k = np.clip(ycb.sum(dim=1).cpu().numpy(), 0, 3).astype(np.int64)

            pred_counts_head.extend(pred_k_head.tolist())
            pred_counts_reg.extend(pred_k_reg.tolist())
            true_counts.extend(true_k.tolist())
            gt50_list.extend((pred.abs() > 50.0).sum(dim=1).cpu().tolist())

    mae_all /= max(n_all, 1)
    mae_changed = mae_changed / max(n_changed, 1)
    avg_gt50 = float(np.mean(gt50_list)) if gt50_list else 0.0

    cm_head = confusion(pred_counts_head, true_counts, NUM_CLASSES)
    cm_reg = confusion(pred_counts_reg, true_counts, NUM_CLASSES)

    print("\nTest Metrics (Full h-multitask):")
    print(f"mae_all={mae_all:.4f}")
    print(f"mae_changed={mae_changed:.4f}")
    print(f"avg(|dR|>50)={avg_gt50:.2f}")
    print(f"CORAL thresholds={coral_thrs} | val_macro_f1={coral_val_f1:.4f}")
    print("Confusion Matrix (Count Head, rows=true, cols=pred):")
    print(cm_head)
    print(f"reg_count_threshold={reg_count_thr:.1f} | val_macro_f1={reg_val_f1:.4f}")
    print("Derived Count Confusion Matrix (from regression threshold):")
    print("(rows=true, cols=pred)")
    print(cm_reg)

    (out_dir / "confusion_matrix_count_head_test.txt").write_text(np.array2string(cm_head), encoding="utf-8")
    (out_dir / "confusion_matrix_count_reg_test.txt").write_text(np.array2string(cm_reg), encoding="utf-8")
    metrics = {
        "mae_all": mae_all,
        "mae_changed": mae_changed,
        "avg_abs_gt50": avg_gt50,
        "temperature": args.temp,
        "coral_thresholds": coral_thrs,
        "coral_val_macro_f1": coral_val_f1,
        "reg_count_threshold": reg_count_thr,
        "reg_val_macro_f1": reg_val_f1,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="64Nodes MLP full task: h-multitask model.")
    parser.add_argument("--data-path", default="../../../data/training_data64.csv")
    parser.add_argument("--cache-path", default="./cache_dataset_full.npz")
    parser.add_argument("--out-dir", default="./outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-abs", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260319)
    parser.add_argument("--log-every", type=int, default=5)

    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--w-change", type=float, default=1.5)
    parser.add_argument("--w-unchange", type=float, default=1.3)
    parser.add_argument("--lambda-hinge", type=float, default=0.3)
    parser.add_argument("--lambda-sparse", type=float, default=0.01)
    parser.add_argument("--lambda-count", type=float, default=0.5)
    parser.add_argument("--hinge-threshold", type=float, default=50.0)
    parser.add_argument("--prob-threshold", type=float, default=50.0)
    parser.add_argument("--prob-tau", type=float, default=12.0)

    parser.add_argument("--temp", type=float, default=2.0)
    parser.add_argument("--thr-step", type=float, default=0.01)
    parser.add_argument("--count-thr-min", type=float, default=40.0)
    parser.add_argument("--count-thr-max", type=float, default=70.0)
    parser.add_argument("--count-thr-step", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    if args.data_path == "../../../data/training_data64.csv":
        args.data_path = str(script_dir.parents[3] / "data" / "training_data64.csv")
    if args.cache_path == "./cache_dataset_full.npz":
        args.cache_path = str(script_dir / "cache_dataset_full.npz")
    if args.out_dir == "./outputs":
        args.out_dir = str(script_dir / "outputs")
    run(args)

