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

from model.model import MLPReg2Prob


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


def confusion(pred, true, num_classes=NUM_CLASSES):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    return cm


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPReg2Prob(in_dim=x.shape[1], out_dim=NUM_RESISTORS, dropout=args.dropout, max_abs=args.max_abs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    huber = nn.SmoothL1Loss(reduction="none")
    l1 = nn.L1Loss(reduction="none")

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, ycb, ydb in train_loader:
            xb, ycb, ydb = xb.to(device), ycb.to(device), ydb.to(device)
            pred = model(xb)
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
                args.w_change * loss_change
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
                    pred = model(xb)
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
                        args.w_change * loss_change
                        + args.w_unchange * loss_unchange
                        + args.lambda_hinge * loss_hinge
                        + args.lambda_sparse * loss_sparse
                        + args.lambda_count * loss_count
                    )
                    va_loss += loss.item() * xb.size(0)
            va_loss /= len(val_loader.dataset)
            print(f"Epoch {ep:03d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f}")

    torch.save(model.state_dict(), out_dir / "model_last.pt")

    model.eval()
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

            p = torch.sigmoid((pred.abs() - args.prob_threshold) / args.prob_tau)
            k_pred = torch.clamp(torch.round(p.sum(dim=1)), 0, 3).cpu().numpy().astype(np.int64)
            k_true = torch.clamp(ycb.sum(dim=1), 0, 3).cpu().numpy().astype(np.int64)
            pred_counts.extend(k_pred.tolist())
            true_counts.extend(k_true.tolist())

    mae_all /= max(n_all, 1)
    mae_changed = mae_changed / max(n_changed, 1)

    # avg(|dR|>50)
    with torch.no_grad():
        gt50_list = []
        for xb, _, _ in test_loader:
            xb = xb.to(device)
            pred = model(xb)
            gt50_list.extend((pred.abs() > 50.0).sum(dim=1).cpu().tolist())
    avg_gt50 = float(np.mean(gt50_list)) if gt50_list else 0.0

    cm = confusion(pred_counts, true_counts, NUM_CLASSES)

    print("\nTest Metrics (Full reg2prob):")
    print(f"mae_all={mae_all:.4f}")
    print(f"mae_changed={mae_changed:.4f}")
    print(f"avg(|dR|>50)={avg_gt50:.2f}")
    print("Derived Count Confusion Matrix (from regression threshold):")
    print("(rows=true, cols=pred)")
    print(cm)

    (out_dir / "confusion_matrix_count_test.txt").write_text(np.array2string(cm), encoding="utf-8")
    metrics = {
        "mae_all": mae_all,
        "mae_changed": mae_changed,
        "avg_abs_gt50": avg_gt50,
        "prob_threshold": args.prob_threshold,
        "prob_tau": args.prob_tau,
        "w_change": args.w_change,
        "w_unchange": args.w_unchange,
        "lambda_hinge": args.lambda_hinge,
        "lambda_sparse": args.lambda_sparse,
        "lambda_count": args.lambda_count,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="64Nodes MLP full task (reg2prob).")
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
    parser.add_argument("--w-change", type=float, default=1.5)
    parser.add_argument("--w-unchange", type=float, default=1.3)
    parser.add_argument("--lambda-hinge", type=float, default=0.3)
    parser.add_argument("--lambda-sparse", type=float, default=0.01)
    parser.add_argument("--lambda-count", type=float, default=0.5)
    parser.add_argument("--hinge-threshold", type=float, default=50.0)
    parser.add_argument("--prob-threshold", type=float, default=50.0)
    parser.add_argument("--prob-tau", type=float, default=12.0)
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
