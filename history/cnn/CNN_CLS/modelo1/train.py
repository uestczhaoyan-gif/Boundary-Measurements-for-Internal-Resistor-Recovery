import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from model.model import CNN2DClassifier


NUM_CLASSES = 4
EXCITATIONS = 32
GRID = 8


class ClsDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def coral_targets(labels, num_classes=NUM_CLASSES):
    thr = torch.arange(num_classes - 1, device=labels.device).view(1, -1)
    return (labels.view(-1, 1) > thr).float()


def coral_loss(logits, labels, sample_w=None):
    tgt = coral_targets(labels)
    loss = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
    if sample_w is not None:
        loss = loss * sample_w.view(-1, 1)
    return loss.mean()


def parse_voltage_columns(fieldnames):
    v_cols = [c for c in fieldnames if c.startswith("v_node")]
    ext_nodes = [int(c.replace("v_node", "")) for c in v_cols]
    return v_cols, ext_nodes


def make_mask(ext_nodes):
    mask = np.zeros((GRID, GRID), dtype=np.float32)
    for n in ext_nodes:
        r, c = divmod(n, GRID)
        mask[r, c] = 1.0
    return mask


def to_cnn_input(x_delta, ext_nodes, src_nodes, gnd_nodes):
    # x_delta: [N, 32, 28] -> [N, 97, 8, 8]
    # channels: 32 voltage + 32 src maps + 32 gnd maps + 1 boundary mask
    n = x_delta.shape[0]
    v_ch = np.zeros((n, EXCITATIONS, GRID, GRID), dtype=np.float32)
    for j, node in enumerate(ext_nodes):
        r, c = divmod(node, GRID)
        v_ch[:, :, r, c] = x_delta[:, :, j]

    src_ch = np.zeros((n, EXCITATIONS, GRID, GRID), dtype=np.float32)
    gnd_ch = np.zeros((n, EXCITATIONS, GRID, GRID), dtype=np.float32)
    for e in range(EXCITATIONS):
        sr, sc = divmod(int(src_nodes[e]), GRID)
        gr, gc = divmod(int(gnd_nodes[e]), GRID)
        src_ch[:, e, sr, sc] = 1.0
        gnd_ch[:, e, gr, gc] = 1.0

    mask = make_mask(ext_nodes)
    mask = np.repeat(mask[None, None, :, :], n, axis=0)
    x = np.concatenate([v_ch, src_ch, gnd_ch, mask], axis=1).astype(np.float32)
    return x


def build_dataset(csv_path, cache_path):
    if cache_path.exists():
        d = np.load(cache_path)
        return d["x"], d["y"]

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        v_cols, ext_nodes = parse_voltage_columns(reader.fieldnames)
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

    x_list = []
    y_list = []
    src_nodes = None
    gnd_nodes = None
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        v_cols, ext_nodes = parse_voltage_columns(reader.fieldnames)
        prev_combo = None
        combo_rows = []
        combo_src = []
        combo_gnd = []
        label = 0
        for row in reader:
            cid = int(row["combo_id"])
            if cid != prev_combo:
                if prev_combo is not None:
                    arr = np.stack(combo_rows, axis=0).astype(np.float32)
                    x_list.append(arr - base_mean)
                    y_list.append(label)
                    if src_nodes is None:
                        src_nodes = np.array(combo_src, dtype=np.int64)
                        gnd_nodes = np.array(combo_gnd, dtype=np.int64)
                prev_combo = cid
                combo_rows = []
                combo_src = []
                combo_gnd = []
                label = int(row["change_count"])
            combo_rows.append(np.array([float(row[c]) for c in v_cols], dtype=np.float32))
            combo_src.append(int(row["src_node"]))
            combo_gnd.append(int(row["gnd_node"]))
        if combo_rows:
            arr = np.stack(combo_rows, axis=0).astype(np.float32)
            x_list.append(arr - base_mean)
            y_list.append(label)
            if src_nodes is None:
                src_nodes = np.array(combo_src, dtype=np.int64)
                gnd_nodes = np.array(combo_gnd, dtype=np.int64)

    x_delta = np.stack(x_list, axis=0).astype(np.float32)
    x = to_cnn_input(x_delta, ext_nodes, src_nodes, gnd_nodes)
    y = np.array(y_list, dtype=np.int64)
    np.savez_compressed(cache_path, x=x, y=y)
    return x, y


def split_indices(n, seed):
    rng = random.Random(seed)
    ids = list(range(n))
    rng.shuffle(ids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def class_weights(y):
    cnt = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    total = cnt.sum()
    w = total / (NUM_CLASSES * np.maximum(cnt, 1.0))
    return torch.tensor(w, dtype=torch.float32)


def confusion(pred, true, num_classes=NUM_CLASSES):
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


def search_thresholds_constrained(val_probs, val_true, step=0.01, fixed_t2=None):
    grid = np.arange(0.05, 0.951, step)
    best_f = -1.0
    best_t = [0.5, 0.5, 0.5]
    for t1 in grid:
        m1 = val_probs[:, 0] > t1
        if fixed_t2 is not None:
            t2_grid = np.array([fixed_t2], dtype=np.float32)
        else:
            t2_grid = grid[grid >= t1]
        for t2 in t2_grid:
            if t2 < t1:
                continue
            m2 = val_probs[:, 1] > t2
            for t3 in grid[grid >= t2]:
                pred = m1.astype(np.int64) + m2.astype(np.int64) + (val_probs[:, 2] > t3).astype(np.int64)
                f = macro_f1(confusion(pred, val_true))
                if f > best_f:
                    best_f = float(f)
                    best_t = [float(t1), float(t2), float(t3)]
    return best_t, best_f


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y = build_dataset(Path(args.data_path), Path(args.cache_path))
    tr, va, te = split_indices(len(y), args.seed)

    mean = x[tr].mean(axis=0, keepdims=True)
    std = x[tr].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x = ((x - mean) / std).astype(np.float32)
    np.savez_compressed(out_dir / "standardization.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))

    ds = ClsDataset(x, y)
    train_loader = DataLoader(Subset(ds, tr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(ds, va), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(ds, te), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN2DClassifier(in_ch=x.shape[1], out_dim=NUM_CLASSES - 1, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    w_cls = class_weights(y[tr]).to(device)
    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = coral_loss(logits, yb, sample_w=w_cls[yb])
            adj_mask = yb >= 2
            if adj_mask.any():
                target_last = (yb[adj_mask] > 2).float()
                loss_adj = F.binary_cross_entropy_with_logits(logits[adj_mask, 2], target_last)
            else:
                loss_adj = torch.tensor(0.0, device=device)
            loss = loss + args.lambda_adj * loss_adj
            opt.zero_grad()
            loss.backward()
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
                    logits = model(xb)
                    loss = coral_loss(logits, yb, sample_w=w_cls[yb])
                    adj_mask = yb >= 2
                    if adj_mask.any():
                        target_last = (yb[adj_mask] > 2).float()
                        loss_adj = F.binary_cross_entropy_with_logits(logits[adj_mask, 2], target_last)
                    else:
                        loss_adj = torch.tensor(0.0, device=device)
                    loss = loss + args.lambda_adj * loss_adj
                    va_loss += loss.item() * xb.size(0)
                    val_logits.append(logits.cpu().numpy())
                    val_true.append(yb.cpu().numpy())
            va_loss /= len(val_loader.dataset)
            val_logits = np.concatenate(val_logits, axis=0)
            val_true = np.concatenate(val_true, axis=0)
            val_probs = 1.0 / (1.0 + np.exp(-val_logits / args.temp))
            thrs, val_f1 = search_thresholds_constrained(
                val_probs,
                val_true,
                step=args.thr_step,
                fixed_t2=args.fixed_t2,
            )
            val_pred = (val_probs > np.array(thrs)).sum(axis=1)
            val_cm = confusion(val_pred, val_true)
            print(
                f"Epoch {ep:03d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f} | val_macro_f1={macro_f1(val_cm):.4f}"
            )

            if val_f1 > best_f1 + 1e-6:
                best_f1 = val_f1
                best_epoch = ep
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"Early stopping at epoch {ep} (best_epoch={best_epoch}, best_val_macro_f1={best_f1:.4f})")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), out_dir / "model_last.pt")

    model.eval()
    val_logits = []
    val_true = []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb.to(device))
            val_logits.append(logits.cpu().numpy())
            val_true.append(yb.numpy())
    val_logits = np.concatenate(val_logits, axis=0)
    val_true = np.concatenate(val_true, axis=0)
    val_probs = 1.0 / (1.0 + np.exp(-val_logits / args.temp))
    thrs, best_val_f1 = search_thresholds_constrained(
        val_probs,
        val_true,
        step=args.thr_step,
        fixed_t2=args.fixed_t2,
    )

    test_logits = []
    test_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            test_logits.append(logits.cpu().numpy())
            test_true.append(yb.numpy())
    test_logits = np.concatenate(test_logits, axis=0)
    test_true = np.concatenate(test_true, axis=0)
    test_probs = 1.0 / (1.0 + np.exp(-test_logits / args.temp))
    test_pred = (test_probs > np.array(thrs)).sum(axis=1)
    cm = confusion(test_pred, test_true)

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    (out_dir / "confusion_matrix_test.txt").write_text(np.array2string(cm), encoding="utf-8")
    metrics = {
        "temperature": args.temp,
        "thresholds": thrs,
        "val_macro_f1": best_val_f1,
        "threshold_search": "constrained_t1_le_t2_le_t3",
        "lambda_adj": args.lambda_adj,
        "best_epoch": best_epoch,
        "early_stopping_patience": args.patience,
        "fixed_t2": args.fixed_t2,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="64Nodes CNN2D-MLP classification (CORAL).")
    parser.add_argument("--data-path", default="../../../data/training_data64.csv")
    parser.add_argument("--cache-path", default="./cache_dataset_cls_v2.npz")
    parser.add_argument("--out-dir", default="./outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temp", type=float, default=2.0)
    parser.add_argument("--thr-step", type=float, default=0.01)
    parser.add_argument("--lambda-adj", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--fixed-t2", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260319)
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    if args.data_path == "../../../data/training_data64.csv":
        args.data_path = str(script_dir.parents[3] / "data" / "training_data64.csv")
    if args.cache_path == "./cache_dataset_cls_v2.npz":
        args.cache_path = str(script_dir / "cache_dataset_cls_v2.npz")
    if args.out_dir == "./outputs":
        args.out_dir = str(script_dir / "outputs")
    run(args)
