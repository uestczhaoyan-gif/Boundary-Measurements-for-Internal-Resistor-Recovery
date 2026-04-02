import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from model.model import MLPClassifierMultiHead


NUM_CLASSES = 4
EXCITATIONS = 32


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


def build_dataset(csv_path, cache_path):
    if cache_path.exists():
        d = np.load(cache_path)
        return d["x"], d["y"]

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

    x_list = []
    y_list = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        v_cols = [c for c in reader.fieldnames if c.startswith("v_node")]
        prev_combo = None
        combo_rows = []
        label = 0
        for row in reader:
            cid = int(row["combo_id"])
            if cid != prev_combo:
                if prev_combo is not None:
                    arr = np.stack(combo_rows, axis=0).astype(np.float32)
                    x_list.append(arr - base_mean)
                    y_list.append(label)
                prev_combo = cid
                combo_rows = []
                label = int(row["change_count"])
            combo_rows.append(np.array([float(row[c]) for c in v_cols], dtype=np.float32))
        if combo_rows:
            arr = np.stack(combo_rows, axis=0).astype(np.float32)
            x_list.append(arr - base_mean)
            y_list.append(label)

    x = np.stack(x_list, axis=0).reshape(len(x_list), -1).astype(np.float32)
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


def class_recall(cm, c):
    row = cm[c, :].sum()
    if row == 0:
        return 0.0
    return float(cm[c, c] / row)


def weighted_score(cm, penalty_32=0.12, bonus_r3=0.06):
    base = macro_f1(cm)
    row3 = max(int(cm[3, :].sum()), 1)
    err_32 = float(cm[3, 2] / row3)
    r3 = class_recall(cm, 3)
    return base - penalty_32 * err_32 + bonus_r3 * r3


def search_thresholds_constrained_weighted(val_probs, val_true, step=0.01, penalty_32=0.12, bonus_r3=0.06):
    grid = np.arange(0.05, 0.951, step)
    best_score = -1e9
    best_t = [0.5, 0.5, 0.5]
    best_f = -1.0
    for t1 in grid:
        m1 = val_probs[:, 0] > t1
        for t2 in grid[grid >= t1]:
            m2 = val_probs[:, 1] > t2
            for t3 in grid[grid >= t2]:
                pred = m1.astype(np.int64) + m2.astype(np.int64) + (val_probs[:, 2] > t3).astype(np.int64)
                cm = confusion(pred, val_true)
                score = weighted_score(cm, penalty_32=penalty_32, bonus_r3=bonus_r3)
                if score > best_score:
                    best_score = float(score)
                    best_t = [float(t1), float(t2), float(t3)]
                    best_f = macro_f1(cm)
    return best_t, best_f, best_score


def fuse_pred(main_probs, aux23_probs, main_thrs, aux_thr):
    pred_main = (main_probs > np.array(main_thrs, dtype=np.float32).reshape(1, -1)).sum(axis=1).astype(np.int64)
    pred = pred_main.copy()
    m = np.logical_or(pred_main == 2, pred_main == 3)
    if np.any(m):
        pred[m] = np.where(aux23_probs[m] > aux_thr, 3, 2)
    return pred


def search_aux23_threshold_weighted(main_probs, aux23_probs, true_labels, main_thrs, step=0.01, penalty_32=0.12, bonus_r3=0.06):
    best_thr = 0.5
    best_f = -1.0
    best_score = -1e9
    for t in np.arange(0.1, 0.901, step):
        pred = fuse_pred(main_probs, aux23_probs, main_thrs, t)
        cm = confusion(pred, true_labels)
        score = weighted_score(cm, penalty_32=penalty_32, bonus_r3=bonus_r3)
        if score > best_score:
            best_score = float(score)
            best_f = macro_f1(cm)
            best_thr = float(t)
    return best_thr, best_f, best_score


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
    model = MLPClassifierMultiHead(in_dim=x.shape[1], out_dim=NUM_CLASSES - 1, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    w_cls = class_weights(y[tr]).to(device)

    stage1_epochs = min(args.stage1_epochs, args.epochs)
    best_state = None
    best_f1 = -1.0
    best_epoch = 0
    bad_epochs = 0

    # Stage1: train only main CORAL path
    for ep in range(1, stage1_epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits_main, _ = model(xb)
            loss = coral_loss(logits_main, yb, sample_w=w_cls[yb])

            adj_mask = yb >= 2
            if adj_mask.any():
                target_last = (yb[adj_mask] > 2).float()
                loss_adj = F.binary_cross_entropy_with_logits(logits_main[adj_mask, 2], target_last)
            else:
                loss_adj = torch.tensor(0.0, device=device)

            loss = loss + args.lambda_adj * loss_adj
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        if ep % args.log_every == 0 or ep == 1 or ep == stage1_epochs:
            model.eval()
            va_loss = 0.0
            val_main = []
            val_true = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits_main, _ = model(xb)
                    loss = coral_loss(logits_main, yb, sample_w=w_cls[yb])
                    adj_mask = yb >= 2
                    if adj_mask.any():
                        target_last = (yb[adj_mask] > 2).float()
                        loss_adj = F.binary_cross_entropy_with_logits(logits_main[adj_mask, 2], target_last)
                    else:
                        loss_adj = torch.tensor(0.0, device=device)
                    loss = loss + args.lambda_adj * loss_adj
                    va_loss += loss.item() * xb.size(0)
                    val_main.append(logits_main.cpu().numpy())
                    val_true.append(yb.cpu().numpy())
            va_loss /= len(val_loader.dataset)
            val_main = np.concatenate(val_main, axis=0)
            val_true = np.concatenate(val_true, axis=0)
            val_main_probs = 1.0 / (1.0 + np.exp(-val_main / args.temp))
            thrs, val_f1, _ = search_thresholds_constrained_weighted(
                val_main_probs,
                val_true,
                step=args.thr_step,
                penalty_32=args.penalty_32,
                bonus_r3=args.bonus_r3,
            )
            val_pred = (val_main_probs > np.array(thrs).reshape(1, -1)).sum(axis=1).astype(np.int64)
            print(
                f"[Stage1] Epoch {ep:03d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f} "
                f"| val_macro_f1={macro_f1(confusion(val_pred, val_true)):.4f}"
            )
            if val_f1 > best_f1 + 1e-6:
                best_f1 = val_f1
                best_epoch = ep
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if args.patience > 0 and bad_epochs >= args.patience:
                    print(
                        f"[Stage1] Early stopping at epoch {ep} "
                        f"(best_epoch={best_epoch}, best_val_macro_f1={best_f1:.4f})"
                    )
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Freeze backbone + main head, train only aux head on true 2/3 samples.
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.main_head.parameters():
        p.requires_grad = False
    for p in model.aux_23_head.parameters():
        p.requires_grad = True
    opt_aux = torch.optim.AdamW(model.aux_23_head.parameters(), lr=args.lr_aux, weight_decay=args.weight_decay)

    # Calibrate main thresholds once from stage1-best model.
    model.eval()
    val_main = []
    val_aux = []
    val_true = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits_main, logits_aux = model(xb)
            val_main.append(logits_main.cpu().numpy())
            val_aux.append(logits_aux.cpu().numpy())
            val_true.append(yb.numpy())
    val_main = np.concatenate(val_main, axis=0)
    val_aux = np.concatenate(val_aux, axis=0)
    val_true = np.concatenate(val_true, axis=0)
    val_main_probs = 1.0 / (1.0 + np.exp(-val_main / args.temp))
    main_thrs, _, _ = search_thresholds_constrained_weighted(
        val_main_probs,
        val_true,
        step=args.thr_step,
        penalty_32=args.penalty_32,
        bonus_r3=args.bonus_r3,
    )
    best_aux_thr, best_fused_f1, _ = search_aux23_threshold_weighted(
        val_main_probs,
        1.0 / (1.0 + np.exp(-val_aux)),
        val_true,
        main_thrs,
        step=args.aux_thr_step,
        penalty_32=args.penalty_32,
        bonus_r3=args.bonus_r3,
    )

    best_aux_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_aux_epoch = 0
    bad_aux_epochs = 0
    stage2_epochs = max(args.epochs - stage1_epochs, 0)

    for ep2 in range(1, stage2_epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            _, logits_aux = model(xb)
            aux_mask = yb >= 2
            if not aux_mask.any():
                continue
            target_aux = (yb[aux_mask] > 2).float()
            target_aux = target_aux * (1.0 - args.aux_label_smoothing) + 0.5 * args.aux_label_smoothing
            loss_aux = F.binary_cross_entropy_with_logits(logits_aux[aux_mask], target_aux)
            opt_aux.zero_grad()
            loss_aux.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.aux_23_head.parameters(), args.grad_clip)
            opt_aux.step()
            tr_loss += loss_aux.item() * xb.size(0)
        tr_loss /= max(len(train_loader.dataset), 1)

        if ep2 % args.log_every == 0 or ep2 == 1 or ep2 == stage2_epochs:
            model.eval()
            val_main = []
            val_aux = []
            val_true = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    logits_main, logits_aux = model(xb)
                    val_main.append(logits_main.cpu().numpy())
                    val_aux.append(logits_aux.cpu().numpy())
                    val_true.append(yb.numpy())
            val_main = np.concatenate(val_main, axis=0)
            val_aux = np.concatenate(val_aux, axis=0)
            val_true = np.concatenate(val_true, axis=0)
            val_main_probs = 1.0 / (1.0 + np.exp(-val_main / args.temp))
            val_aux_probs = 1.0 / (1.0 + np.exp(-val_aux))

            aux_thr, fused_f1, _ = search_aux23_threshold_weighted(
                val_main_probs,
                val_aux_probs,
                val_true,
                main_thrs,
                step=args.aux_thr_step,
                penalty_32=args.penalty_32,
                bonus_r3=args.bonus_r3,
            )
            print(
                f"[Stage2] Epoch {ep2:03d} | train_aux_loss={tr_loss:.6f} "
                f"| val_macro_f1={fused_f1:.4f} | aux_thr={aux_thr:.2f}"
            )
            if fused_f1 > best_fused_f1 + 1e-6:
                best_fused_f1 = fused_f1
                best_aux_thr = aux_thr
                best_aux_epoch = ep2
                bad_aux_epochs = 0
                best_aux_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_aux_epochs += 1
                if args.patience_aux > 0 and bad_aux_epochs >= args.patience_aux:
                    print(
                        f"[Stage2] Early stopping at epoch {ep2} "
                        f"(best_epoch={best_aux_epoch}, best_val_macro_f1={best_fused_f1:.4f})"
                    )
                    break

    model.load_state_dict(best_aux_state)
    torch.save(model.state_dict(), out_dir / "model_last.pt")

    # Final test
    model.eval()
    test_main = []
    test_aux = []
    test_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits_main, logits_aux = model(xb.to(device))
            test_main.append(logits_main.cpu().numpy())
            test_aux.append(logits_aux.cpu().numpy())
            test_true.append(yb.numpy())
    test_main = np.concatenate(test_main, axis=0)
    test_aux = np.concatenate(test_aux, axis=0)
    test_true = np.concatenate(test_true, axis=0)
    test_main_probs = 1.0 / (1.0 + np.exp(-test_main / args.temp))
    test_aux_probs = 1.0 / (1.0 + np.exp(-test_aux))
    test_pred = fuse_pred(test_main_probs, test_aux_probs, main_thrs, best_aux_thr)
    cm = confusion(test_pred, test_true)

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    (out_dir / "confusion_matrix_test.txt").write_text(np.array2string(cm), encoding="utf-8")
    metrics = {
        "temperature": args.temp,
        "thresholds": main_thrs,
        "aux23_threshold": best_aux_thr,
        "val_macro_f1": best_fused_f1,
        "threshold_search": "constrained_t1_le_t2_le_t3_weighted_score",
        "stage1_epochs": stage1_epochs,
        "stage2_epochs": stage2_epochs,
        "best_stage1_epoch": best_epoch,
        "best_stage2_epoch": best_aux_epoch,
        "lambda_adj": args.lambda_adj,
        "penalty_32": args.penalty_32,
        "bonus_r3": args.bonus_r3,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="64Nodes MLP classification modelo4 (staged CORAL + 2v3 refiner).")
    parser.add_argument("--data-path", default="../../../data/training_data64.csv")
    parser.add_argument("--cache-path", default="./cache_dataset_cls.npz")
    parser.add_argument("--out-dir", default="./outputs")
    parser.add_argument("--epochs", type=int, default=110)
    parser.add_argument("--stage1-epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-aux", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--temp", type=float, default=2.0)
    parser.add_argument("--thr-step", type=float, default=0.01)
    parser.add_argument("--aux-thr-step", type=float, default=0.01)
    parser.add_argument("--lambda-adj", type=float, default=0.12)
    parser.add_argument("--aux-label-smoothing", type=float, default=0.05)
    parser.add_argument("--penalty-32", type=float, default=0.12)
    parser.add_argument("--bonus-r3", type=float, default=0.06)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--patience-aux", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260320)
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    if args.data_path == "../../../data/training_data64.csv":
        args.data_path = str(script_dir.parents[3] / "data" / "training_data64.csv")
    if args.cache_path == "./cache_dataset_cls.npz":
        args.cache_path = str(script_dir / "cache_dataset_cls.npz")
    if args.out_dir == "./outputs":
        args.out_dir = str(script_dir / "outputs")
    run(args)
