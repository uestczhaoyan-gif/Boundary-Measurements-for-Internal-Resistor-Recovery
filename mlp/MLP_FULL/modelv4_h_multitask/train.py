import argparse
import csv
import json
import random
import re
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
DEFAULT_MAIN_DATA_PATH = "../../../data/training_data64Nodes_2.csv"


def sanitize_dataset_tag(raw_tag):
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", raw_tag.strip())
    safe = safe.strip("._-")
    return safe or "dataset"


def resolve_input_data_path(raw_path, script_dir):
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()

    project_root = script_dir.parents[2]
    candidates = [path, script_dir / path, project_root / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (script_dir / path).resolve()


def resolve_dataset_runtime_paths(args, script_dir, default_cache_path):
    data_path = resolve_input_data_path(args.data_path, script_dir)

    dataset_tag = sanitize_dataset_tag(args.dataset_tag or data_path.stem)

    if args.cache_path == default_cache_path:
        cache_path = script_dir / "cache" / dataset_tag / Path(default_cache_path).name
    else:
        cache_path = Path(args.cache_path)

    if args.out_dir == "./outputs":
        out_base = script_dir / "outputs"
    else:
        out_base = Path(args.out_dir)
    out_dir = out_base / dataset_tag if args.dataset_subdir else out_base

    args.data_path = str(data_path)
    args.cache_path = str(cache_path)
    args.out_dir = str(out_dir)
    args.dataset_tag = dataset_tag


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


def staged_sparse_weights(stage_ep, args):
    if stage_ep <= args.sparse_warmup_epochs:
        return args.lambda_hinge_warm, args.lambda_sparse_warm
    if stage_ep >= args.sparse_ramp_end:
        return args.lambda_hinge_target, args.lambda_sparse_target
    ratio = (stage_ep - args.sparse_warmup_epochs) / max(args.sparse_ramp_end - args.sparse_warmup_epochs, 1)
    lam_h = args.lambda_hinge_warm + ratio * (args.lambda_hinge_target - args.lambda_hinge_warm)
    lam_s = args.lambda_sparse_warm + ratio * (args.lambda_sparse_target - args.lambda_sparse_warm)
    return lam_h, lam_s


def staged_w_change(stage_ep, args):
    if stage_ep <= args.change_warmup_epochs:
        return args.w_change_start
    if stage_ep >= args.change_ramp_end:
        return args.w_change_target
    ratio = (stage_ep - args.change_warmup_epochs) / max(args.change_ramp_end - args.change_warmup_epochs, 1)
    return args.w_change_start + ratio * (args.w_change_target - args.w_change_start)


def staged_cls_weight(stage_ep, args):
    if stage_ep <= 1:
        return args.lambda_cls_stage2_start
    if stage_ep >= args.cls_decay_epochs:
        return args.lambda_cls_stage2_end
    ratio = (stage_ep - 1) / max(args.cls_decay_epochs - 1, 1)
    return args.lambda_cls_stage2_start + ratio * (args.lambda_cls_stage2_end - args.lambda_cls_stage2_start)


def rank_separation_loss(pred, y_change, margin):
    abs_pred = pred.abs()
    losses = []
    for i in range(abs_pred.size(0)):
        mask = y_change[i] > 0.5
        if mask.any() and (~mask).any():
            m_pos = abs_pred[i][mask].mean()
            m_neg = abs_pred[i][~mask].mean()
            losses.append(torch.relu(margin - (m_pos - m_neg)))
    if not losses:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(losses).mean()


def eval_stage1(model, loader, device, w_cls, args):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, ycb, _ in loader:
            xb, ycb = xb.to(device), ycb.to(device)
            yk = torch.clamp(ycb.sum(dim=1), 0, 3).long()
            logits, _ = model(xb, detach_cls=False)
            loss_cls = coral_loss(logits, yk, sample_w=w_cls[yk])
            adj_mask = yk >= 2
            if adj_mask.any():
                target_last = (yk[adj_mask] > 2).float()
                loss_adj = F.binary_cross_entropy_with_logits(logits[adj_mask, 2], target_last)
            else:
                loss_adj = torch.tensor(0.0, device=device)
            loss = args.lambda_cls_stage1 * loss_cls + args.lambda_adj * loss_adj
            val_loss += loss.item() * xb.size(0)
    return val_loss / len(loader.dataset)


def eval_stage2(model, loader, device, huber, l1, w_cls, args, stage_ep):
    model.eval()
    val_loss = 0.0
    sum_changed_abs = 0.0
    n_changed = 0
    pred_gt50 = []
    lam_hinge, lam_sparse = staged_sparse_weights(stage_ep, args)
    w_change = staged_w_change(stage_ep, args)
    lambda_cls = staged_cls_weight(stage_ep, args)
    with torch.no_grad():
        for xb, ycb, ydb in loader:
            xb, ycb, ydb = xb.to(device), ycb.to(device), ydb.to(device)
            yk = torch.clamp(ycb.sum(dim=1), 0, 3).long()
            logits, pred = model(xb, detach_cls=args.detach_cls_in_stage2)

            loss_cls = coral_loss(logits, yk, sample_w=w_cls[yk])
            adj_mask = yk >= 2
            if adj_mask.any():
                target_last = (yk[adj_mask] > 2).float()
                loss_adj = F.binary_cross_entropy_with_logits(logits[adj_mask, 2], target_last)
            else:
                loss_adj = torch.tensor(0.0, device=device)

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
                loss_hinge = (hinge * hinge).mean()
            else:
                loss_unchange = torch.tensor(0.0, device=device)
                loss_hinge = torch.tensor(0.0, device=device)

            loss_sparse = pred.abs().mean()
            p = torch.sigmoid((pred.abs() - args.prob_threshold) / args.prob_tau)
            loss_count = torch.abs(p.sum(dim=1) - ycb.sum(dim=1)).mean()
            loss_rank = rank_separation_loss(pred, ycb, margin=args.rank_margin)
            count_cls_soft = torch.sigmoid(logits / args.temp).sum(dim=1)
            count_reg_soft = p.sum(dim=1)
            loss_align = F.smooth_l1_loss(count_cls_soft, count_reg_soft.detach())

            loss = (
                lambda_cls * (loss_cls + args.lambda_adj * loss_adj)
                + w_change * loss_change
                + args.w_unchange * loss_unchange
                + lam_hinge * loss_hinge
                + lam_sparse * loss_sparse
                + args.lambda_count * loss_count
                + args.lambda_rank * loss_rank
                + args.lambda_align * loss_align
            )
            val_loss += loss.item() * xb.size(0)
            pred_gt50.extend((pred.abs() > args.eval_sparse_threshold).sum(dim=1).cpu().tolist())

    val_loss /= len(loader.dataset)
    val_mae_changed = sum_changed_abs / max(n_changed, 1)
    val_avg_gt50 = float(np.mean(pred_gt50)) if pred_gt50 else 0.0
    val_score = val_mae_changed + args.val_sparse_alpha * val_avg_gt50
    return val_loss, val_mae_changed, val_avg_gt50, val_score


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


def warmstart_from_reg(model, reg_state_dict):
    loaded = 0
    model_sd = model.state_dict()
    for k, v in reg_state_dict.items():
        if not k.startswith("backbone."):
            continue
        parts = k.split(".")
        idx = int(parts[1])
        if idx >= 12:
            continue
        new_key = "trunk." + ".".join(parts[1:])
        if new_key in model_sd and model_sd[new_key].shape == v.shape:
            model_sd[new_key] = v
            loaded += 1
    model.load_state_dict(model_sd, strict=False)
    return loaded


def run(args):
    out_dir = Path(args.out_dir)
    cache_path = Path(args.cache_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    x, y_change, y_delta = build_dataset(Path(args.data_path), cache_path)
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
    if args.reg_warmstart_path:
        reg_path = Path(args.reg_warmstart_path)
        if reg_path.exists():
            reg_sd = torch.load(reg_path, map_location=device)
            loaded_n = warmstart_from_reg(model, reg_sd)
            print(f"[WarmStart] Loaded {loaded_n} trunk tensors from REG checkpoint: {reg_path}")
        else:
            print(f"[WarmStart] Skip, checkpoint not found: {reg_path}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    huber = nn.SmoothL1Loss(reduction="none")
    l1 = nn.L1Loss(reduction="none")
    w_cls = w_cls.to(device)

    stage1_epochs = min(args.stage1_epochs, args.epochs)
    stage2_total = max(args.epochs - stage1_epochs, 0)

    best_stage1 = None
    best_stage1_loss = float("inf")
    best_stage2 = None
    best_stage2_score = float("inf")
    best_stage2_epoch = 0
    bad_epochs = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0

        if ep <= stage1_epochs:
            for xb, ycb, _ in train_loader:
                xb, ycb = xb.to(device), ycb.to(device)
                yk = torch.clamp(ycb.sum(dim=1), 0, 3).long()
                logits, _ = model(xb, detach_cls=False)
                loss_cls = coral_loss(logits, yk, sample_w=w_cls[yk])

                adj_mask = yk >= 2
                if adj_mask.any():
                    target_last = (yk[adj_mask] > 2).float()
                    loss_adj = F.binary_cross_entropy_with_logits(logits[adj_mask, 2], target_last)
                else:
                    loss_adj = torch.tensor(0.0, device=device)

                loss = args.lambda_cls_stage1 * loss_cls + args.lambda_adj * loss_adj
                opt.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                tr_loss += loss.item() * xb.size(0)
            tr_loss /= len(train_loader.dataset)

            if ep % args.log_every == 0 or ep == 1 or ep == stage1_epochs:
                va_loss = eval_stage1(model, val_loader, device, w_cls, args)
                print(f"[Stage1] Epoch {ep:03d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f}")
                if va_loss < best_stage1_loss:
                    best_stage1_loss = va_loss
                    best_stage1 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            continue

        # Before stage2 starts, recover the best stage1 checkpoint once.
        if ep == stage1_epochs + 1 and best_stage1 is not None:
            model.load_state_dict(best_stage1)

        stage_ep = ep - stage1_epochs
        lam_hinge, lam_sparse = staged_sparse_weights(stage_ep, args)
        w_change = staged_w_change(stage_ep, args)
        lambda_cls = staged_cls_weight(stage_ep, args)

        for xb, ycb, ydb in train_loader:
            xb, ycb, ydb = xb.to(device), ycb.to(device), ydb.to(device)
            yk = torch.clamp(ycb.sum(dim=1), 0, 3).long()
            logits, pred = model(xb, detach_cls=args.detach_cls_in_stage2)

            loss_cls = coral_loss(logits, yk, sample_w=w_cls[yk])
            adj_mask = yk >= 2
            if adj_mask.any():
                target_last = (yk[adj_mask] > 2).float()
                loss_adj = F.binary_cross_entropy_with_logits(logits[adj_mask, 2], target_last)
            else:
                loss_adj = torch.tensor(0.0, device=device)

            mask = ycb > 0.5
            if mask.any():
                loss_change = huber(pred[mask], ydb[mask]).mean()
            else:
                loss_change = huber(pred, ydb).mean()

            if (~mask).any():
                abs_u = pred[~mask].abs()
                loss_unchange = l1(pred[~mask], torch.zeros_like(pred[~mask])).mean()
                hinge = torch.relu(abs_u - args.hinge_threshold)
                loss_hinge = (hinge * hinge).mean()
            else:
                loss_unchange = torch.tensor(0.0, device=device)
                loss_hinge = torch.tensor(0.0, device=device)

            loss_sparse = pred.abs().mean()
            p = torch.sigmoid((pred.abs() - args.prob_threshold) / args.prob_tau)
            loss_count = torch.abs(p.sum(dim=1) - ycb.sum(dim=1)).mean()
            loss_rank = rank_separation_loss(pred, ycb, margin=args.rank_margin)
            count_cls_soft = torch.sigmoid(logits / args.temp).sum(dim=1)
            count_reg_soft = p.sum(dim=1)
            loss_align = F.smooth_l1_loss(count_cls_soft, count_reg_soft.detach())

            loss = (
                lambda_cls * (loss_cls + args.lambda_adj * loss_adj)
                + w_change * loss_change
                + args.w_unchange * loss_unchange
                + lam_hinge * loss_hinge
                + lam_sparse * loss_sparse
                + args.lambda_count * loss_count
                + args.lambda_rank * loss_rank
                + args.lambda_align * loss_align
            )
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        if ep % args.log_every == 0 or stage_ep == 1:
            va_loss, va_mae_changed, va_avg_gt50, va_score = eval_stage2(
                model, val_loader, device, huber, l1, w_cls, args, stage_ep
            )
            print(
                f"[Stage2] Epoch {stage_ep:03d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f} "
                f"| val_mae_changed={va_mae_changed:.4f} | val_avg(|dR|>{args.eval_sparse_threshold:.0f})={va_avg_gt50:.2f}"
            )
            if va_score < best_stage2_score - 1e-6:
                best_stage2_score = va_score
                best_stage2_epoch = stage_ep
                best_stage2 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if args.patience > 0 and bad_epochs >= args.patience:
                    print(
                        f"[Stage2] Early stopping at epoch {stage_ep} "
                        f"(best_stage2_epoch={best_stage2_epoch}, best_val_score={best_stage2_score:.4f})"
                    )
                    break

    if stage2_total > 0 and best_stage2 is not None:
        model.load_state_dict(best_stage2)
    elif best_stage1 is not None:
        model.load_state_dict(best_stage1)
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
        "dataset_tag": args.dataset_tag,
        "data_path": str(Path(args.data_path)),
        "cache_path": str(cache_path),
        "out_dir": str(out_dir),
        "mae_all": mae_all,
        "mae_changed": mae_changed,
        "avg_abs_gt50": avg_gt50,
        "temperature": args.temp,
        "coral_thresholds": coral_thrs,
        "coral_val_macro_f1": coral_val_f1,
        "reg_count_threshold": reg_count_thr,
        "reg_val_macro_f1": reg_val_f1,
        "stage1_epochs": stage1_epochs,
        "stage2_epochs": stage2_total,
        "best_stage2_epoch": best_stage2_epoch,
        "best_stage2_score": best_stage2_score,
        "detach_cls_in_stage2": args.detach_cls_in_stage2,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="64Nodes MLP full task: modelv4_h_multitask (warmstart + align).")
    parser.add_argument("--data-path", default=DEFAULT_MAIN_DATA_PATH)
    parser.add_argument("--cache-path", default="./cache_dataset_full.npz")
    parser.add_argument("--out-dir", default="./outputs")
    parser.add_argument("--dataset-tag", default="", help="数据集标签；默认取 data-path 文件名。")
    parser.set_defaults(dataset_subdir=True)
    parser.add_argument("--dataset-subdir", dest="dataset_subdir", action="store_true", help="按数据集标签拆分 cache/outputs 子目录。")
    parser.add_argument("--no-dataset-subdir", dest="dataset_subdir", action="store_false", help="关闭按数据集拆分子目录。")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-abs", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260320)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--lambda-cls-stage1", type=float, default=1.0)
    parser.add_argument("--lambda-cls-stage2-start", type=float, default=0.20)
    parser.add_argument("--lambda-cls-stage2-end", type=float, default=0.10)
    parser.add_argument("--cls-decay-epochs", type=int, default=40)
    parser.add_argument("--lambda-adj", type=float, default=0.12)

    parser.set_defaults(detach_cls_in_stage2=True)
    parser.add_argument("--detach-cls-in-stage2", dest="detach_cls_in_stage2", action="store_true")
    parser.add_argument("--no-detach-cls-in-stage2", dest="detach_cls_in_stage2", action="store_false")

    parser.add_argument("--w-change-start", type=float, default=1.4)
    parser.add_argument("--w-change-target", type=float, default=2.0)
    parser.add_argument("--change-warmup-epochs", type=int, default=10)
    parser.add_argument("--change-ramp-end", type=int, default=45)
    parser.add_argument("--w-unchange", type=float, default=1.4)

    parser.add_argument("--lambda-hinge-warm", type=float, default=0.10)
    parser.add_argument("--lambda-sparse-warm", type=float, default=0.006)
    parser.add_argument("--lambda-hinge-target", type=float, default=0.30)
    parser.add_argument("--lambda-sparse-target", type=float, default=0.015)
    parser.add_argument("--sparse-warmup-epochs", type=int, default=10)
    parser.add_argument("--sparse-ramp-end", type=int, default=45)

    parser.add_argument("--lambda-count", type=float, default=0.30)
    parser.add_argument("--prob-threshold", type=float, default=50.0)
    parser.add_argument("--prob-tau", type=float, default=12.0)
    parser.add_argument("--hinge-threshold", type=float, default=50.0)
    parser.add_argument("--lambda-rank", type=float, default=0.12)
    parser.add_argument("--lambda-align", type=float, default=0.05)
    parser.add_argument("--rank-margin", type=float, default=35.0)
    parser.add_argument("--eval-sparse-threshold", type=float, default=50.0)
    parser.add_argument("--val-sparse-alpha", type=float, default=0.30)
    parser.add_argument("--reg-warmstart-path", default="")

    parser.add_argument("--temp", type=float, default=2.0)
    parser.add_argument("--thr-step", type=float, default=0.01)
    parser.add_argument("--count-thr-min", type=float, default=40.0)
    parser.add_argument("--count-thr-max", type=float, default=70.0)
    parser.add_argument("--count-thr-step", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    resolve_dataset_runtime_paths(args, script_dir, "./cache_dataset_full.npz")
    if args.reg_warmstart_path == "":
        reg_outputs = script_dir.parents[1] / "MLP_REG" / "modelo4" / "outputs"
        if args.dataset_subdir:
            args.reg_warmstart_path = str(reg_outputs / args.dataset_tag / "model_last.pt")
        else:
            args.reg_warmstart_path = str(reg_outputs / "model_last.pt")
    run(args)
