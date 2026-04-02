import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from model.model import CNN2DRegressor


NUM_RESISTORS = 112
EXCITATIONS = 32
GRID = 8
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


class RegDataset(Dataset):
    def __init__(self, x, y_change, y_delta):
        self.x = torch.from_numpy(x).float()
        self.y_change = torch.from_numpy(y_change).float()
        self.y_delta = torch.from_numpy(y_delta).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_change[idx], self.y_delta[idx]


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
        return d["x"], d["y_change"], d["y_delta"]

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

    x_list, yc_list, yd_list = [], [], []
    src_nodes = None
    gnd_nodes = None
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        v_cols, ext_nodes = parse_voltage_columns(reader.fieldnames)
        prev_combo = None
        combo_rows = []
        combo_src = []
        combo_gnd = []
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
                    if src_nodes is None:
                        src_nodes = np.array(combo_src, dtype=np.int64)
                        gnd_nodes = np.array(combo_gnd, dtype=np.int64)
                prev_combo = cid
                combo_rows = []
                combo_src = []
                combo_gnd = []
                y_change = np.zeros(NUM_RESISTORS, dtype=np.float32)
                y_delta = np.zeros(NUM_RESISTORS, dtype=np.float32)
                for i in (1, 2, 3):
                    rid = int(row[f"r{i}_id"])
                    if rid >= 0:
                        val = float(row[f"r{i}_value"])
                        y_change[rid] = 1.0
                        y_delta[rid] = val - BASE_R
            combo_rows.append(np.array([float(row[c]) for c in v_cols], dtype=np.float32))
            combo_src.append(int(row["src_node"]))
            combo_gnd.append(int(row["gnd_node"]))
        if combo_rows:
            arr = np.stack(combo_rows, axis=0).astype(np.float32)
            x_list.append(arr - base_mean)
            yc_list.append(y_change)
            yd_list.append(y_delta)
            if src_nodes is None:
                src_nodes = np.array(combo_src, dtype=np.int64)
                gnd_nodes = np.array(combo_gnd, dtype=np.int64)

    x_delta = np.stack(x_list, axis=0).astype(np.float32)
    x = to_cnn_input(x_delta, ext_nodes, src_nodes, gnd_nodes)
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


def class_f1(cm, c):
    tp = cm[c, c]
    fp = cm[:, c].sum() - tp
    fn = cm[c, :].sum() - tp
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 0.0 if (p + r) == 0 else float(2 * p * r / (p + r))


def staged_sparse_weights(epoch, args):
    if epoch <= args.sparse_warmup_epochs:
        return args.lambda_hinge_warm, args.lambda_sparse_warm
    if epoch >= args.sparse_ramp_end:
        return args.lambda_hinge_target, args.lambda_sparse_target
    ratio = (epoch - args.sparse_warmup_epochs) / max(args.sparse_ramp_end - args.sparse_warmup_epochs, 1)
    lam_h = args.lambda_hinge_warm + ratio * (args.lambda_hinge_target - args.lambda_hinge_warm)
    lam_s = args.lambda_sparse_warm + ratio * (args.lambda_sparse_target - args.lambda_sparse_warm)
    return lam_h, lam_s


def count_threshold_score(cm, cls2_weight=0.35):
    return macro_f1(cm) + cls2_weight * class_f1(cm, 2)


def search_best_count_threshold(model, loader, device, t_min=40, t_max=70, t_step=1, cls2_weight=0.35):
    all_pred = []
    all_true = []
    with torch.no_grad():
        for xb, ycb, _ in loader:
            pred = model(xb.to(device)).cpu().numpy()
            all_pred.append(np.abs(pred))
            all_true.append((ycb.numpy() > 0.5).sum(axis=1))
    pred_abs = np.concatenate(all_pred, axis=0)
    true_k = np.concatenate(all_true, axis=0).astype(np.int64)

    best_t = float(t_min)
    best_f = -1.0
    best_score = -1e9
    for t in np.arange(t_min, t_max + 1e-9, t_step):
        pred_k = np.clip((pred_abs > t).sum(axis=1), 0, 3).astype(np.int64)
        cm = confusion(pred_k, true_k, 4)
        f = macro_f1(cm)
        score = count_threshold_score(cm, cls2_weight=cls2_weight)
        if score > best_score:
            best_score = float(score)
            best_f = float(f)
            best_t = float(t)
    return best_t, best_f, best_score


def false_positive_next_loss(pred, y_change, threshold=50.0, w_k2=1.25, w_k3=1.45):
    abs_sorted, _ = torch.sort(pred.abs(), dim=1, descending=True)
    k = torch.clamp(y_change.sum(dim=1).long(), 0, 3)
    next_rank = abs_sorted.gather(1, k.unsqueeze(1)).squeeze(1)
    weights = torch.ones_like(next_rank)
    weights = torch.where(k == 2, torch.full_like(weights, w_k2), weights)
    weights = torch.where(k == 3, torch.full_like(weights, w_k3), weights)
    return (weights * torch.relu(next_rank - threshold).pow(2)).mean()


def evaluate_val(model, loader, device, huber, l1, args, ep):
    model.eval()
    val_loss = 0.0
    sum_all_abs = 0.0
    n_all = 0
    sum_changed_abs = 0.0
    n_changed = 0
    pred_gt_list = []
    overpredict_list = []
    lam_hinge, lam_sparse = staged_sparse_weights(ep, args)

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

            sum_all_abs += torch.abs(pred - ydb).sum().item()
            n_all += ydb.numel()

            if (~mask).any():
                loss_unchange = l1(pred[~mask], torch.zeros_like(pred[~mask])).mean()
                hinge = torch.relu(pred[~mask].abs() - args.hinge_threshold)
                loss_hinge = (hinge * hinge).mean()
            else:
                loss_unchange = torch.tensor(0.0, device=device)
                loss_hinge = torch.tensor(0.0, device=device)

            loss_sparse = pred.abs().mean()
            loss_fp_next = false_positive_next_loss(
                pred,
                ycb,
                threshold=args.fp_next_threshold,
                w_k2=args.fp_next_weight_k2,
                w_k3=args.fp_next_weight_k3,
            )
            loss = (
                args.w_change * loss_change
                + args.w_unchange * loss_unchange
                + lam_hinge * loss_hinge
                + lam_sparse * loss_sparse
                + args.lambda_fp_next * loss_fp_next
            )
            val_loss += loss.item() * xb.size(0)

            pred_count = (pred.abs() > args.eval_sparse_threshold).sum(dim=1)
            true_count = (ycb > 0.5).sum(dim=1)
            pred_gt_list.extend(pred_count.cpu().tolist())
            overpredict = torch.relu(pred_count - true_count).float()
            overpredict_list.extend(overpredict.cpu().tolist())

    val_loss /= len(loader.dataset)
    val_mae_all = sum_all_abs / max(n_all, 1)
    val_mae_changed = sum_changed_abs / max(n_changed, 1)
    val_avg_gt = float(np.mean(pred_gt_list)) if pred_gt_list else 0.0
    val_avg_overpredict = float(np.mean(overpredict_list)) if overpredict_list else 0.0
    val_score = (
        val_mae_changed
        + args.val_mae_all_alpha * val_mae_all
        + args.val_sparse_alpha * val_avg_gt
        + args.val_overpredict_alpha * val_avg_overpredict
    )
    return val_loss, val_mae_all, val_mae_changed, val_avg_gt, val_avg_overpredict, val_score


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

    ds = RegDataset(x, y_change, y_delta)
    train_loader = DataLoader(Subset(ds, tr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(ds, va), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(ds, te), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN2DRegressor(in_ch=x.shape[1], out_dim=NUM_RESISTORS, dropout=args.dropout, max_abs=args.max_abs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    huber = nn.SmoothL1Loss(reduction="none")
    l1 = nn.L1Loss(reduction="none")
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        lam_hinge, lam_sparse = staged_sparse_weights(ep, args)
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
            loss_fp_next = false_positive_next_loss(
                pred,
                ycb,
                threshold=args.fp_next_threshold,
                w_k2=args.fp_next_weight_k2,
                w_k3=args.fp_next_weight_k3,
            )
            loss = (
                args.w_change * loss_change
                + args.w_unchange * loss_unchange
                + lam_hinge * loss_hinge
                + lam_sparse * loss_sparse
                + args.lambda_fp_next * loss_fp_next
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        if ep % args.log_every == 0 or ep == 1:
            va_loss, va_mae_all, va_mae_changed, va_avg_gt, va_avg_overpredict, va_score = evaluate_val(
                model, val_loader, device, huber, l1, args, ep
            )
            print(
                f"Epoch {ep:03d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f} "
                f"| val_mae_changed={va_mae_changed:.4f} | val_avg(|dR|>{args.eval_sparse_threshold:.0f})={va_avg_gt:.2f} "
                f"| val_overpredict={va_avg_overpredict:.3f}"
            )
            if va_loss < best_val - 1e-6:
                best_val = va_loss
                best_epoch = ep
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"Early stopping at epoch {ep} (best_epoch={best_epoch}, best_val_loss={best_val:.6f})")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "model_last.pt")
    model.eval()

    best_count_thr, val_count_f1, val_count_score = search_best_count_threshold(
        model,
        val_loader,
        device,
        t_min=args.count_thr_min,
        t_max=args.count_thr_max,
        t_step=args.count_thr_step,
        cls2_weight=args.count_cls2_weight,
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
    mae_changed /= max(n_changed, 1)
    avg_gt = float(np.mean(pred_counts)) if pred_counts else 0.0
    cm = confusion([min(3, int(x)) for x in pred_counts], [min(3, int(x)) for x in true_counts], 4)

    print("\nTest Metrics (Regression):")
    print(f"mae_all={mae_all:.4f}")
    print(f"mae_changed={mae_changed:.4f}")
    print(f"best_count_threshold(val)={best_count_thr:.1f} | val_macro_f1={val_count_f1:.4f} | val_score={val_count_score:.4f}")
    print(f"avg(|dR|>{best_count_thr:.1f})={avg_gt:.2f}")
    print("Derived Count Confusion Matrix (from regression threshold):")
    print("(rows=true, cols=pred)")
    print(cm)

    (out_dir / "confusion_matrix_count_test.txt").write_text(np.array2string(cm), encoding="utf-8")
    metrics = {
        "dataset_tag": args.dataset_tag,
        "data_path": str(Path(args.data_path)),
        "cache_path": str(cache_path),
        "out_dir": str(out_dir),
        "mae_all": mae_all,
        "mae_changed": mae_changed,
        "avg_abs_gt50": avg_gt,
        "best_count_threshold": best_count_thr,
        "val_count_macro_f1": val_count_f1,
        "val_count_score": val_count_score,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "early_stopping_patience": args.patience,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="64Nodes CNN2D-MLP regression.")
    parser.add_argument("--data-path", default=DEFAULT_MAIN_DATA_PATH)
    parser.add_argument("--cache-path", default="./cache_dataset_reg_v2.npz")
    parser.add_argument("--out-dir", default="./outputs")
    parser.add_argument("--dataset-tag", default="", help="数据集标签；默认取 data-path 文件名。")
    parser.set_defaults(dataset_subdir=True)
    parser.add_argument("--dataset-subdir", dest="dataset_subdir", action="store_true", help="按数据集标签拆分 cache/outputs 子目录。")
    parser.add_argument("--no-dataset-subdir", dest="dataset_subdir", action="store_false", help="关闭按数据集拆分子目录。")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-abs", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260319)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--w-change", type=float, default=2.0)
    parser.add_argument("--w-unchange", type=float, default=1.4)
    parser.add_argument("--lambda-hinge-warm", type=float, default=0.10)
    parser.add_argument("--lambda-sparse-warm", type=float, default=0.005)
    parser.add_argument("--lambda-hinge-target", type=float, default=0.30)
    parser.add_argument("--lambda-sparse-target", type=float, default=0.015)
    parser.add_argument("--sparse-warmup-epochs", type=int, default=20)
    parser.add_argument("--sparse-ramp-end", type=int, default=60)
    parser.add_argument("--hinge-threshold", type=float, default=50.0)
    parser.add_argument("--lambda-fp-next", type=float, default=0.0)
    parser.add_argument("--fp-next-threshold", type=float, default=50.0)
    parser.add_argument("--fp-next-weight-k2", type=float, default=1.25)
    parser.add_argument("--fp-next-weight-k3", type=float, default=1.45)
    parser.add_argument("--eval-sparse-threshold", type=float, default=50.0)
    parser.add_argument("--val-mae-all-alpha", type=float, default=0.12)
    parser.add_argument("--val-sparse-alpha", type=float, default=0.25)
    parser.add_argument("--val-overpredict-alpha", type=float, default=0.0)
    parser.add_argument("--count-thr-min", type=float, default=40.0)
    parser.add_argument("--count-thr-max", type=float, default=70.0)
    parser.add_argument("--count-thr-step", type=float, default=1.0)
    parser.add_argument("--count-cls2-weight", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    resolve_dataset_runtime_paths(args, script_dir, "./cache_dataset_reg_v2.npz")
    run(args)
