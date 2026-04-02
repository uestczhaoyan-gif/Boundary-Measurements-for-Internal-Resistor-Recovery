import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

_VENDOR_DIR = Path(__file__).resolve().parents[3] / ".vendor_torchpy311"
if _VENDOR_DIR.exists() and str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from model.model import PhysicalGNNRegressor, build_boundary_mask, build_message_topology


NUM_RESISTORS = 112
NUM_EXCITATIONS = 32
NUM_NODES = 64
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

    out_base = script_dir / "outputs" if args.out_dir == "./outputs" else Path(args.out_dir)
    out_dir = out_base / dataset_tag if args.dataset_subdir else out_base

    args.data_path = str(data_path)
    args.cache_path = str(cache_path)
    args.out_dir = str(out_dir)
    args.dataset_tag = dataset_tag


def infer_current_a(data_path, fallback=0.01):
    data_path = Path(data_path)
    meta_candidates = [
        data_path.with_name(f"{data_path.stem}_meta.json"),
        data_path.parent / f"{data_path.stem}_meta.json",
    ]
    for candidate in meta_candidates:
        if candidate.exists():
            try:
                meta = json.loads(candidate.read_text(encoding="utf-8"))
                if "current_source_a" in meta:
                    return float(meta["current_source_a"])
            except Exception:
                pass
    return float(fallback)


def parse_voltage_columns(fieldnames):
    v_cols = [c for c in fieldnames if c.startswith("v_node")]
    ext_nodes = [int(c.replace("v_node", "")) for c in v_cols]
    return v_cols, np.array(ext_nodes, dtype=np.int64)


def expand_boundary_voltage(x_delta, ext_nodes):
    n = x_delta.shape[0]
    x_full = np.zeros((n, NUM_NODES, NUM_EXCITATIONS), dtype=np.float32)
    x_full[:, ext_nodes, :] = np.transpose(x_delta, (0, 2, 1))
    return x_full


def build_dataset(csv_path, cache_path):
    if cache_path.exists():
        d = np.load(cache_path)
        return d["x"], d["y_change"], d["y_delta"], d["ext_nodes"]

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        v_cols, ext_nodes = parse_voltage_columns(reader.fieldnames)
        v_num = len(v_cols)
        sums = np.zeros((NUM_EXCITATIONS, v_num), dtype=np.float64)
        cnts = np.zeros(NUM_EXCITATIONS, dtype=np.int64)

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
        v_cols, ext_nodes = parse_voltage_columns(reader.fieldnames)
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

            combo_rows.append(np.array([float(row[c]) for c in v_cols], dtype=np.float32))

        if combo_rows:
            arr = np.stack(combo_rows, axis=0).astype(np.float32)
            x_list.append(arr - base_mean)
            yc_list.append(y_change)
            yd_list.append(y_delta)

    x_delta = np.stack(x_list, axis=0).astype(np.float32)
    x_full = expand_boundary_voltage(x_delta, ext_nodes)
    y_change = np.stack(yc_list, axis=0).astype(np.float32)
    y_delta = np.stack(yd_list, axis=0).astype(np.float32)
    np.savez_compressed(cache_path, x=x_full, y_change=y_change, y_delta=y_delta, ext_nodes=ext_nodes)
    return x_full, y_change, y_delta, ext_nodes


def standardize_graph_boundary(x, mean, std, ext_nodes):
    x_std = x.copy()
    x_std[:, ext_nodes, :] = (x_std[:, ext_nodes, :] - mean) / std
    return x_std.astype(np.float32)


def split_indices(n, seed):
    rng = random.Random(seed)
    ids = list(range(n))
    rng.shuffle(ids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


class PhysicalRegDataset(torch.utils.data.Dataset):
    def __init__(self, x, y_change, y_delta):
        _resistor_edges, edge_index, edge_id = build_message_topology()
        boundary_mask = build_boundary_mask()
        self.x = torch.from_numpy(x).float()
        self.y_change = torch.from_numpy(y_change).float()
        self.y_delta = torch.from_numpy(y_delta).float()
        self.edge_index = edge_index
        self.edge_id = edge_id
        self.boundary_mask = boundary_mask

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return Data(
            x=self.x[idx],
            edge_index=self.edge_index,
            edge_id=self.edge_id,
            boundary_mask=self.boundary_mask,
            y_change=self.y_change[idx].unsqueeze(0),
            y_delta=self.y_delta[idx].unsqueeze(0),
        )


def confusion(pred, true, num_classes=4):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(pred, true):
        cm[t, p] += 1
    return cm


def macro_f1(cm):
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1s.append(0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(f1s))


def current_kcl_weight(epoch, args):
    if args.lambda_kcl <= 0:
        return 0.0
    if args.kcl_warmup_epochs <= 0:
        return float(args.lambda_kcl)
    warm = min(max(epoch / args.kcl_warmup_epochs, 0.0), 1.0)
    return float(args.lambda_kcl) * warm


def search_best_count_threshold(model, loader, device, t_min=40, t_max=80, t_step=1):
    all_pred = []
    all_true = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred, _aux = model(batch, return_aux=True)
            all_pred.append(np.abs(pred.cpu().numpy()))
            all_true.append((batch.y_change.cpu().numpy() > 0.5).sum(axis=1))

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


def evaluate_val(model, loader, device, args, kcl_weight):
    model.eval()
    total_loss = 0.0
    sum_all_abs = 0.0
    n_all = 0
    sum_changed_abs = 0.0
    n_changed = 0
    active_counts = []
    mask_means = []
    residuals = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            ycb = batch.y_change
            ydb = batch.y_delta
            pred, aux = model(batch, return_aux=True)
            loss_mse = F.mse_loss(pred, ydb)
            loss_mask = aux["mask_prob"].mean()
            loss_phys = aux["kcl_residual"]
            loss = loss_mse + args.lambda_mask_l1 * loss_mask + kcl_weight * loss_phys
            total_loss += loss.item() * batch.num_graphs

            sum_all_abs += torch.abs(pred - ydb).sum().item()
            n_all += ydb.numel()
            mask = ycb > 0.5
            if mask.any():
                sum_changed_abs += torch.abs(pred[mask] - ydb[mask]).sum().item()
                n_changed += int(mask.sum().item())
            active_counts.extend((pred.abs() > args.eval_sparse_threshold).sum(dim=1).cpu().tolist())
            mask_means.append(aux["mask_prob"].mean().item())
            residuals.append(loss_phys.item())

    val_loss = total_loss / max(len(loader.dataset), 1)
    val_mae_all = sum_all_abs / max(n_all, 1)
    val_mae_changed = sum_changed_abs / max(n_changed, 1)
    val_avg_gt = float(np.mean(active_counts)) if active_counts else 0.0
    val_mask_mean = float(np.mean(mask_means)) if mask_means else 0.0
    val_kcl = float(np.mean(residuals)) if residuals else 0.0
    val_score = (
        val_mae_changed
        + args.val_mae_all_alpha * val_mae_all
        + args.val_sparse_alpha * val_avg_gt
        + args.val_kcl_alpha * val_kcl
    )
    return val_loss, val_mae_all, val_mae_changed, val_avg_gt, val_mask_mean, val_kcl, val_score


def run(args):
    out_dir = Path(args.out_dir)
    cache_path = Path(args.cache_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    x, y_change, y_delta, ext_nodes = build_dataset(Path(args.data_path), cache_path)
    tr, va, te = split_indices(len(x), args.seed)

    mean = x[tr][:, ext_nodes, :].mean(axis=0, keepdims=True)
    std = x[tr][:, ext_nodes, :].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x = standardize_graph_boundary(x, mean, std, ext_nodes)
    np.savez_compressed(
        out_dir / "standardization.npz",
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        ext_nodes=ext_nodes.astype(np.int64),
    )

    ds = PhysicalRegDataset(x, y_change, y_delta)
    train_loader = DataLoader([ds[i] for i in tr], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader([ds[i] for i in va], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader([ds[i] for i in te], batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_a = infer_current_a(args.data_path, args.current_a if args.current_a is not None else 0.01)
    model = PhysicalGNNRegressor(
        num_excitations=NUM_EXCITATIONS,
        num_iters=args.num_iters,
        node_hidden=args.node_hidden,
        edge_hidden=args.edge_hidden,
        global_dim=args.global_dim,
        dropout=args.dropout,
        max_abs=args.max_abs,
        learn_alpha=not args.fix_alpha,
        alpha_init=args.alpha_init,
        alpha_max=args.alpha_max,
        conductance_init=args.conductance_init,
        edge_update_scale=args.edge_update_scale,
        voltage_update_scale=args.voltage_update_scale,
        current_a=current_a,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_epoch = 0
    best_score = float("inf")
    bad_epochs = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        kcl_weight = current_kcl_weight(ep, args)
        for batch in train_loader:
            batch = batch.to(device)
            ydb = batch.y_delta
            pred, aux = model(batch, return_aux=True)
            loss = (
                F.mse_loss(pred, ydb)
                + args.lambda_mask_l1 * aux["mask_prob"].mean()
                + kcl_weight * aux["kcl_residual"]
            )
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tr_loss += loss.item() * batch.num_graphs
        tr_loss /= len(train_loader.dataset)

        if ep % args.log_every == 0 or ep == 1:
            va_loss, va_mae_all, va_mae_changed, va_avg_gt, va_mask_mean, va_kcl, va_score = evaluate_val(
                model, val_loader, device, args, kcl_weight
            )
            print(
                f"Epoch {ep:03d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f} "
                f"| val_mae_all={va_mae_all:.4f} | val_mae_changed={va_mae_changed:.4f} "
                f"| val_avg(|dR|>{args.eval_sparse_threshold:.0f})={va_avg_gt:.2f} "
                f"| val_mask_mean={va_mask_mean:.4f} | val_kcl={va_kcl:.6f} | kcl_w={kcl_weight:.4f}"
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
    kcl_vals = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            ycb = batch.y_change
            ydb = batch.y_delta
            pred, aux = model(batch, return_aux=True)
            mae_all += torch.abs(pred - ydb).sum().item()
            n_all += ydb.numel()
            mask = ycb > 0.5
            if mask.any():
                mae_changed += torch.abs(pred[mask] - ydb[mask]).sum().item()
                n_changed += int(mask.sum().item())
            pred_counts.extend((pred.abs() > best_count_thr).sum(dim=1).cpu().tolist())
            true_counts.extend((ycb > 0.5).sum(dim=1).cpu().tolist())
            mask_probs.append(aux["mask_prob"].mean().item())
            kcl_vals.append(aux["kcl_residual"].item())

    mae_all /= max(n_all, 1)
    mae_changed /= max(n_changed, 1)
    avg_gt = float(np.mean(pred_counts)) if pred_counts else 0.0
    avg_mask_prob = float(np.mean(mask_probs)) if mask_probs else 0.0
    avg_kcl = float(np.mean(kcl_vals)) if kcl_vals else 0.0
    cm = confusion([min(3, int(x)) for x in pred_counts], [min(3, int(x)) for x in true_counts], 4)

    print("\nTest Metrics (Regression):")
    print(f"mae_all={mae_all:.4f}")
    print(f"mae_changed={mae_changed:.4f}")
    print(f"best_count_threshold(val)={best_count_thr:.1f} | val_macro_f1={val_count_f1:.4f}")
    print(f"avg(|dR|>{best_count_thr:.1f})={avg_gt:.2f}")
    print(f"avg(mask_prob)={avg_mask_prob:.4f}")
    print(f"avg(kcl_residual)={avg_kcl:.6f}")
    print("Derived Count Confusion Matrix (from regression threshold):")
    print("(rows=true, cols=pred)")
    print(cm)

    (out_dir / "confusion_matrix_count_test.txt").write_text(np.array2string(cm), encoding="utf-8")
    learned_conductance = model.conductance().detach().cpu().numpy()
    metrics = {
        "dataset_tag": args.dataset_tag,
        "data_path": str(Path(args.data_path)),
        "cache_path": str(cache_path),
        "out_dir": str(out_dir),
        "mae_all": mae_all,
        "mae_changed": mae_changed,
        "avg_abs_gt_threshold": avg_gt,
        "avg_mask_prob": avg_mask_prob,
        "avg_kcl_residual": avg_kcl,
        "best_count_threshold": best_count_thr,
        "val_count_macro_f1": val_count_f1,
        "lambda_mask_l1": args.lambda_mask_l1,
        "lambda_kcl": args.lambda_kcl,
        "kcl_warmup_epochs": args.kcl_warmup_epochs,
        "num_iters": args.num_iters,
        "alpha": float(model.alpha().detach().cpu().item()),
        "current_a": float(current_a),
        "mean_conductance": float(learned_conductance.mean()),
        "std_conductance": float(learned_conductance.std()),
        "best_epoch": best_epoch,
        "best_val_score": best_score,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="64Nodes physical-iteration GNN regression model_tp1.")
    parser.add_argument("--data-path", default=DEFAULT_MAIN_DATA_PATH)
    parser.add_argument("--cache-path", default="./cache_dataset_reg_tp1.npz")
    parser.add_argument("--out-dir", default="./outputs")
    parser.add_argument("--dataset-tag", default="", help="数据集标签；默认取 data-path 文件名。")
    parser.set_defaults(dataset_subdir=True)
    parser.add_argument("--dataset-subdir", dest="dataset_subdir", action="store_true")
    parser.add_argument("--no-dataset-subdir", dest="dataset_subdir", action="store_false")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-abs", type=float, default=300.0)
    parser.add_argument("--node-hidden", type=int, default=64)
    parser.add_argument("--edge-hidden", type=int, default=128)
    parser.add_argument("--global-dim", type=int, default=16)
    parser.add_argument("--num-iters", type=int, default=6)
    parser.add_argument("--alpha-init", type=float, default=0.10)
    parser.add_argument("--alpha-max", type=float, default=0.25)
    parser.add_argument("--fix-alpha", action="store_true")
    parser.add_argument("--conductance-init", type=float, default=1.0)
    parser.add_argument("--edge-update-scale", type=float, default=0.15)
    parser.add_argument("--voltage-update-scale", type=float, default=0.10)
    parser.add_argument("--current-a", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260326)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-mask-l1", type=float, default=0.002)
    parser.add_argument("--lambda-kcl", type=float, default=0.005)
    parser.add_argument("--kcl-warmup-epochs", type=int, default=30)
    parser.add_argument("--eval-sparse-threshold", type=float, default=50.0)
    parser.add_argument("--val-mae-all-alpha", type=float, default=0.12)
    parser.add_argument("--val-sparse-alpha", type=float, default=0.05)
    parser.add_argument("--val-kcl-alpha", type=float, default=0.02)
    parser.add_argument("--count-thr-min", type=float, default=40.0)
    parser.add_argument("--count-thr-max", type=float, default=80.0)
    parser.add_argument("--count-thr-step", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    resolve_dataset_runtime_paths(args, script_dir, "./cache_dataset_reg_tp1.npz")
    run(args)
