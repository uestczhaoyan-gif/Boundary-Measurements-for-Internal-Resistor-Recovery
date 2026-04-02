import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from model.model import Change3Regressor


GRID_SIZE = 8
NUM_NODES = GRID_SIZE * GRID_SIZE
NUM_RESISTORS = (GRID_SIZE * (GRID_SIZE - 1)) * 2
BASE_R = 1000.0
EXCITATIONS = 32


def build_edges():
    edges = [None] * NUM_RESISTORS
    block = (GRID_SIZE - 1) + GRID_SIZE
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE - 1):
            rid = block * r + c
            n1 = r * GRID_SIZE + c
            n2 = r * GRID_SIZE + (c + 1)
            edges[rid] = (n1, n2)
        if r < GRID_SIZE - 1:
            for c in range(GRID_SIZE):
                rid = block * r + (GRID_SIZE - 1) + c
                n1 = r * GRID_SIZE + c
                n2 = (r + 1) * GRID_SIZE + c
                edges[rid] = (n1, n2)
    return edges


class Change3Dataset(Dataset):
    def __init__(self, x, x_raw, y_change, y_delta, true_ids, true_vals):
        self.x = torch.from_numpy(x).float()
        self.x_raw = torch.from_numpy(x_raw).float()
        self.y_change = torch.from_numpy(y_change).float()
        self.y_delta = torch.from_numpy(y_delta).float()
        self.true_ids = torch.from_numpy(true_ids).long()
        self.true_vals = torch.from_numpy(true_vals).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.x_raw[idx],
            self.y_change[idx],
            self.y_delta[idx],
            self.true_ids[idx],
            self.true_vals[idx],
        )


def build_dataset(csv_path, cache_path):
    if cache_path.exists():
        d = np.load(cache_path)
        return (
            d["x"],
            d["x_raw"],
            d["y_change"],
            d["y_delta"],
            d["true_ids"],
            d["true_vals"],
            d["ext_nodes"],
            d["excitations"],
        )

    x_list = []
    x_raw_list = []
    yc_list = []
    yd_list = []
    tid_list = []
    tval_list = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        v_cols = [c for c in reader.fieldnames if c.startswith("v_node")]
        ext_nodes = np.array([int(c.replace("v_node", "")) for c in v_cols], dtype=np.int64)

        prev_combo = None
        combo_rows = []
        combo_ex = []
        y_change = None
        y_delta = None
        true_ids = None
        true_vals = None
        ex_ref = None

        def flush_combo():
            if not combo_rows:
                return
            nonlocal ex_ref
            x_raw = np.stack(combo_rows, axis=0).astype(np.float32)
            if x_raw.shape[0] != EXCITATIONS:
                raise RuntimeError(f"Each combo must have {EXCITATIONS} rows, got {x_raw.shape[0]}")
            ex_arr = np.array(combo_ex, dtype=np.int64)
            if ex_ref is None:
                ex_ref = ex_arr
            else:
                if not np.array_equal(ex_ref, ex_arr):
                    raise RuntimeError("Excitation order mismatch across combos.")

            x_raw_list.append(x_raw)
            x_list.append(x_raw.reshape(-1))
            yc_list.append(y_change)
            yd_list.append(y_delta)
            tid_list.append(true_ids)
            tval_list.append(true_vals)

        for row in reader:
            cid = int(row["combo_id"])
            if cid != prev_combo:
                if prev_combo is not None:
                    flush_combo()
                prev_combo = cid
                combo_rows = []
                combo_ex = []
                y_change = np.zeros(NUM_RESISTORS, dtype=np.float32)
                y_delta = np.zeros(NUM_RESISTORS, dtype=np.float32)
                true_ids = np.zeros(3, dtype=np.int64)
                true_vals = np.zeros(3, dtype=np.float32)
                for i in (1, 2, 3):
                    rid = int(row[f"r{i}_id"])
                    rv = float(row[f"r{i}_value"])
                    true_ids[i - 1] = rid
                    true_vals[i - 1] = rv - BASE_R
                    y_change[rid] = 1.0
                    y_delta[rid] = rv - BASE_R

            combo_rows.append(np.array([float(row[c]) for c in v_cols], dtype=np.float32))
            combo_ex.append((int(row["src_node"]), int(row["gnd_node"])))

        if combo_rows:
            flush_combo()

    x = np.stack(x_list, axis=0).astype(np.float32)
    x_raw = np.stack(x_raw_list, axis=0).astype(np.float32)
    y_change = np.stack(yc_list, axis=0).astype(np.float32)
    y_delta = np.stack(yd_list, axis=0).astype(np.float32)
    true_ids = np.stack(tid_list, axis=0).astype(np.int64)
    true_vals = np.stack(tval_list, axis=0).astype(np.float32)
    excitations = ex_ref.astype(np.int64)

    np.savez_compressed(
        cache_path,
        x=x,
        x_raw=x_raw,
        y_change=y_change,
        y_delta=y_delta,
        true_ids=true_ids,
        true_vals=true_vals,
        ext_nodes=ext_nodes,
        excitations=excitations,
    )
    return x, x_raw, y_change, y_delta, true_ids, true_vals, ext_nodes, excitations


def split_indices(n, seed):
    rng = random.Random(seed)
    ids = list(range(n))
    rng.shuffle(ids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def build_g_batch(rvals, edge_n1, edge_n2):
    bsz = rvals.shape[0]
    g = torch.zeros((bsz, NUM_NODES, NUM_NODES), device=rvals.device, dtype=rvals.dtype)
    cond = 1.0 / torch.clamp(rvals, min=1e-3)
    for e in range(NUM_RESISTORS):
        n1 = int(edge_n1[e])
        n2 = int(edge_n2[e])
        ce = cond[:, e]
        g[:, n1, n1] = g[:, n1, n1] + ce
        g[:, n2, n2] = g[:, n2, n2] + ce
        g[:, n1, n2] = g[:, n1, n2] - ce
        g[:, n2, n1] = g[:, n2, n1] - ce
    return g


def solve_ext_batch(g_batch, src, gnd, keep_idx, ext_nodes, current_a):
    bsz = g_batch.shape[0]
    idx = keep_idx[gnd]
    g_red = g_batch.index_select(1, idx).index_select(2, idx)

    i = torch.zeros((bsz, NUM_NODES), device=g_batch.device, dtype=g_batch.dtype)
    i[:, src] = current_a
    i[:, gnd] = -current_a
    i_red = i.index_select(1, idx).unsqueeze(-1)

    v_red = torch.linalg.solve(g_red, i_red).squeeze(-1)
    v_full = torch.zeros((bsz, NUM_NODES), device=g_batch.device, dtype=g_batch.dtype)
    v_full[:, idx] = v_red
    return v_full.index_select(1, ext_nodes)


def physics_loss(pred_delta, target_vraw, edge_n1, edge_n2, keep_idx, ext_nodes, excitations, args):
    rvals = torch.clamp(BASE_R + pred_delta, min=args.r_min, max=args.r_max)
    g_batch = build_g_batch(rvals, edge_n1, edge_n2)
    mse = nn.MSELoss(reduction="mean")

    ex_sel = random.sample(range(EXCITATIONS), k=min(4, EXCITATIONS))
    loss = torch.tensor(0.0, device=pred_delta.device)
    for exi in ex_sel:
        src = int(excitations[exi, 0])
        gnd = int(excitations[exi, 1])
        pred_ext = solve_ext_batch(g_batch, src, gnd, keep_idx, ext_nodes, args.current_a)
        true_ext = target_vraw[:, exi, :]
        loss = loss + mse(pred_ext, true_ext)
    return loss / len(ex_sel)


def load_coords_tensor(coords_path, device):
    data = json.loads(Path(coords_path).read_text(encoding="utf-8"))
    arr = np.zeros((NUM_RESISTORS, 2), dtype=np.float32)
    for k, v in data.items():
        arr[int(k)] = np.array(v, dtype=np.float32)
    return torch.from_numpy(arr).to(device)


def coord_moment_loss(pred_delta, y_change, coords, temp=20.0):
    pred_p = torch.softmax(pred_delta.abs() / temp, dim=1)
    denom = y_change.sum(dim=1, keepdim=True).clamp_min(1.0)
    true_p = y_change / denom

    mu_pred = pred_p @ coords
    mu_true = true_p @ coords

    c_pred = coords.unsqueeze(0) - mu_pred.unsqueeze(1)
    c_true = coords.unsqueeze(0) - mu_true.unsqueeze(1)
    var_pred = (pred_p.unsqueeze(-1) * (c_pred * c_pred)).sum(dim=1)
    var_true = (true_p.unsqueeze(-1) * (c_true * c_true)).sum(dim=1)

    loss_mu = torch.nn.functional.smooth_l1_loss(mu_pred, mu_true)
    loss_var = torch.nn.functional.smooth_l1_loss(var_pred, var_true)
    return loss_mu + loss_var


def position_hit_rates(pred_delta, true_ids):
    pred_top3 = torch.topk(pred_delta.abs(), k=3, dim=1).indices
    hit_count = {0: 0, 1: 0, 2: 0, 3: 0}
    n = pred_top3.size(0)
    for i in range(n):
        pset = set(pred_top3[i].cpu().tolist())
        tset = set(true_ids[i].cpu().tolist())
        hit = len(pset.intersection(tset))
        hit_count[hit] += 1
    rates = {k: hit_count[k] / max(n, 1) for k in hit_count}
    return hit_count, rates


def evaluate(model, loader, device, mse, args, coords=None):
    model.eval()
    val_loss = 0.0
    sum_all_abs = 0.0
    n_all = 0
    sum_changed_abs = 0.0
    n_changed = 0
    hit_count_all = {0: 0, 1: 0, 2: 0, 3: 0}

    with torch.no_grad():
        for xb, xrawb, ycb, ydb, tids, _tvals in loader:
            xb = xb.to(device)
            xrawb = xrawb.to(device)
            ycb = ycb.to(device)
            ydb = ydb.to(device)
            tids = tids.to(device)

            pred = model(xb)
            loss_mse = mse(pred, ydb)
            loss_sparse = pred.abs().mean()
            if coords is not None:
                loss_id = coord_moment_loss(pred, ycb, coords, temp=args.coord_temp)
            else:
                loss_id = torch.tensor(0.0, device=device)
            loss_phys = physics_loss(
                pred_delta=pred,
                target_vraw=xrawb,
                edge_n1=args.edge_n1,
                edge_n2=args.edge_n2,
                keep_idx=args.keep_idx,
                ext_nodes=args.ext_nodes,
                excitations=args.excitations,
                args=args,
            )
            loss = (
                args.lambda_mse * loss_mse
                + args.lambda_id * loss_id
                + args.lambda_phys * loss_phys
                + args.lambda_sparse * loss_sparse
            )
            val_loss += loss.item() * xb.size(0)

            sum_all_abs += torch.abs(pred - ydb).sum().item()
            n_all += ydb.numel()
            mask = ycb > 0.5
            if mask.any():
                sum_changed_abs += torch.abs(pred[mask] - ydb[mask]).sum().item()
                n_changed += int(mask.sum().item())

            hit_count, _rates = position_hit_rates(pred, tids)
            for k in hit_count_all:
                hit_count_all[k] += hit_count[k]

    n_samples = len(loader.dataset)
    pos_rates = {k: hit_count_all[k] / max(n_samples, 1) for k in hit_count_all}
    mae_all = sum_all_abs / max(n_all, 1)
    mae_changed = sum_changed_abs / max(n_changed, 1)
    val_loss = val_loss / max(n_samples, 1)
    score = mae_changed + args.score_mae_all * mae_all + args.score_pos3 * (1.0 - pos_rates[3])
    return {
        "loss": val_loss,
        "mae_all": mae_all,
        "mae_changed": mae_changed,
        "pos_hit_count": hit_count_all,
        "pos_hit_rate": pos_rates,
        "score": score,
    }


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, x_raw, y_change, y_delta, true_ids, true_vals, ext_nodes_np, ex_np = build_dataset(
        Path(args.data_path), Path(args.cache_path)
    )
    tr, va, te = split_indices(len(x), args.seed)

    mean = x[tr].mean(axis=0, keepdims=True)
    std = x[tr].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_std = ((x - mean) / std).astype(np.float32)
    np.savez_compressed(out_dir / "standardization.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))

    ds = Change3Dataset(x_std, x_raw, y_change, y_delta, true_ids, true_vals)
    tr_loader = DataLoader(Subset(ds, tr), batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(Subset(ds, va), batch_size=args.batch_size, shuffle=False)
    te_loader = DataLoader(Subset(ds, te), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Change3Regressor(
        in_dim=x_std.shape[1],
        out_dim=NUM_RESISTORS,
        dropout=args.dropout,
        max_abs=args.max_abs,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse = nn.MSELoss(reduction="mean")

    edges = build_edges()
    args.edge_n1 = torch.tensor([e[0] for e in edges], dtype=torch.long, device=device)
    args.edge_n2 = torch.tensor([e[1] for e in edges], dtype=torch.long, device=device)
    args.ext_nodes = torch.tensor(ext_nodes_np.tolist(), dtype=torch.long, device=device)
    args.excitations = torch.tensor(ex_np.tolist(), dtype=torch.long, device=device)
    args.keep_idx = {
        g: torch.tensor([i for i in range(NUM_NODES) if i != g], dtype=torch.long, device=device)
        for g in range(NUM_NODES)
    }
    coords = load_coords_tensor(args.coords_path, device)

    best_state = None
    best_score = float("inf")
    best_epoch = 0
    bad_epochs = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0

        for xb, xrawb, ycb, ydb, _tids, _tvals in tr_loader:
            xb = xb.to(device)
            xrawb = xrawb.to(device)
            ycb = ycb.to(device)
            ydb = ydb.to(device)

            pred = model(xb)
            loss_mse = mse(pred, ydb)
            loss_id = coord_moment_loss(pred, ycb, coords, temp=args.coord_temp)
            loss_phys = physics_loss(
                pred_delta=pred,
                target_vraw=xrawb,
                edge_n1=args.edge_n1,
                edge_n2=args.edge_n2,
                keep_idx=args.keep_idx,
                ext_nodes=args.ext_nodes,
                excitations=args.excitations,
                args=args,
            )
            loss_sparse = pred.abs().mean()

            loss = (
                args.lambda_mse * loss_mse
                + args.lambda_id * loss_id
                + args.lambda_phys * loss_phys
                + args.lambda_sparse * loss_sparse
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tr_loss += loss.item() * xb.size(0)

        tr_loss /= len(tr_loader.dataset)
        val = evaluate(model, va_loader, device, mse, args, coords=coords)

        if val["score"] < best_score:
            best_score = val["score"]
            best_epoch = ep
            bad_epochs = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if ep == 1 or ep % 5 == 0:
            pr = val["pos_hit_rate"]
            print(
                f"Epoch {ep:03d} | train_loss={tr_loss:.6f} | val_loss={val['loss']:.6f} "
                f"| val_mae_changed={val['mae_changed']:.4f} "
                f"| 位置准确率(0/1/2/3)={pr[0]:.3f}/{pr[1]:.3f}/{pr[2]:.3f}/{pr[3]:.3f}"
            )

        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {ep}, best epoch={best_epoch}")
            break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    torch.save(best_state, out_dir / "model_best.pt")
    torch.save(model.state_dict(), out_dir / "model_last.pt")

    model.load_state_dict(best_state)
    test = evaluate(model, te_loader, device, mse, args, coords=coords)

    metrics = {
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_score),
        "test_mae_all": float(test["mae_all"]),
        "test_mae_changed": float(test["mae_changed"]),
        "test_position_hit_count": {str(k): int(v) for k, v in test["pos_hit_count"].items()},
        "test_position_hit_rate": {str(k): float(v) for k, v in test["pos_hit_rate"].items()},
        "loss_weights": {
            "lambda_mse": float(args.lambda_mse),
            "lambda_id": float(args.lambda_id),
            "lambda_phys": float(args.lambda_phys),
            "lambda_sparse": float(args.lambda_sparse),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    pr = test["pos_hit_rate"]
    print("\nTest Metrics (Change3 Regression):")
    print(f"mae_all={test['mae_all']:.4f}")
    print(f"mae_changed={test['mae_changed']:.4f}")
    print(
        "位置准确率(对0/1/2/3个)="
        f"{pr[0]:.4f}/{pr[1]:.4f}/{pr[2]:.4f}/{pr[3]:.4f}"
    )


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Change3 重建模型 v1_new（MSE+ID+Physics+Sparse）")
    parser.add_argument("--data-path", default=str(script_dir.parent / "data_fixed" / "training_data64_fixed_3.csv"))
    parser.add_argument("--cache-path", default=str(script_dir / "cache_change3_v1_new.npz"))
    parser.add_argument("--coords-path", default=str(script_dir.parent / "data_fixed" / "resistor_coords_bl_origin.json"))
    parser.add_argument("--out-dir", default=str(script_dir / "outputs"))

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-abs", type=float, default=310.0)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260322)

    parser.add_argument("--lambda-mse", type=float, default=1.00)
    parser.add_argument("--lambda-id", type=float, default=0.35)
    parser.add_argument("--lambda-phys", type=float, default=0.15)
    parser.add_argument("--lambda-sparse", type=float, default=0.05)
    parser.add_argument("--coord-temp", type=float, default=20.0)
    parser.add_argument("--score-mae-all", type=float, default=0.15)
    parser.add_argument("--score-pos3", type=float, default=6.0)

    parser.add_argument("--current-a", type=float, default=0.005)
    parser.add_argument("--r-min", type=float, default=650.0)
    parser.add_argument("--r-max", type=float, default=1350.0)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run(args)


if __name__ == "__main__":
    main()

