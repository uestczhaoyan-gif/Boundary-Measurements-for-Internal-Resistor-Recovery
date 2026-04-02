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


def external_nodes_clockwise():
    top = list(range(0, GRID_SIZE))
    right = [r * GRID_SIZE + (GRID_SIZE - 1) for r in range(1, GRID_SIZE)]
    bottom = list(range(GRID_SIZE * GRID_SIZE - 2, GRID_SIZE * (GRID_SIZE - 1) - 1, -1))
    left = [r * GRID_SIZE for r in range(GRID_SIZE - 2, 0, -1)]
    return top + right + bottom + left


def build_excitations(ext_nodes):
    ex = []
    n = len(ext_nodes)
    for i in range(n):
        ex.append((ext_nodes[i], ext_nodes[(i + 1) % n]))
    ex.extend([(0, 63), (7, 56), (3, 60), (31, 32)])
    return ex


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


def physics_loss(pred_delta, target_vraw, ex_indices, edge_n1, edge_n2, keep_idx, ext_nodes, excitations, args):
    rvals = torch.clamp(BASE_R + pred_delta, min=args.r_min, max=args.r_max)
    g_batch = build_g_batch(rvals, edge_n1, edge_n2)

    loss = torch.tensor(0.0, device=pred_delta.device)
    mse = nn.MSELoss(reduction="mean")
    for exi in ex_indices:
        src = int(excitations[exi, 0])
        gnd = int(excitations[exi, 1])
        pred_ext = solve_ext_batch(g_batch, src, gnd, keep_idx, ext_nodes, args.current_a)
        true_ext = target_vraw[:, exi, :]
        loss = loss + mse(pred_ext, true_ext)
    return loss / max(len(ex_indices), 1)


def top3_order_loss(pred_delta, t_on=65.0, t_off=45.0):
    s, _ = torch.sort(pred_delta.abs(), dim=1, descending=True)
    s2 = s[:, 2]
    s3 = s[:, 3]
    return (torch.relu(t_on - s2) + torch.relu(s3 - t_off)).mean()


def evaluate(model, loader, device, huber, args):
    model.eval()
    val_loss = 0.0
    sum_all_abs = 0.0
    n_all = 0
    sum_changed_abs = 0.0
    n_changed = 0
    gt50_counts = []
    top3_hit_sum = 0.0
    n_samples = 0

    with torch.no_grad():
        for xb, _xraw, ycb, ydb, tids, _tvals in loader:
            xb, ycb, ydb, tids = xb.to(device), ycb.to(device), ydb.to(device), tids.to(device)
            pred = model(xb)

            mask = ycb > 0.5
            if mask.any():
                loss_change = huber(pred[mask], ydb[mask]).mean()
                sum_changed_abs += torch.abs(pred[mask] - ydb[mask]).sum().item()
                n_changed += int(mask.sum().item())
            else:
                loss_change = huber(pred, ydb).mean()

            if (~mask).any():
                loss_unchange = torch.abs(pred[~mask]).mean()
                hinge = torch.relu(pred[~mask].abs() - args.hinge_threshold)
                loss_hinge = (hinge * hinge).mean()
            else:
                loss_unchange = torch.tensor(0.0, device=device)
                loss_hinge = torch.tensor(0.0, device=device)

            loss_sparse = pred.abs().mean()
            loss_order = top3_order_loss(pred, t_on=args.top3_on, t_off=args.top3_off)

            loss = (
                args.w_change * loss_change
                + args.w_unchange * loss_unchange
                + args.lambda_hinge * loss_hinge
                + args.lambda_sparse * loss_sparse
                + args.lambda_top3 * loss_order
            )
            val_loss += loss.item() * xb.size(0)

            sum_all_abs += torch.abs(pred - ydb).sum().item()
            n_all += ydb.numel()
            gt50_counts.extend((pred.abs() > args.eval_threshold).sum(dim=1).cpu().tolist())

            pred_top3 = torch.topk(pred.abs(), k=3, dim=1).indices
            for i in range(pred_top3.size(0)):
                pset = set(pred_top3[i].cpu().tolist())
                tset = set(tids[i].cpu().tolist())
                top3_hit_sum += len(pset.intersection(tset)) / 3.0
                n_samples += 1

    val_loss /= len(loader.dataset)
    mae_all = sum_all_abs / max(n_all, 1)
    mae_changed = sum_changed_abs / max(n_changed, 1)
    avg_gt50 = float(np.mean(gt50_counts)) if gt50_counts else 0.0
    top3_precision = top3_hit_sum / max(n_samples, 1)
    score = mae_changed + args.score_mae_all * mae_all + args.score_sparse * abs(avg_gt50 - 3.0)
    return {
        "loss": val_loss,
        "mae_all": mae_all,
        "mae_changed": mae_changed,
        "avg_gt": avg_gt50,
        "top3_precision": top3_precision,
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
    model = Change3Regressor(in_dim=x_std.shape[1], out_dim=NUM_RESISTORS, dropout=args.dropout, max_abs=args.max_abs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    huber = nn.SmoothL1Loss(reduction="none")

    edges = build_edges()
    edge_n1 = torch.tensor([e[0] for e in edges], dtype=torch.long, device=device)
    edge_n2 = torch.tensor([e[1] for e in edges], dtype=torch.long, device=device)
    ext_nodes = torch.tensor(ext_nodes_np.tolist(), dtype=torch.long, device=device)
    excitations = torch.tensor(ex_np.tolist(), dtype=torch.long, device=device)
    keep_idx = {
        g: torch.tensor([i for i in range(NUM_NODES) if i != g], dtype=torch.long, device=device)
        for g in range(NUM_NODES)
    }

    phys_k_choices = [int(x.strip()) for x in args.phys_k_choices.split(",") if x.strip()]
    if not phys_k_choices:
        phys_k_choices = [4]

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
            mask = ycb > 0.5

            if mask.any():
                loss_change = huber(pred[mask], ydb[mask]).mean()
            else:
                loss_change = huber(pred, ydb).mean()

            if (~mask).any():
                loss_unchange = torch.abs(pred[~mask]).mean()
                hinge = torch.relu(pred[~mask].abs() - args.hinge_threshold)
                loss_hinge = (hinge * hinge).mean()
            else:
                loss_unchange = torch.tensor(0.0, device=device)
                loss_hinge = torch.tensor(0.0, device=device)

            loss_sparse = pred.abs().mean()
            loss_order = top3_order_loss(pred, t_on=args.top3_on, t_off=args.top3_off)

            if ep >= args.phys_start_epoch and args.lambda_phys > 0:
                k_pick = random.choice(phys_k_choices)
                ex_sel = random.sample(range(EXCITATIONS), k=min(k_pick, EXCITATIONS))
                loss_phys = physics_loss(
                    pred_delta=pred,
                    target_vraw=xrawb,
                    ex_indices=ex_sel,
                    edge_n1=edge_n1,
                    edge_n2=edge_n2,
                    keep_idx=keep_idx,
                    ext_nodes=ext_nodes,
                    excitations=excitations,
                    args=args,
                )
            else:
                loss_phys = torch.tensor(0.0, device=device)

            loss = (
                args.w_change * loss_change
                + args.w_unchange * loss_unchange
                + args.lambda_hinge * loss_hinge
                + args.lambda_sparse * loss_sparse
                + args.lambda_top3 * loss_order
                + args.lambda_phys * loss_phys
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tr_loss += loss.item() * xb.size(0)

        tr_loss /= len(tr_loader.dataset)
        val = evaluate(model, va_loader, device, huber, args)

        if val["score"] < best_score:
            best_score = val["score"]
            best_epoch = ep
            bad_epochs = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if ep == 1 or ep % 5 == 0:
            print(
                f"Epoch {ep:03d} | train_loss={tr_loss:.6f} | val_loss={val['loss']:.6f} "
                f"| val_mae_changed={val['mae_changed']:.4f} | val_avg(|dR|>{args.eval_threshold})={val['avg_gt']:.2f}"
            )

        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {ep}, best epoch={best_epoch}")
            break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    torch.save(best_state, out_dir / "model_best.pt")
    torch.save(model.state_dict(), out_dir / "model_last.pt")

    model.load_state_dict(best_state)
    test = evaluate(model, te_loader, device, huber, args)

    metrics = {
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_score),
        "test_mae_all": float(test["mae_all"]),
        "test_mae_changed": float(test["mae_changed"]),
        "test_avg_gt_threshold": float(test["avg_gt"]),
        "test_top3_id_precision": float(test["top3_precision"]),
        "eval_threshold": float(args.eval_threshold),
        "physics": {
            "enabled_after_epoch": int(args.phys_start_epoch),
            "lambda_phys": float(args.lambda_phys),
            "phys_k_choices": phys_k_choices,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nTest Metrics (Change3 Regression):")
    print(f"mae_all={test['mae_all']:.4f}")
    print(f"mae_changed={test['mae_changed']:.4f}")
    print(f"avg(|dR|>{args.eval_threshold})={test['avg_gt']:.2f}")
    print(f"top3_id_precision={test['top3_precision']:.4f}")


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Change3 reconstruction baseline (v1).")
    parser.add_argument("--data-path", default=str(script_dir.parent / "data_fixed" / "training_data64_fixed_3.csv"))
    parser.add_argument("--cache-path", default=str(script_dir / "cache_change3_v1.npz"))
    parser.add_argument("--out-dir", default=str(script_dir / "outputs"))

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-abs", type=float, default=320.0)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260321)

    parser.add_argument("--w-change", type=float, default=2.0)
    parser.add_argument("--w-unchange", type=float, default=1.3)
    parser.add_argument("--lambda-hinge", type=float, default=0.30)
    parser.add_argument("--lambda-sparse", type=float, default=0.015)
    parser.add_argument("--lambda-top3", type=float, default=0.20)
    parser.add_argument("--hinge-threshold", type=float, default=45.0)
    parser.add_argument("--top3-on", type=float, default=65.0)
    parser.add_argument("--top3-off", type=float, default=45.0)

    parser.add_argument("--phys-start-epoch", type=int, default=80)
    parser.add_argument("--lambda-phys", type=float, default=0.08)
    parser.add_argument("--phys-k-choices", default="4,8")
    parser.add_argument("--current-a", type=float, default=0.005)
    parser.add_argument("--r-min", type=float, default=650.0)
    parser.add_argument("--r-max", type=float, default=1350.0)

    parser.add_argument("--eval-threshold", type=float, default=50.0)
    parser.add_argument("--score-mae-all", type=float, default=0.15)
    parser.add_argument("--score-sparse", type=float, default=1.5)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run(args)


if __name__ == "__main__":
    main()

