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

from model.model import Change3Regressor


GRID_SIZE = 8
NUM_NODES = GRID_SIZE * GRID_SIZE
NUM_RESISTORS = (GRID_SIZE * (GRID_SIZE - 1)) * 2
BASE_R = 1000.0
EXCITATIONS = 32
DEFAULT_FIXED_K = 3
DATA_DIRNAME = "data_fixed"
DATA_PREFIX = "training_data64_fixed"
SUPPORTED_FIXED_K = (2, 3)


def sanitize_dataset_tag(raw_tag):
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", raw_tag.strip())
    safe = safe.strip("._-")
    return safe or "dataset"


def validate_fixed_k(fixed_k, source_desc):
    fixed_k = int(fixed_k)
    if fixed_k not in SUPPORTED_FIXED_K:
        supported = "/".join(str(k) for k in SUPPORTED_FIXED_K)
        raise ValueError(f"{source_desc} only supports fixed_k in {{{supported}}}, got {fixed_k}.")
    return fixed_k


def fixed_scope_name(fixed_k):
    return f"fixed_{int(fixed_k)}"


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


def default_fixed_dataset_info(data_path):
    stem = Path(data_path).stem
    m = re.fullmatch(rf"{re.escape(DATA_PREFIX)}_(\d+)(?:_(.+))?", stem)
    if m:
        fixed_k = validate_fixed_k(int(m.group(1)), f"Dataset stem '{stem}'")
        raw_tag = m.group(2)
        dataset_tag = sanitize_dataset_tag(raw_tag) if raw_tag else "5mA"
        return fixed_k, dataset_tag
    return DEFAULT_FIXED_K, sanitize_dataset_tag(stem)


def resolve_fixed_runtime_paths(args, script_dir, default_cache_name):
    args.fixed_k = validate_fixed_k(args.fixed_k, "Argument --fixed-k")
    default_data_path = script_dir.parent / DATA_DIRNAME / f"{DATA_PREFIX}_{int(args.fixed_k)}.csv"
    raw_data_path = resolve_input_data_path(args.data_path, script_dir)
    raw_tag = sanitize_dataset_tag(args.dataset_tag) if args.dataset_tag else ""
    if raw_data_path == default_data_path and raw_tag:
        candidate = default_data_path.with_name(f"{DATA_PREFIX}_{int(args.fixed_k)}_{raw_tag}.csv")
        data_path = candidate if candidate.exists() else default_data_path
    else:
        data_path = raw_data_path
    data_path = data_path.resolve()

    parsed_fixed_k, parsed_dataset_tag = default_fixed_dataset_info(data_path)
    dataset_tag = raw_tag or parsed_dataset_tag
    fixed_scope = fixed_scope_name(parsed_fixed_k)

    default_cache_path = script_dir / default_cache_name
    cache_path = Path(args.cache_path)
    if cache_path == default_cache_path:
        cache_path = script_dir / "cache" / fixed_scope / dataset_tag / default_cache_name

    default_out_dir = script_dir / "outputs"
    out_dir = Path(args.out_dir)
    if out_dir == default_out_dir:
        out_dir = default_out_dir / fixed_scope / dataset_tag

    args.data_path = str(data_path)
    args.cache_path = str(cache_path)
    args.out_dir = str(out_dir)
    args.dataset_tag = dataset_tag
    args.fixed_k = parsed_fixed_k
    args.fixed_scope = fixed_scope


def sync_current_from_meta(args):
    data_path = Path(args.data_path)
    meta_path = data_path.with_name(f"{data_path.stem}_meta.json")
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_current = meta.get("current_source_a")
    if meta_current is None:
        meta_current = None
    if meta_current is not None:
        meta_current = float(meta_current)
        if abs(meta_current - args.current_a) > 1e-12:
            print(f"[Meta] current_a override: {args.current_a} -> {meta_current} ({meta_path.name})")
        args.current_a = meta_current

    meta_fixed_k = meta.get("change_count_fixed")
    if meta_fixed_k is not None:
        meta_fixed_k = validate_fixed_k(int(meta_fixed_k), f"Meta '{meta_path.name}'")
        if meta_fixed_k != int(args.fixed_k):
            raise RuntimeError(
                f"fixed_k mismatch: args/data resolved to {args.fixed_k}, but meta '{meta_path.name}' says {meta_fixed_k}."
            )
        args.fixed_k = meta_fixed_k
        args.fixed_scope = fixed_scope_name(args.fixed_k)


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


def build_dataset(csv_path, cache_path, expected_fixed_k=None):
    csv_path = Path(csv_path).resolve()
    cache_path = Path(cache_path)
    if expected_fixed_k is not None:
        expected_fixed_k = validate_fixed_k(expected_fixed_k, "Expected fixed_k")
    if cache_path.exists():
        d = np.load(cache_path)
        fixed_k = int(d["fixed_k"][0]) if "fixed_k" in d.files else int(d["true_ids"].shape[1])
        if expected_fixed_k is not None and fixed_k != expected_fixed_k:
            raise RuntimeError(
                f"Cache fixed_k mismatch: cache={fixed_k}, expected={expected_fixed_k}. "
                f"Please clear cache or use a different dataset-tag/cache-path. ({cache_path})"
            )
        if "source_csv" in d.files:
            cache_source = Path(str(d["source_csv"][0])).resolve()
            if cache_source != csv_path:
                raise RuntimeError(
                    f"Cache source mismatch: cache built from '{cache_source}', current data is '{csv_path}'. "
                    f"Please clear cache or use a different dataset-tag/cache-path. ({cache_path})"
                )
        return (
            d["x"],
            d["x_raw"],
            d["y_change"],
            d["y_delta"],
            d["true_ids"],
            d["true_vals"],
            d["ext_nodes"],
            d["excitations"],
            fixed_k,
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
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
        resistor_slots = sorted(
            int(re.fullmatch(r"r(\d+)_id", c).group(1))
            for c in reader.fieldnames
            if re.fullmatch(r"r(\d+)_id", c)
        )
        if not resistor_slots:
            raise RuntimeError("No resistor id columns found in fixed-change dataset.")

        prev_combo = None
        combo_rows = []
        combo_ex = []
        y_change = None
        y_delta = None
        true_ids = None
        true_vals = None
        ex_ref = None
        dataset_fixed_k = None

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
                change_count = int(row["change_count"])
                if dataset_fixed_k is None:
                    dataset_fixed_k = validate_fixed_k(change_count, f"Dataset '{csv_path.name}'")
                elif dataset_fixed_k != change_count:
                    raise RuntimeError("Mixed fixed-k values found in one dataset.")
                if dataset_fixed_k <= 0:
                    raise RuntimeError("change_count must be positive for fixed-change dataset.")
                y_change = np.zeros(NUM_RESISTORS, dtype=np.float32)
                y_delta = np.zeros(NUM_RESISTORS, dtype=np.float32)
                true_ids = np.zeros(dataset_fixed_k, dtype=np.int64)
                true_vals = np.zeros(dataset_fixed_k, dtype=np.float32)
                if len(resistor_slots) < dataset_fixed_k:
                    raise RuntimeError("Dataset header does not contain enough resistor slots.")
                for i in range(1, dataset_fixed_k + 1):
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
    if expected_fixed_k is not None and int(dataset_fixed_k) != int(expected_fixed_k):
        raise RuntimeError(
            f"Dataset fixed_k mismatch: data='{csv_path.name}' gives {dataset_fixed_k}, expected {expected_fixed_k}."
        )

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
        fixed_k=np.array([dataset_fixed_k], dtype=np.int64),
        source_csv=np.array([str(csv_path)], dtype=np.str_),
    )
    return x, x_raw, y_change, y_delta, true_ids, true_vals, ext_nodes, excitations, int(dataset_fixed_k)


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


def weighted_regression_loss(pred_delta, true_delta, y_change, w_change, w_unchange):
    weights = torch.where(
        y_change > 0.5,
        torch.full_like(y_change, w_change),
        torch.full_like(y_change, w_unchange),
    )
    se = (pred_delta - true_delta).pow(2)
    return (se * weights).sum() / weights.sum().clamp_min(1.0)


def next_rank_losses(pred_delta, fixed_k, threshold, gap_margin):
    topk_abs = torch.topk(pred_delta.abs(), k=min(fixed_k + 1, pred_delta.shape[1]), dim=1).values
    kept_abs = topk_abs[:, fixed_k - 1]
    next_abs = topk_abs[:, fixed_k]
    loss_next = torch.relu(next_abs - threshold).pow(2).mean()
    loss_gap = torch.relu(gap_margin - (kept_abs - next_abs)).pow(2).mean()
    return loss_next, loss_gap


def physics_scale_for_epoch(epoch, args):
    if epoch < args.phys_start_epoch:
        return 0.0
    if args.phys_ramp_epochs <= 1:
        return 1.0
    return min(1.0, float(epoch - args.phys_start_epoch + 1) / float(args.phys_ramp_epochs))


def position_hit_rates(pred_delta, true_ids, fixed_k):
    pred_topk = torch.topk(pred_delta.abs(), k=fixed_k, dim=1).indices
    hit_count = {k: 0 for k in range(fixed_k + 1)}
    n = pred_topk.size(0)
    for i in range(n):
        pset = set(pred_topk[i].cpu().tolist())
        tset = set(true_ids[i].cpu().tolist())
        hit = len(pset.intersection(tset))
        hit_count[hit] += 1
    rates = {k: hit_count[k] / max(n, 1) for k in hit_count}
    return hit_count, rates


def evaluate(model, loader, device, args, coords=None, phys_scale=1.0):
    model.eval()
    val_loss = 0.0
    sum_all_abs = 0.0
    n_all = 0
    sum_changed_abs = 0.0
    n_changed = 0
    sum_active_count = 0.0
    hit_count_all = {k: 0 for k in range(args.fixed_k + 1)}

    with torch.no_grad():
        for xb, xrawb, ycb, ydb, tids, _tvals in loader:
            xb = xb.to(device)
            xrawb = xrawb.to(device)
            ycb = ycb.to(device)
            ydb = ydb.to(device)
            tids = tids.to(device)

            pred = model(xb)
            loss_reg = weighted_regression_loss(pred, ydb, ycb, args.w_change, args.w_unchange)
            if coords is not None and args.lambda_id > 0.0:
                loss_id = coord_moment_loss(pred, ycb, coords, temp=args.coord_temp)
            else:
                loss_id = torch.tensor(0.0, device=device)
            if phys_scale > 0.0 and args.lambda_phys > 0.0:
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
            else:
                loss_phys = torch.tensor(0.0, device=device)
            loss_fp_next, loss_rank_gap = next_rank_losses(
                pred,
                fixed_k=args.fixed_k,
                threshold=args.fp_next_threshold,
                gap_margin=args.rank_gap_margin,
            )
            loss = (
                args.lambda_reg * loss_reg
                + args.lambda_id * loss_id
                + args.lambda_phys * phys_scale * loss_phys
                + args.lambda_fp_next * loss_fp_next
                + args.lambda_rank_gap * loss_rank_gap
            )
            val_loss += loss.item() * xb.size(0)

            sum_all_abs += torch.abs(pred - ydb).sum().item()
            n_all += ydb.numel()
            mask = ycb > 0.5
            if mask.any():
                sum_changed_abs += torch.abs(pred[mask] - ydb[mask]).sum().item()
                n_changed += int(mask.sum().item())
            sum_active_count += (pred.abs() > args.report_threshold).float().sum(dim=1).sum().item()

            hit_count, _rates = position_hit_rates(pred, tids, args.fixed_k)
            for k in hit_count_all:
                hit_count_all[k] += hit_count[k]

    n_samples = len(loader.dataset)
    pos_rates = {k: hit_count_all[k] / max(n_samples, 1) for k in hit_count_all}
    mae_all = sum_all_abs / max(n_all, 1)
    mae_changed = sum_changed_abs / max(n_changed, 1)
    avg_active_count = sum_active_count / max(n_samples, 1)
    val_loss = val_loss / max(n_samples, 1)
    active_excess = max(0.0, avg_active_count - args.target_active_count)
    full_hit_key = int(args.fixed_k)
    score = (
        mae_changed
        + args.score_avg50_excess * active_excess
        + args.score_pos_hit * (1.0 - pos_rates[full_hit_key])
        + args.score_mae_all * mae_all
    )
    return {
        "loss": val_loss,
        "mae_all": mae_all,
        "mae_changed": mae_changed,
        "avg_active_count": avg_active_count,
        "pos_hit_count": hit_count_all,
        "pos_hit_rate": pos_rates,
        "score": score,
    }


def run(args):
    out_dir = Path(args.out_dir)
    cache_path = Path(args.cache_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "[Runtime] "
        f"fixed_scope={args.fixed_scope} | dataset_tag={args.dataset_tag} | fixed_k={args.fixed_k} | "
        f"data_path={Path(args.data_path)} | cache_path={cache_path} | out_dir={out_dir}"
    )

    x, x_raw, y_change, y_delta, true_ids, true_vals, ext_nodes_np, ex_np, data_fixed_k = build_dataset(
        Path(args.data_path), cache_path, expected_fixed_k=args.fixed_k
    )
    if int(data_fixed_k) != int(args.fixed_k):
        raise RuntimeError(f"Resolved fixed_k={args.fixed_k}, but loaded data fixed_k={data_fixed_k}.")
    args.fixed_k = int(data_fixed_k)
    if args.target_active_count is None:
        args.target_active_count = float(args.fixed_k)
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
        phys_scale = physics_scale_for_epoch(ep, args)

        for xb, xrawb, ycb, ydb, _tids, _tvals in tr_loader:
            xb = xb.to(device)
            xrawb = xrawb.to(device)
            ycb = ycb.to(device)
            ydb = ydb.to(device)

            pred = model(xb)
            loss_reg = weighted_regression_loss(pred, ydb, ycb, args.w_change, args.w_unchange)
            loss_id = coord_moment_loss(pred, ycb, coords, temp=args.coord_temp)
            if phys_scale > 0.0 and args.lambda_phys > 0.0:
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
            else:
                loss_phys = torch.tensor(0.0, device=device)
            loss_fp_next, loss_rank_gap = next_rank_losses(
                pred,
                fixed_k=args.fixed_k,
                threshold=args.fp_next_threshold,
                gap_margin=args.rank_gap_margin,
            )

            loss = (
                args.lambda_reg * loss_reg
                + args.lambda_id * loss_id
                + args.lambda_phys * phys_scale * loss_phys
                + args.lambda_fp_next * loss_fp_next
                + args.lambda_rank_gap * loss_rank_gap
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tr_loss += loss.item() * xb.size(0)

        tr_loss /= len(tr_loader.dataset)
        val = evaluate(model, va_loader, device, args, coords=coords, phys_scale=phys_scale)

        if val["score"] < best_score:
            best_score = val["score"]
            best_epoch = ep
            bad_epochs = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if ep == 1 or ep % 5 == 0:
            pr = val["pos_hit_rate"]
            pos_text = "/".join(f"{pr[k]:.3f}" for k in range(args.fixed_k + 1))
            print(
                f"Epoch {ep:03d} | train_loss={tr_loss:.6f} | val_loss={val['loss']:.6f} "
                f"| val_mae_changed={val['mae_changed']:.4f} "
                f"| val_avg(|dR|>{args.report_threshold:.0f})={val['avg_active_count']:.2f} "
                f"| phys_scale={phys_scale:.2f} "
                f"| 位置准确率(0..{args.fixed_k})={pos_text}"
            )

        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {ep}, best epoch={best_epoch}")
            break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    torch.save(best_state, out_dir / "model_best.pt")
    torch.save(model.state_dict(), out_dir / "model_last.pt")

    model.load_state_dict(best_state)
    test = evaluate(model, te_loader, device, args, coords=coords, phys_scale=1.0)

    metrics = {
        "dataset_tag": args.dataset_tag,
        "fixed_k": int(args.fixed_k),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_score),
        "test_mae_all": float(test["mae_all"]),
        "test_mae_changed": float(test["mae_changed"]),
        "test_avg_active_count": float(test["avg_active_count"]),
        "test_position_hit_count": {str(k): int(v) for k, v in test["pos_hit_count"].items()},
        "test_position_hit_rate": {str(k): float(v) for k, v in test["pos_hit_rate"].items()},
        "loss_weights": {
            "lambda_reg": float(args.lambda_reg),
            "w_change": float(args.w_change),
            "w_unchange": float(args.w_unchange),
            "lambda_id": float(args.lambda_id),
            "lambda_phys": float(args.lambda_phys),
            "lambda_fp_next": float(args.lambda_fp_next),
            "fp_next_threshold": float(args.fp_next_threshold),
            "lambda_rank_gap": float(args.lambda_rank_gap),
            "rank_gap_margin": float(args.rank_gap_margin),
            "phys_start_epoch": int(args.phys_start_epoch),
            "phys_ramp_epochs": int(args.phys_ramp_epochs),
            "report_threshold": float(args.report_threshold),
            "target_active_count": float(args.target_active_count),
            "score_avg50_excess": float(args.score_avg50_excess),
            "score_pos_hit": float(args.score_pos_hit),
            "score_mae_all": float(args.score_mae_all),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    pr = test["pos_hit_rate"]
    pos_text = "/".join(f"{pr[k]:.4f}" for k in range(args.fixed_k + 1))
    print("\nTest Metrics (Fixed-change Regression):")
    print(f"mae_all={test['mae_all']:.4f}")
    print(f"mae_changed={test['mae_changed']:.4f}")
    print(f"avg(|dR|>{args.report_threshold:.0f})={test['avg_active_count']:.4f}")
    print(f"位置准确率(对0..{args.fixed_k}个)={pos_text}")


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Fixed-change 重建模型 v3_new（简化 loss 版）")
    parser.add_argument("--data-path", default=str(script_dir.parent / DATA_DIRNAME / f"{DATA_PREFIX}_{DEFAULT_FIXED_K}.csv"))
    parser.add_argument("--dataset-tag", default="", help="数据集标签；默认按 fixed 数据文件名推导。")
    parser.add_argument("--fixed-k", type=int, default=DEFAULT_FIXED_K, help="固定变化数量；当前只支持 2 或 3。")
    parser.add_argument("--cache-path", default=str(script_dir / "cache_fixed_v3_new.npz"))
    parser.add_argument("--coords-path", default=str(script_dir.parent / DATA_DIRNAME / "resistor_coords_bl_origin.json"))
    parser.add_argument("--out-dir", default=str(script_dir / "outputs"))

    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-abs", type=float, default=310.0)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260322)

    parser.add_argument("--lambda-reg", type=float, default=1.00)
    parser.add_argument("--w-change", type=float, default=7.0)
    parser.add_argument("--w-unchange", type=float, default=1.0)
    parser.add_argument("--lambda-id", type=float, default=0.35)
    parser.add_argument("--lambda-phys", type=float, default=0.15)
    parser.add_argument("--lambda-fp-next", "--lambda-fp4", dest="lambda_fp_next", type=float, default=0.30)
    parser.add_argument("--fp-next-threshold", "--fp4-threshold", dest="fp_next_threshold", type=float, default=45.0)
    parser.add_argument("--lambda-rank-gap", type=float, default=0.18)
    parser.add_argument("--rank-gap-margin", type=float, default=14.0)
    parser.add_argument("--coord-temp", type=float, default=20.0)
    parser.add_argument("--score-avg50-excess", type=float, default=2.5)
    parser.add_argument("--score-pos-hit", type=float, default=8.0)
    parser.add_argument("--score-mae-all", type=float, default=0.05)
    parser.add_argument("--report-threshold", type=float, default=50.0)
    parser.add_argument("--target-active-count", type=float, default=None)

    parser.add_argument("--current-a", type=float, default=0.005)
    parser.add_argument("--r-min", type=float, default=650.0)
    parser.add_argument("--r-max", type=float, default=1350.0)
    parser.add_argument("--phys-start-epoch", type=int, default=25)
    parser.add_argument("--phys-ramp-epochs", type=int, default=20)

    args = parser.parse_args()
    resolve_fixed_runtime_paths(args, script_dir, "cache_fixed_v3_new.npz")
    sync_current_from_meta(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run(args)


if __name__ == "__main__":
    main()

