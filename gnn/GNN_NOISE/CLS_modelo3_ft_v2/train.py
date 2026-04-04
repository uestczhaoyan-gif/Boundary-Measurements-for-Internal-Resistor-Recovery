import argparse
import csv
import json
import math
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
from torch.utils.data import DataLoader, Dataset

try:
    from scipy.special import expit
except Exception:
    def expit(x):
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x, dtype=np.float64)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[~pos])
        out[~pos] = exp_x / (1.0 + exp_x)
        return out

from model.model import PhysicsInformedGNNClassifier


NUM_CLASSES = 4
EXCITATIONS = 32
GRID = 8
DEFAULT_MAIN_DATA_PATH = "../../../data/training_data64Nodes_2.csv"
DEFAULT_PRETRAINED_MODEL_PATH = "../CLS_modelo3_ft/outputs/training_data64Nodes_2_noiseft_rand_boundary_20260401/model_last.pt"


def ensure_cli_option_has_value(argv, option_name):
    for idx, token in enumerate(argv[1:], start=1):
        if token != option_name:
            continue
        next_idx = idx + 1
        if next_idx >= len(argv):
            raise SystemExit(
                f"{Path(argv[0]).name}: {option_name} is missing a value. "
                "If you used TAG, make sure it has been set before running this command."
            )
        next_token = argv[next_idx]
        if next_token == "" or next_token.startswith("-"):
            raise SystemExit(
                f"{Path(argv[0]).name}: {option_name} did not receive a usable value. "
                "If you used ${TAG}, it likely expanded to an empty string."
            )


def validate_dataset_tag_arg(raw_tag):
    if raw_tag and any(ch in raw_tag for ch in "$ {}"):
        raise SystemExit(
            f"{Path(sys.argv[0]).name}: --dataset-tag looks like an unexpanded placeholder: {raw_tag!r}. "
            "Please replace it with the real run tag."
        )


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def resolve_runtime_path(raw_path, script_dir):
    if not raw_path:
        return ""
    path = Path(raw_path)
    if path.is_absolute():
        return str(path.resolve())
    project_root = script_dir.parents[2]
    candidates = [path, script_dir / path, project_root / path]
    for candidate in candidates:
        if candidate.exists() or candidate.parent.exists():
            return str(candidate.resolve())
    return str((script_dir / path).resolve())


def resolve_dataset_runtime_paths(args, script_dir, default_cache_path):
    data_path = resolve_input_data_path(args.data_path, script_dir)
    validate_dataset_tag_arg(args.dataset_tag)
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
    args.pretrained_model_path = resolve_runtime_path(args.pretrained_model_path, script_dir)


def parse_float_list(raw_value, fallback):
    if raw_value is None:
        return list(fallback)
    text = str(raw_value).strip()
    if not text:
        return list(fallback)
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    return values or list(fallback)


def normalize_probs(values):
    arr = np.asarray(values, dtype=np.float32)
    arr = np.where(arr < 0, 0.0, arr)
    total = float(arr.sum())
    if total <= 0:
        arr = np.ones_like(arr, dtype=np.float32)
        total = float(arr.sum())
    return (arr / total).astype(np.float32)


class ClsDataset(Dataset):
    def __init__(
        self,
        x,
        y,
        ext_nodes,
        add_noise=False,
        is_train=False,
        noise_schedule="random",
        noise_std_max=0.1,
        fixed_noise_std=0.1,
        noise_mode="gaussian",
        noise_scope="boundary",
        curriculum_stds="0.01,0.0316227766,0.1",
        curriculum_probs="0.5,0.35,0.15",
        clean_mix_prob=0.15,
        structured_iid_ratio=0.55,
        structured_drift_ratio=0.25,
        structured_common_ratio=0.15,
        structured_bad_ratio=0.20,
        structured_bad_prob=0.06,
    ):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        self.ext_nodes = torch.as_tensor(ext_nodes, dtype=torch.long)
        self.add_noise = bool(add_noise)
        self.is_train = bool(is_train)
        self.noise_schedule = str(noise_schedule)
        self.noise_std_max = float(noise_std_max)
        self.fixed_noise_std = float(fixed_noise_std)
        self.noise_mode = str(noise_mode)
        self.noise_scope = str(noise_scope)
        curriculum_stds = parse_float_list(curriculum_stds, [0.01, 0.0316227766, 0.1])
        curriculum_probs = parse_float_list(curriculum_probs, [0.5, 0.35, 0.15])
        if len(curriculum_probs) != len(curriculum_stds):
            curriculum_probs = [1.0] * len(curriculum_stds)
        self.curriculum_stds = torch.tensor(curriculum_stds, dtype=torch.float32)
        self.curriculum_probs = torch.tensor(normalize_probs(curriculum_probs), dtype=torch.float32)
        self.clean_mix_prob = float(clean_mix_prob)
        self.structured_iid_ratio = float(structured_iid_ratio)
        self.structured_drift_ratio = float(structured_drift_ratio)
        self.structured_common_ratio = float(structured_common_ratio)
        self.structured_bad_ratio = float(structured_bad_ratio)
        self.structured_bad_prob = float(structured_bad_prob)

    def _sample_noise_std(self):
        if self.clean_mix_prob > 0 and torch.rand(1).item() < self.clean_mix_prob:
            return 0.0
        if self.noise_schedule == "fixed":
            return self.fixed_noise_std
        if self.noise_schedule == "curriculum":
            idx = int(torch.multinomial(self.curriculum_probs, 1, replacement=True).item())
            base_std = float(self.curriculum_stds[idx].item())
            return base_std * (0.85 + 0.30 * torch.rand(1).item())
        return self.noise_std_max * torch.rand(1).item()

    def _sample_noise(self, voltage_view, noise_std):
        if self.noise_mode == "uniform":
            amplitude = math.sqrt(3.0) * noise_std
            return torch.empty_like(voltage_view).uniform_(-amplitude, amplitude)
        if self.noise_mode != "structured":
            return torch.randn_like(voltage_view) * noise_std

        if voltage_view.ndim != 2:
            return torch.randn_like(voltage_view) * noise_std

        num_exc, num_pos = voltage_view.shape
        iid = torch.randn_like(voltage_view) * noise_std * self.structured_iid_ratio
        drift = torch.randn((1, num_pos), dtype=voltage_view.dtype, device=voltage_view.device)
        drift = drift * noise_std * self.structured_drift_ratio
        common = torch.randn((num_exc, 1), dtype=voltage_view.dtype, device=voltage_view.device)
        common = common * noise_std * self.structured_common_ratio
        bad_mask = (
            torch.rand((1, num_pos), dtype=voltage_view.dtype, device=voltage_view.device) < self.structured_bad_prob
        ).to(voltage_view.dtype)
        bad = torch.randn((1, num_pos), dtype=voltage_view.dtype, device=voltage_view.device)
        bad = bad * bad_mask * noise_std * self.structured_bad_ratio
        return iid + drift + common + bad

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx].clone()
        if self.is_train and self.add_noise:
            noise_std = self._sample_noise_std()
            if noise_std <= 0:
                return x_sample, self.y[idx]
            if self.noise_scope == "all":
                voltage_view = x_sample[:, :, 2]
            else:
                voltage_view = x_sample[:, self.ext_nodes, 2]
            noise = self._sample_noise(voltage_view, noise_std)
            if self.noise_scope == "all":
                x_sample[:, :, 2] = voltage_view + noise
            else:
                x_sample[:, self.ext_nodes, 2] = voltage_view + noise
        return x_sample, self.y[idx]


def coral_targets(labels, num_classes=NUM_CLASSES):
    thr = torch.arange(num_classes - 1, device=labels.device).view(1, -1)
    return (labels.view(-1, 1) > thr).float()


def coral_loss(logits, labels, sample_w=None):
    tgt = coral_targets(labels)
    loss = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
    if sample_w is not None:
        loss = loss * sample_w.view(-1, 1)
    return loss.mean()


def supervised_contrastive_loss(features, labels, temperature=0.12):
    if features.size(0) < 2:
        return features.new_zeros(())

    features = F.normalize(features, dim=-1)
    labels = labels.view(-1)
    logits = features @ features.t() / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    valid_anchor = ((labels == 2) | (labels == 3))
    pos_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & (~eye)

    exp_logits = torch.exp(logits) * (~eye).float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    pos_count = pos_mask.sum(dim=1)
    anchor_mask = valid_anchor & (pos_count > 0)
    if not anchor_mask.any():
        return features.new_zeros(())
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_count.clamp_min(1)
    return -mean_log_prob_pos[anchor_mask].mean()


def parse_voltage_columns(fieldnames):
    v_cols = [c for c in fieldnames if c.startswith("v_node")]
    ext_nodes = [int(c.replace("v_node", "")) for c in v_cols]
    return v_cols, np.array(ext_nodes, dtype=np.int64)


def to_graph_input(x_delta, ext_nodes, src_nodes, gnd_nodes):
    n = x_delta.shape[0]
    graphs = np.zeros((n, EXCITATIONS, GRID * GRID, 4), dtype=np.float32)
    graphs[:, :, ext_nodes, 3] = 1.0
    graphs[:, :, ext_nodes, 2] = x_delta

    ex_ids = np.arange(EXCITATIONS, dtype=np.int64)
    for i in range(n):
        graphs[i, ex_ids, src_nodes, 0] = 1.0
        graphs[i, ex_ids, gnd_nodes, 1] = 1.0
    return graphs


def build_dataset(csv_path, cache_path):
    if cache_path.exists():
        d = np.load(cache_path)
        return d["x"], d["y"], d["ext_nodes"]

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
    x = to_graph_input(x_delta, ext_nodes, src_nodes, gnd_nodes)
    y = np.array(y_list, dtype=np.int64)
    np.savez_compressed(cache_path, x=x, y=y, ext_nodes=ext_nodes)
    return x, y, ext_nodes


def standardize_graph_voltage(x, mean, std, ext_nodes):
    x_std = x.copy()
    x_std[:, :, ext_nodes, 2] = (x_std[:, :, ext_nodes, 2] - mean) / std
    return x_std.astype(np.float32)


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
        f1s.append(0.0 if (p + r) == 0 else (2 * p * r / (p + r)))
    return float(np.mean(f1s))


def class_recall(cm, c):
    row = cm[c, :].sum()
    if row == 0:
        return 0.0
    return float(cm[c, c] / row)


def weighted_score(cm, penalty_32=0.12, bonus_r3=0.06, bonus_r2=0.05):
    base = macro_f1(cm)
    row3 = max(int(cm[3, :].sum()), 1)
    err_32 = float(cm[3, 2] / row3)
    r2 = class_recall(cm, 2)
    r3 = class_recall(cm, 3)
    return base - penalty_32 * err_32 + bonus_r3 * r3 + bonus_r2 * r2


def search_thresholds_constrained_weighted(val_probs, val_true, step=0.01, penalty_32=0.12, bonus_r3=0.06, bonus_r2=0.05):
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
                score = weighted_score(cm, penalty_32=penalty_32, bonus_r3=bonus_r3, bonus_r2=bonus_r2)
                if score > best_score:
                    best_score = float(score)
                    best_t = [float(t1), float(t2), float(t3)]
                    best_f = macro_f1(cm)
    return best_t, best_f, best_score


def run(args):
    out_dir = Path(args.out_dir)
    cache_path = Path(args.cache_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)

    x, y, ext_nodes = build_dataset(Path(args.data_path), cache_path)
    tr, va, te = split_indices(len(y), args.seed)

    mean = x[tr][:, :, ext_nodes, 2].mean(axis=0, keepdims=True)
    std = x[tr][:, :, ext_nodes, 2].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x = standardize_graph_voltage(x, mean, std, ext_nodes)
    np.savez_compressed(
        out_dir / "standardization.npz",
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        ext_nodes=ext_nodes.astype(np.int64),
    )

    train_ds = ClsDataset(
        x[tr],
        y[tr],
        ext_nodes,
        add_noise=args.add_noise,
        is_train=True,
        noise_schedule=args.noise_schedule,
        noise_std_max=args.noise_std_max,
        fixed_noise_std=args.fixed_noise_std,
        noise_mode=args.noise_mode,
        noise_scope=args.noise_scope,
        curriculum_stds=args.curriculum_stds,
        curriculum_probs=args.curriculum_probs,
        clean_mix_prob=args.clean_mix_prob,
        structured_iid_ratio=args.structured_iid_ratio,
        structured_drift_ratio=args.structured_drift_ratio,
        structured_common_ratio=args.structured_common_ratio,
        structured_bad_ratio=args.structured_bad_ratio,
        structured_bad_prob=args.structured_bad_prob,
    )
    val_ds = ClsDataset(x[va], y[va], ext_nodes)
    test_ds = ClsDataset(x[te], y[te], ext_nodes)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicsInformedGNNClassifier(
        in_dim=4,
        hidden_dim=args.hidden_dim,
        proj_dim=args.proj_dim,
        out_dim=NUM_CLASSES - 1,
        heads=args.gat_heads,
        excitation_chunk_size=args.excitation_chunk_size,
        dropout=args.dropout,
    ).to(device)
    if args.pretrained_model_path:
        state_dict = torch.load(args.pretrained_model_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        print(f"[Warm Start] loaded pretrained classifier from {args.pretrained_model_path}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    w_cls = class_weights(y[tr]).to(device)

    best_state = None
    best_epoch = 0
    best_score = -1e9
    best_thrs = [0.5, 0.5, 0.5]
    bad_epochs = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, aux = model(xb, return_aux=True)
            loss_coral = coral_loss(logits, yb, sample_w=w_cls[yb])
            loss_supcon = supervised_contrastive_loss(aux["contrast_feat"], yb, temperature=args.contrast_temp)
            loss = loss_coral + args.lambda_supcon * loss_supcon
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
                    logits, aux = model(xb, return_aux=True)
                    loss_coral = coral_loss(logits, yb, sample_w=w_cls[yb])
                    loss_supcon = supervised_contrastive_loss(aux["contrast_feat"], yb, temperature=args.contrast_temp)
                    va_loss += (loss_coral + args.lambda_supcon * loss_supcon).item() * xb.size(0)
                    val_logits.append(logits.cpu().numpy())
                    val_true.append(yb.cpu().numpy())

            va_loss /= len(val_loader.dataset)
            val_logits = np.concatenate(val_logits, axis=0)
            val_true = np.concatenate(val_true, axis=0)
            val_probs = expit(val_logits)
            thrs, va_f1, va_score = search_thresholds_constrained_weighted(
                val_probs,
                val_true,
                step=args.threshold_step,
                penalty_32=args.penalty_32,
                bonus_r3=args.bonus_r3,
                bonus_r2=args.bonus_r2,
            )
            print(f"Epoch {ep:03d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f} | val_macro_f1={va_f1:.4f}")
            if va_score > best_score + 1e-6:
                best_score = va_score
                best_epoch = ep
                best_thrs = thrs
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

    model.eval()
    test_logits = []
    test_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device)).cpu().numpy()
            test_logits.append(logits)
            test_true.append(yb.numpy())

    test_logits = np.concatenate(test_logits, axis=0)
    test_true = np.concatenate(test_true, axis=0)
    test_probs = expit(test_logits)
    test_pred = (
        (test_probs[:, 0] > best_thrs[0]).astype(np.int64)
        + (test_probs[:, 1] > best_thrs[1]).astype(np.int64)
        + (test_probs[:, 2] > best_thrs[2]).astype(np.int64)
    )
    cm = confusion(test_pred, test_true)
    final_f1 = macro_f1(cm)

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)
    print(f"val_best_thresholds={best_thrs} | test_macro_f1={final_f1:.4f}")

    (out_dir / "confusion_matrix.txt").write_text(np.array2string(cm), encoding="utf-8")
    metrics = {
        "dataset_tag": args.dataset_tag,
        "data_path": str(Path(args.data_path)),
        "cache_path": str(cache_path),
        "out_dir": str(out_dir),
        "best_thresholds": best_thrs,
        "test_macro_f1": final_f1,
        "lambda_supcon": args.lambda_supcon,
        "contrast_temp": args.contrast_temp,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "proj_dim": args.proj_dim,
        "gat_heads": args.gat_heads,
        "excitation_chunk_size": args.excitation_chunk_size,
        "pretrained_model_path": args.pretrained_model_path,
        "add_noise": args.add_noise,
        "noise_schedule": args.noise_schedule,
        "noise_mode": args.noise_mode,
        "noise_std_max": args.noise_std_max,
        "fixed_noise_std": args.fixed_noise_std,
        "noise_scope": args.noise_scope,
        "curriculum_stds": parse_float_list(args.curriculum_stds, []),
        "curriculum_probs": parse_float_list(args.curriculum_probs, []),
        "clean_mix_prob": args.clean_mix_prob,
        "structured_iid_ratio": args.structured_iid_ratio,
        "structured_drift_ratio": args.structured_drift_ratio,
        "structured_common_ratio": args.structured_common_ratio,
        "structured_bad_ratio": args.structured_bad_ratio,
        "structured_bad_prob": args.structured_bad_prob,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def parse_args():
    ensure_cli_option_has_value(sys.argv, "--dataset-tag")
    parser = argparse.ArgumentParser(description="64Nodes noisy fine-tuning for physics-informed GNN classifier modelo3.")
    parser.add_argument("--data-path", default=DEFAULT_MAIN_DATA_PATH)
    parser.add_argument("--cache-path", default="./cache_dataset_cls_graphattn.npz")
    parser.add_argument("--out-dir", default="./outputs")
    parser.add_argument("--dataset-tag", default="", help="数据集标签；默认取 data-path 文件名。")
    parser.set_defaults(dataset_subdir=True)
    parser.add_argument("--dataset-subdir", dest="dataset_subdir", action="store_true")
    parser.add_argument("--no-dataset-subdir", dest="dataset_subdir", action="store_false")
    parser.add_argument("--pretrained-model-path", default=DEFAULT_PRETRAINED_MODEL_PATH)
    parser.set_defaults(add_noise=True)
    parser.add_argument("--add-noise", dest="add_noise", action="store_true")
    parser.add_argument("--no-add-noise", dest="add_noise", action="store_false")
    parser.add_argument("--noise-schedule", choices=["random", "fixed", "curriculum"], default="curriculum")
    parser.add_argument("--noise-mode", choices=["gaussian", "uniform", "structured"], default="structured")
    parser.add_argument("--noise-std-max", type=float, default=0.1)
    parser.add_argument("--fixed-noise-std", type=float, default=0.1)
    parser.add_argument("--noise-scope", choices=["boundary", "all"], default="boundary")
    parser.add_argument("--curriculum-stds", default="0.01,0.0316227766,0.1")
    parser.add_argument("--curriculum-probs", default="0.5,0.35,0.15")
    parser.add_argument("--clean-mix-prob", type=float, default=0.15)
    parser.add_argument("--structured-iid-ratio", type=float, default=0.55)
    parser.add_argument("--structured-drift-ratio", type=float, default=0.25)
    parser.add_argument("--structured-common-ratio", type=float, default=0.15)
    parser.add_argument("--structured-bad-ratio", type=float, default=0.20)
    parser.add_argument("--structured-bad-prob", type=float, default=0.06)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--excitation-chunk-size", type=int, default=4)
    parser.add_argument("--lambda-supcon", type=float, default=0.15)
    parser.add_argument("--contrast-temp", type=float, default=0.12)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--penalty-32", type=float, default=0.12)
    parser.add_argument("--bonus-r3", type=float, default=0.06)
    parser.add_argument("--bonus-r2", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260325)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    resolve_dataset_runtime_paths(args, script_dir, "./cache_dataset_cls_graphattn.npz")
    run(args)
