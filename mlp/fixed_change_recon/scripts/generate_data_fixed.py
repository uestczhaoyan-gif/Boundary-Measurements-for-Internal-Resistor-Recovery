import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np


GRID_SIZE = 8
NUM_NODES = GRID_SIZE * GRID_SIZE
NUM_RESISTORS = (GRID_SIZE * (GRID_SIZE - 1)) * 2
BASE_R = 1000.0
DEFAULT_FIXED_K = 3
DATA_PREFIX = "training_data64_fixed"
SUPPORTED_FIXED_K = (2, 3)


def sanitize_dataset_tag(raw_tag):
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw_tag.strip())
    safe = safe.strip("._-")
    return safe or "dataset"


def current_to_dataset_tag(current_a):
    ma = current_a * 1000.0
    rounded = round(ma)
    if abs(ma - rounded) < 1e-9:
        return f"{int(rounded)}mA"
    text = f"{ma:.3f}".rstrip("0").rstrip(".")
    return f"{text.replace('.', 'p')}mA"


def dataset_stem(fixed_k, dataset_tag):
    base = f"{DATA_PREFIX}_{int(fixed_k)}"
    if dataset_tag == "5mA":
        return base
    return f"{base}_{dataset_tag}"


def resolve_output_paths(args):
    base_dir = Path("64Nodes/mlp/fixed_change_recon/data_fixed")
    raw_tag = args.dataset_tag.strip() if args.dataset_tag else ""
    dataset_tag = sanitize_dataset_tag(raw_tag) if raw_tag else current_to_dataset_tag(args.current_a)
    stem = dataset_stem(args.fixed_k, dataset_tag)

    default_output = base_dir / f"{DATA_PREFIX}_{DEFAULT_FIXED_K}.csv"
    default_meta = base_dir / f"{DATA_PREFIX}_{DEFAULT_FIXED_K}_meta.json"
    default_coords = base_dir / "resistor_coords_bl_origin.json"

    output = Path(args.output)
    meta_output = Path(args.meta_output)
    coords_output = Path(args.coords_output)

    if output == default_output:
        output = base_dir / f"{stem}.csv"
    if meta_output == default_meta:
        meta_output = base_dir / f"{stem}_meta.json"
    if coords_output == default_coords:
        coords_output = default_coords

    args.output = str(output)
    args.meta_output = str(meta_output)
    args.coords_output = str(coords_output)
    args.dataset_tag = dataset_tag


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


def node_coord_bottom_left(nid):
    row_top = nid // GRID_SIZE
    col = nid % GRID_SIZE
    x = float(col)
    y = float((GRID_SIZE - 1) - row_top)
    return x, y


def resistor_mid_coords(edges):
    coords = {}
    for rid, (n1, n2) in enumerate(edges):
        x1, y1 = node_coord_bottom_left(n1)
        x2, y2 = node_coord_bottom_left(n2)
        coords[rid] = [round((x1 + x2) / 2.0, 6), round((y1 + y2) / 2.0, 6)]
    return coords


def sample_unique_changes(rng, seen, total, min_ratio, max_ratio, fixed_k):
    combos = []
    while len(combos) < total:
        rids = sorted(rng.sample(range(NUM_RESISTORS), fixed_k))
        vals = []
        for rid in rids:
            ratio = rng.uniform(min_ratio, max_ratio)
            sign = -1.0 if rng.random() < 0.5 else 1.0
            vals.append((rid, round(BASE_R * (1.0 + sign * ratio), 6)))
        key = tuple(vals)
        if key in seen:
            continue
        seen.add(key)
        combos.append(vals)
    return combos


def build_conductance(values, edges):
    gmat = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float64)
    for rid, (n1, n2) in enumerate(edges):
        g = 1.0 / values[rid]
        gmat[n1, n1] += g
        gmat[n2, n2] += g
        gmat[n1, n2] -= g
        gmat[n2, n1] -= g
    return gmat


def solve_excitation(gmat, src, gnd, current_a, keep_idx):
    i_vec = np.zeros(NUM_NODES, dtype=np.float64)
    i_vec[src] += current_a
    i_vec[gnd] -= current_a

    idx = keep_idx[gnd]
    g_red = gmat[np.ix_(idx, idx)]
    i_red = i_vec[idx]
    v_red = np.linalg.solve(g_red, i_red)

    v = np.zeros(NUM_NODES, dtype=np.float64)
    v[idx] = v_red
    v[gnd] = 0.0
    return v


def fmt(x, decimals):
    if abs(x) < 1e-12:
        x = 0.0
    return f"{x:.{decimals}f}"


def main():
    parser = argparse.ArgumentParser(description="Generate fixed-k-change data for 8x8 resistor grid via direct Kirchhoff solve.")
    parser.add_argument("--output", default="64Nodes/mlp/fixed_change_recon/data_fixed/training_data64_fixed_3.csv")
    parser.add_argument("--meta-output", default="64Nodes/mlp/fixed_change_recon/data_fixed/training_data64_fixed_3_meta.json")
    parser.add_argument("--coords-output", default="64Nodes/mlp/fixed_change_recon/data_fixed/resistor_coords_bl_origin.json")
    parser.add_argument("--dataset-tag", default="", help="数据集标签；默认按激励电流推导。")
    parser.add_argument("--fixed-k", type=int, choices=SUPPORTED_FIXED_K, default=DEFAULT_FIXED_K, help="固定变化电阻数量；当前固定为 2 或 3。")
    parser.add_argument("--seed", type=int, default=20260321)
    parser.add_argument("--total-combos", type=int, default=5000)
    parser.add_argument("--current-a", type=float, default=0.005)
    parser.add_argument("--min-ratio", type=float, default=0.05)
    parser.add_argument("--max-ratio", type=float, default=0.30)
    parser.add_argument("--float-decimals", type=int, default=6)
    args = parser.parse_args()
    resolve_output_paths(args)

    rng = random.Random(args.seed)
    edges = build_edges()
    ext_nodes = external_nodes_clockwise()
    excitations = build_excitations(ext_nodes)
    keep_idx = {gnd: np.array([i for i in range(NUM_NODES) if i != gnd], dtype=np.int64) for gnd in range(NUM_NODES)}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.meta_output)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    coords_path = Path(args.coords_output)
    coords_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["row_id", "combo_id", "src_node", "gnd_node"]
    header += [f"v_node{n}" for n in ext_nodes]
    header += ["change_count"]
    for i in range(1, args.fixed_k + 1):
        header += [f"r{i}_id", f"r{i}_value"]

    seen = set()
    combos = sample_unique_changes(
        rng=rng,
        seen=seen,
        total=args.total_combos,
        min_ratio=args.min_ratio,
        max_ratio=args.max_ratio,
        fixed_k=args.fixed_k,
    )

    row_id = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for combo_id, changes in enumerate(combos):
            vals = np.full(NUM_RESISTORS, BASE_R, dtype=np.float64)
            for rid, rv in changes:
                vals[rid] = rv
            gmat = build_conductance(vals, edges)

            for src, gnd in excitations:
                v = solve_excitation(gmat, src, gnd, args.current_a, keep_idx)
                row = [row_id, combo_id, src, gnd]
                row.extend(fmt(v[n], args.float_decimals) for n in ext_nodes)
                row.append(args.fixed_k)
                for rid, rv in changes:
                    row.append(rid)
                    row.append(fmt(rv, args.float_decimals))
                writer.writerow(row)
                row_id += 1

    coords_map = resistor_mid_coords(edges)
    coords_path.write_text(json.dumps(coords_map, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "grid_size": GRID_SIZE,
        "num_nodes": NUM_NODES,
        "num_resistors": NUM_RESISTORS,
        "base_resistance_ohm": BASE_R,
        "change_count_fixed": int(args.fixed_k),
        "change_ratio_range": [args.min_ratio, args.max_ratio],
        "total_combos": args.total_combos,
        "rows_total": int(args.total_combos * len(excitations)),
        "dataset_tag": args.dataset_tag,
        "current_source_a": args.current_a,
        "external_nodes_clockwise": ext_nodes,
        "excitations": excitations,
        "resistor_edges": edges,
        "coords_file": str(coords_path),
        "seed": args.seed,
        "float_decimals": args.float_decimals,
        "note": f"Direct Kirchhoff solve per combo/per excitation. Fixed exactly {int(args.fixed_k)} changed resistors per combo.",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Generated rows: {row_id}")
    print(f"Generated combos: {args.total_combos}")
    print(f"Fixed k: {args.fixed_k}")
    print(f"Dataset tag: {args.dataset_tag}")
    print(f"CSV saved to: {out_path}")
    print(f"Meta saved to: {meta_path}")
    print(f"Coords saved to: {coords_path}")


if __name__ == "__main__":
    main()

