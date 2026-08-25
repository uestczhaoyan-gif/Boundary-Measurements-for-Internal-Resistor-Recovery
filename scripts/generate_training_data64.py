import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


GRID_SIZE = 8
NUM_NODES = GRID_SIZE * GRID_SIZE
NUM_RESISTORS = (GRID_SIZE * (GRID_SIZE - 1)) * 2
BASE_R = 1000.0


def build_edges():
    """
    Resistor numbering rule:
    For each row r:
    1) horizontal resistors in row r, left-to-right
    2) vertical resistors between row r and row r+1, left-to-right (except last row)
    """
    edges = [None] * NUM_RESISTORS
    block = (GRID_SIZE - 1) + GRID_SIZE  # 7 + 8 = 15
    for r in range(GRID_SIZE):
        # Horizontal resistors (7 per row)
        for c in range(GRID_SIZE - 1):
            rid = block * r + c
            n1 = r * GRID_SIZE + c
            n2 = r * GRID_SIZE + (c + 1)
            edges[rid] = (n1, n2)
        # Vertical resistors (8 per row gap), except last row
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
    excitations = []
    n = len(ext_nodes)
    for i in range(n):
        src = ext_nodes[i]
        gnd = ext_nodes[(i + 1) % n]
        excitations.append((src, gnd))
    excitations.extend(
        [
            (0, 63),
            (7, 56),
            (3, 60),
            (31, 32),
        ]
    )
    return excitations


def combo_counts(total_combos):
    if total_combos == 10000:
        return {0: 700, 1: 3100, 2: 3100, 3: 3100}
    c0 = int(round(total_combos * 0.07))
    c1 = int(round(total_combos * 0.31))
    c2 = int(round(total_combos * 0.31))
    c3 = total_combos - c0 - c1 - c2
    return {0: c0, 1: c1, 2: c2, 3: c3}


def sample_changes(k, rng, seen, min_ratio, max_ratio):
    if k == 0:
        return []
    while True:
        rids = sorted(rng.sample(range(NUM_RESISTORS), k))
        changes = []
        for rid in rids:
            ratio = rng.uniform(min_ratio, max_ratio)
            sign = -1.0 if rng.random() < 0.5 else 1.0
            new_r = BASE_R * (1.0 + sign * ratio)
            changes.append((rid, round(new_r, 6)))
        key = tuple(changes)
        if key not in seen:
            seen.add(key)
            return changes


def build_conductance(values, edges):
    gmat = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float64)
    for rid, (n1, n2) in enumerate(edges):
        g = 1.0 / values[rid]
        gmat[n1, n1] += g
        gmat[n2, n2] += g
        gmat[n1, n2] -= g
        gmat[n2, n1] -= g
    return gmat


def solve_for_excitation(gmat, src, gnd, current_a, keep_idx):
    i_vec = np.zeros(NUM_NODES, dtype=np.float64)
    i_vec[src] += current_a
    i_vec[gnd] -= current_a

    idx = keep_idx[gnd]
    g_reduced = gmat[np.ix_(idx, idx)]
    i_reduced = i_vec[idx]
    v_reduced = np.linalg.solve(g_reduced, i_reduced)

    v = np.zeros(NUM_NODES, dtype=np.float64)
    v[idx] = v_reduced
    v[gnd] = 0.0
    return v


def fmt(x, decimals):
    if abs(x) < 1e-12:
        x = 0.0
    return f"{x:.{decimals}f}"


def main():
    parser = argparse.ArgumentParser(description="Generate 8x8 resistor-grid training data by direct Kirchhoff solves.")
    parser.add_argument("--output", default="data/training_data64.csv", help="CSV output path")
    parser.add_argument("--meta-output", default="data/training_data64_meta.json", help="Metadata JSON path")
    parser.add_argument("--seed", type=int, default=20260319, help="Random seed")
    parser.add_argument("--total-combos", type=int, default=10000, help="Number of resistor combinations")
    parser.add_argument("--current-a", type=float, default=0.005, help="Current source amplitude (A)")
    parser.add_argument("--min-ratio", type=float, default=0.05, help="Minimum absolute change ratio")
    parser.add_argument("--max-ratio", type=float, default=0.30, help="Maximum absolute change ratio")
    parser.add_argument("--float-decimals", type=int, default=6, help="Decimals for output voltages/values")
    args = parser.parse_args()
    if np is None:
        raise SystemExit(
            "NumPy is required to generate data. Install dependencies first, "
            "for example: python -m pip install -r requirements.txt"
        )

    rng = random.Random(args.seed)
    edges = build_edges()
    ext_nodes = external_nodes_clockwise()
    excitations = build_excitations(ext_nodes)
    counts = combo_counts(args.total_combos)

    keep_idx = {}
    for gnd in range(NUM_NODES):
        keep_idx[gnd] = np.array([i for i in range(NUM_NODES) if i != gnd], dtype=np.int64)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.meta_output)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        ["row_id", "combo_id", "src_node", "gnd_node"]
        + [f"v_node{n}" for n in ext_nodes]
        + ["change_count", "r1_id", "r1_value", "r2_id", "r2_value", "r3_id", "r3_value"]
    )

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "grid_size": GRID_SIZE,
        "num_nodes": NUM_NODES,
        "num_resistors": NUM_RESISTORS,
        "base_resistance_ohm": BASE_R,
        "change_ratio_range": [args.min_ratio, args.max_ratio],
        "change_count_ratio": {"0": 0.07, "1": 0.31, "2": 0.31, "3": 0.31},
        "combo_counts": counts,
        "current_source_a": args.current_a,
        "external_nodes_clockwise": ext_nodes,
        "excitations": excitations,
        "resistor_edges": edges,
        "seed": args.seed,
        "float_decimals": args.float_decimals,
        "note": "Direct solve of Kirchhoff linear systems for every combo and every excitation. No low-rank acceleration.",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    seen_changed = set()
    row_id = 0
    combo_id = 0

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for k in [0, 1, 2, 3]:
            produced = 0
            while produced < counts[k]:
                changes = sample_changes(k, rng, seen_changed, args.min_ratio, args.max_ratio)
                values = np.full(NUM_RESISTORS, BASE_R, dtype=np.float64)
                for rid, new_r in changes:
                    values[rid] = new_r

                gmat = build_conductance(values, edges)
                padded = changes + [(-1, 0.0)] * (3 - k)

                for src, gnd in excitations:
                    v = solve_for_excitation(gmat, src, gnd, args.current_a, keep_idx)
                    row = [row_id, combo_id, src, gnd]
                    row.extend(fmt(v[n], args.float_decimals) for n in ext_nodes)
                    row.append(k)
                    for rid, val in padded[:3]:
                        row.append(rid)
                        row.append(fmt(val, args.float_decimals))
                    writer.writerow(row)
                    row_id += 1

                combo_id += 1
                produced += 1

    print(f"Generated rows: {row_id}")
    print(f"Generated combos: {combo_id}")
    print(f"CSV saved to: {out_path}")
    print(f"Meta saved to: {meta_path}")


if __name__ == "__main__":
    main()
