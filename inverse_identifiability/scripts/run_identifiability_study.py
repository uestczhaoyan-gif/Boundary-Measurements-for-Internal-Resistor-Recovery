import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
VENDOR_DIR = PROJECT_DIR / ".vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


GRID_SIZE = 8
NUM_NODES = GRID_SIZE * GRID_SIZE
NUM_RESISTORS = (GRID_SIZE * (GRID_SIZE - 1)) * 2
BASE_R = 1000.0

SINGLE_AMPLITUDES = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
PAIR_TRIPLE_SAME_AMPLITUDES = (5.0, 10.0, 20.0)
TOP_SIMILARITY_LIMIT = 2000

FOCUS4_EXCITATIONS = (
    (0, 63),
    (7, 56),
    (3, 60),
    (31, 32),
)


@dataclass(frozen=True)
class CaseDef:
    case_id: str
    family: str
    pattern: str
    amplitude_abs: float
    location_label: str
    changed_rids: tuple[int, ...]
    delta_values: tuple[float, ...]
    display_label: str


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


def project_excitations(ext_nodes):
    excitations = []
    n = len(ext_nodes)
    for i in range(n):
        excitations.append((ext_nodes[i], ext_nodes[(i + 1) % n]))
    excitations.extend(FOCUS4_EXCITATIONS)
    return excitations


def node_coord_bottom_left(nid):
    row_top = nid // GRID_SIZE
    col = nid % GRID_SIZE
    return float(col), float((GRID_SIZE - 1) - row_top)


def resistor_midpoint(rid, edges):
    n1, n2 = edges[rid]
    x1, y1 = node_coord_bottom_left(n1)
    x2, y2 = node_coord_bottom_left(n2)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def build_lookup(edges):
    return {tuple(sorted((n1, n2))): rid for rid, (n1, n2) in enumerate(edges)}


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


def boundary_nodes_set():
    return set(external_nodes_clockwise())


def classify_single_location(rid, edges):
    corner_nodes = {0, 7, 56, 63}
    boundary_nodes = boundary_nodes_set()
    n1, n2 = edges[rid]
    if n1 in corner_nodes or n2 in corner_nodes:
        return "corner"
    x, y = resistor_midpoint(rid, edges)
    if x in (0.0, 7.0) or y in (0.0, 7.0):
        return "edge"
    if n1 in boundary_nodes and n2 in boundary_nodes:
        return "edge"
    return "interior"


def classify_multi_location(rids, edges):
    labels = {classify_single_location(rid, edges) for rid in rids}
    if labels == {"corner"}:
        return "corner"
    if "corner" in labels:
        return "mixed_corner"
    if labels == {"edge"}:
        return "edge"
    if labels == {"interior"}:
        xs = []
        ys = []
        for rid in rids:
            x, y = resistor_midpoint(rid, edges)
            xs.append(x)
            ys.append(y)
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        if abs(cx - 3.5) <= 1.5 and abs(cy - 3.5) <= 1.5:
            return "center"
        return "interior"
    return "mixed"


def build_adjacent_pairs(edges):
    incidence = {nid: [] for nid in range(NUM_NODES)}
    for rid, (n1, n2) in enumerate(edges):
        incidence[n1].append(rid)
        incidence[n2].append(rid)
    pair_set = set()
    for rid_list in incidence.values():
        for rid1, rid2 in combinations(sorted(rid_list), 2):
            pair_set.add((rid1, rid2))
    return sorted(pair_set)


def build_connected_triples(adjacent_pairs):
    pair_set = set(adjacent_pairs)
    triples = []
    for rid1, rid2, rid3 in combinations(range(NUM_RESISTORS), 3):
        links = 0
        if (rid1, rid2) in pair_set:
            links += 1
        if (rid1, rid3) in pair_set:
            links += 1
        if (rid2, rid3) in pair_set:
            links += 1
        if links >= 2:
            triples.append((rid1, rid2, rid3))
    return triples


def select_triples(triples, edges, max_triple_bases):
    if max_triple_bases is None or max_triple_bases <= 0 or len(triples) <= max_triple_bases:
        return triples
    scored = []
    for triple in triples:
        xs = []
        ys = []
        for rid in triple:
            x, y = resistor_midpoint(rid, edges)
            xs.append(x)
            ys.append(y)
        cx = sum(xs) / 3.0
        cy = sum(ys) / 3.0
        boundary_touch = sum(1 for rid in triple if classify_single_location(rid, edges) != "interior")
        scored.append((boundary_touch, abs(cx - 3.5) + abs(cy - 3.5), triple))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    if max_triple_bases == 1:
        return [scored[len(scored) // 2][2]]
    chosen = []
    last_idx = -1
    for i in range(max_triple_bases):
        idx = round(i * (len(scored) - 1) / (max_triple_bases - 1))
        if idx == last_idx:
            continue
        chosen.append(scored[idx][2])
        last_idx = idx
    return chosen


def build_cases(edges, max_triple_bases):
    adjacent_pairs = build_adjacent_pairs(edges)
    all_connected_triples = build_connected_triples(adjacent_pairs)
    connected_triples = select_triples(all_connected_triples, edges, max_triple_bases)

    cases = []

    for rid in range(NUM_RESISTORS):
        location = classify_single_location(rid, edges)
        n1, n2 = edges[rid]
        for amp in SINGLE_AMPLITUDES:
            cases.append(
                CaseDef(
                    case_id=f"single_r{rid:03d}_p{int(amp):02d}",
                    family="single",
                    pattern="single_plus",
                    amplitude_abs=amp,
                    location_label=location,
                    changed_rids=(rid,),
                    delta_values=(amp,),
                    display_label=f"Single {n1}-{n2} (+{int(amp)} ohm)",
                )
            )

    for rid1, rid2 in adjacent_pairs:
        location = classify_multi_location((rid1, rid2), edges)
        n11, n12 = edges[rid1]
        n21, n22 = edges[rid2]
        for amp in PAIR_TRIPLE_SAME_AMPLITUDES:
            cases.append(
                CaseDef(
                    case_id=f"pair_same_r{rid1:03d}_r{rid2:03d}_p{int(amp):02d}",
                    family="pair",
                    pattern="pair_same",
                    amplitude_abs=amp,
                    location_label=location,
                    changed_rids=(rid1, rid2),
                    delta_values=(amp, amp),
                    display_label=f"Pair {n11}-{n12} & {n21}-{n22} (+{int(amp)} ohm each)",
                )
            )
        for deltas in ((10.0, -10.0), (-10.0, 10.0)):
            signed = "".join("p" if delta > 0 else "m" for delta in deltas)
            cases.append(
                CaseDef(
                    case_id=f"pair_cancel_r{rid1:03d}_r{rid2:03d}_{signed}",
                    family="pair",
                    pattern="pair_cancel",
                    amplitude_abs=10.0,
                    location_label=location,
                    changed_rids=(rid1, rid2),
                    delta_values=deltas,
                    display_label=f"Pair {n11}-{n12} & {n21}-{n22} ({int(deltas[0]):+d}/{int(deltas[1]):+d} ohm)",
                )
            )

    for rid1, rid2, rid3 in connected_triples:
        location = classify_multi_location((rid1, rid2, rid3), edges)
        label_edges = [f"{edges[rid][0]}-{edges[rid][1]}" for rid in (rid1, rid2, rid3)]
        for amp in PAIR_TRIPLE_SAME_AMPLITUDES:
            cases.append(
                CaseDef(
                    case_id=f"triple_same_r{rid1:03d}_r{rid2:03d}_r{rid3:03d}_p{int(amp):02d}",
                    family="triple",
                    pattern="triple_same",
                    amplitude_abs=amp,
                    location_label=location,
                    changed_rids=(rid1, rid2, rid3),
                    delta_values=(amp, amp, amp),
                    display_label=f"Triple {' / '.join(label_edges)} (+{int(amp)} ohm each)",
                )
            )
        cancel_patterns = (
            (10.0, -5.0, -5.0),
            (-5.0, 10.0, -5.0),
            (-5.0, -5.0, 10.0),
        )
        for idx, deltas in enumerate(cancel_patterns):
            cases.append(
                CaseDef(
                    case_id=f"triple_cancel_r{rid1:03d}_r{rid2:03d}_r{rid3:03d}_o{idx}",
                    family="triple",
                    pattern="triple_cancel",
                    amplitude_abs=10.0,
                    location_label=location,
                    changed_rids=(rid1, rid2, rid3),
                    delta_values=deltas,
                    display_label=f"Triple {' / '.join(label_edges)} ({int(deltas[0]):+d}/{int(deltas[1]):+d}/{int(deltas[2]):+d} ohm)",
                )
            )

    meta = {
        "adjacent_pairs_total": len(adjacent_pairs),
        "connected_triples_total": len(all_connected_triples),
        "connected_triples_used": len(connected_triples),
    }
    return cases, meta


def choose_excitations(mode, ext_nodes):
    if mode == "focus4":
        return list(FOCUS4_EXCITATIONS)
    if mode == "full32":
        return project_excitations(ext_nodes)
    raise ValueError(f"Unsupported excitation mode: {mode}")


def simulate_case(case_def, excitations, ext_nodes, edges, keep_idx, current_a):
    values = np.full(NUM_RESISTORS, BASE_R, dtype=np.float64)
    for rid, delta in zip(case_def.changed_rids, case_def.delta_values):
        values[rid] = BASE_R + delta
    gmat = build_conductance(values, edges)
    responses = []
    for src, gnd in excitations:
        v = solve_for_excitation(gmat, src, gnd, current_a, keep_idx)
        responses.append(v[ext_nodes])
    return np.stack(responses, axis=0)


def support_key(case_def):
    return tuple(case_def.changed_rids)


def compute_similarity_pairs(vectors, case_defs, threshold, limit):
    unit_vectors = vectors.copy().astype(np.float32)
    norms = np.linalg.norm(unit_vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit_vectors /= norms

    pairs = []
    n = unit_vectors.shape[0]
    block = 512
    supports = [support_key(case_def) for case_def in case_defs]
    for start in range(0, n, block):
        end = min(start + block, n)
        sims = unit_vectors[start:end] @ unit_vectors.T
        for local_row, row_idx in enumerate(range(start, end)):
            sims[local_row, : row_idx + 1] = -np.inf
            col_idx = np.where(sims[local_row] >= threshold)[0]
            for j in col_idx.tolist():
                if supports[row_idx] == supports[j]:
                    continue
                cosine = float(np.clip(sims[local_row, j], -1.0, 1.0))
                pairs.append(
                    {
                        "idx_i": row_idx,
                        "idx_j": int(j),
                        "cosine_similarity": cosine,
                    }
                )
    pairs.sort(key=lambda item: item["cosine_similarity"], reverse=True)
    if limit and len(pairs) > limit:
        pairs = pairs[:limit]
    max_sim = np.full(n, -np.inf, dtype=np.float32)
    for item in pairs:
        i = item["idx_i"]
        j = item["idx_j"]
        sim = item["cosine_similarity"]
        if sim > max_sim[i]:
            max_sim[i] = sim
        if sim > max_sim[j]:
            max_sim[j] = sim
    max_sim[max_sim == -np.inf] = np.nan
    return pairs, max_sim, unit_vectors


def format_rid_list(case_def, edges):
    return "|".join(f"r{rid}:{edges[rid][0]}-{edges[rid][1]}" for rid in case_def.changed_rids)


def write_case_metrics(path, case_defs, metrics, edges):
    fieldnames = [
        "case_id",
        "family",
        "pattern",
        "location_label",
        "amplitude_abs_ohm",
        "changed_resistors",
        "delta_values_ohm",
        "norm",
        "max_abs",
        "primary_norm",
        "primary_max_abs",
        "display_label",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case_def, item in zip(case_defs, metrics):
            writer.writerow(
                {
                    "case_id": case_def.case_id,
                    "family": case_def.family,
                    "pattern": case_def.pattern,
                    "location_label": case_def.location_label,
                    "amplitude_abs_ohm": f"{case_def.amplitude_abs:.6f}",
                    "changed_resistors": format_rid_list(case_def, edges),
                    "delta_values_ohm": "|".join(f"{delta:+.6f}" for delta in case_def.delta_values),
                    "norm": f"{item['norm']:.10e}",
                    "max_abs": f"{item['max_abs']:.10e}",
                    "primary_norm": f"{item['primary_norm']:.10e}",
                    "primary_max_abs": f"{item['primary_max_abs']:.10e}",
                    "display_label": case_def.display_label,
                }
            )


def detection_summary(case_defs, metrics, edges, vector_dim, noise_threshold):
    noise_norm_threshold = noise_threshold * math.sqrt(vector_dim)
    single_cases = []
    for case_def, item in zip(case_defs, metrics):
        if case_def.family == "single":
            single_cases.append((case_def, item))

    per_rid_threshold = {}
    for rid in range(NUM_RESISTORS):
        rid_items = [(case_def, item) for case_def, item in single_cases if case_def.changed_rids == (rid,)]
        rid_items.sort(key=lambda pair: pair[0].amplitude_abs)
        threshold = None
        for case_def, item in rid_items:
            if item["norm"] >= noise_norm_threshold:
                threshold = case_def.amplitude_abs
                break
        per_rid_threshold[rid] = threshold

    category_stats = {}
    all_detectable_rates = {}
    for amplitude in SINGLE_AMPLITUDES:
        amp_cases = [(case_def, item) for case_def, item in single_cases if case_def.amplitude_abs == amplitude]
        detectable = sum(1 for _, item in amp_cases if item["norm"] >= noise_norm_threshold)
        all_detectable_rates[str(int(amplitude))] = detectable / len(amp_cases)

    for category in ("corner", "edge", "interior"):
        rid_thresholds = []
        for rid in range(NUM_RESISTORS):
            if classify_single_location(rid, edges) == category and per_rid_threshold[rid] is not None:
                rid_thresholds.append(per_rid_threshold[rid])
        rates = {}
        for amplitude in SINGLE_AMPLITUDES:
            amp_cases = [
                item
                for case_def, item in single_cases
                if case_def.location_label == category and case_def.amplitude_abs == amplitude
            ]
            detectable = sum(1 for item in amp_cases if item["norm"] >= noise_norm_threshold)
            rates[str(int(amplitude))] = detectable / len(amp_cases) if amp_cases else 0.0
        category_stats[category] = {
            "count": sum(1 for rid in range(NUM_RESISTORS) if classify_single_location(rid, edges) == category),
            "threshold_min_ohm": None if not rid_thresholds else min(rid_thresholds),
            "threshold_median_ohm": None if not rid_thresholds else float(np.median(np.array(rid_thresholds))),
            "threshold_max_ohm": None if not rid_thresholds else max(rid_thresholds),
            "detectable_rate_by_amplitude": rates,
        }

    robust_threshold = None
    for amplitude in SINGLE_AMPLITUDES:
        if all_detectable_rates[str(int(amplitude))] >= 0.90:
            robust_threshold = amplitude
            break

    optimistic_threshold = None
    for amplitude in SINGLE_AMPLITUDES:
        if all_detectable_rates[str(int(amplitude))] > 0.0:
            optimistic_threshold = amplitude
            break

    return {
        "vector_dim": vector_dim,
        "noise_per_measurement_v": noise_threshold,
        "noise_norm_threshold_v": noise_norm_threshold,
        "optimistic_detectable_threshold_ohm": optimistic_threshold,
        "robust_detectable_threshold_ohm_at_90pct": robust_threshold,
        "overall_detectable_rate_by_amplitude": all_detectable_rates,
        "category_stats": category_stats,
    }


def save_similarity_pairs(path, pairs, case_defs, metrics):
    fieldnames = [
        "case_i",
        "case_j",
        "support_i",
        "support_j",
        "family_i",
        "family_j",
        "pattern_i",
        "pattern_j",
        "location_i",
        "location_j",
        "cosine_similarity",
        "norm_i",
        "norm_j",
        "label_i",
        "label_j",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in pairs:
            i = item["idx_i"]
            j = item["idx_j"]
            writer.writerow(
                {
                    "case_i": case_defs[i].case_id,
                    "case_j": case_defs[j].case_id,
                    "support_i": "|".join(str(rid) for rid in case_defs[i].changed_rids),
                    "support_j": "|".join(str(rid) for rid in case_defs[j].changed_rids),
                    "family_i": case_defs[i].family,
                    "family_j": case_defs[j].family,
                    "pattern_i": case_defs[i].pattern,
                    "pattern_j": case_defs[j].pattern,
                    "location_i": case_defs[i].location_label,
                    "location_j": case_defs[j].location_label,
                    "cosine_similarity": f"{item['cosine_similarity']:.8f}",
                    "norm_i": f"{metrics[i]['norm']:.10e}",
                    "norm_j": f"{metrics[j]['norm']:.10e}",
                    "label_i": case_defs[i].display_label,
                    "label_j": case_defs[j].display_label,
                }
            )


def find_special_triplets(case_defs, vectors, unit_vectors):
    results = []
    for family in ("single", "pair", "triple"):
        indices = [idx for idx, case_def in enumerate(case_defs) if case_def.family == family]
        if len(indices) < 3:
            continue
        family_units = unit_vectors[indices]
        sims = family_units @ family_units.T
        np.fill_diagonal(sims, -np.inf)
        best = None
        for local_idx, global_idx in enumerate(indices):
            neighbor_local = np.argsort(sims[local_idx])[::-1][:2]
            if len(neighbor_local) < 2:
                continue
            trio = sorted({global_idx, indices[int(neighbor_local[0])], indices[int(neighbor_local[1])]})
            if len(trio) < 3:
                continue
            if len({support_key(case_defs[idx]) for idx in trio}) < 3:
                continue
            pair_cosines = []
            pair_diffs = []
            pair_max_abs_diffs = []
            for i, j in combinations(trio, 2):
                cosine = float(np.clip(unit_vectors[i] @ unit_vectors[j], -1.0, 1.0))
                pair_cosines.append(cosine)
                diff = vectors[i] - vectors[j]
                pair_diffs.append(float(np.linalg.norm(diff)))
                pair_max_abs_diffs.append(float(np.max(np.abs(diff))))
            score = (min(pair_cosines), sum(pair_cosines) / len(pair_cosines))
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "family": family,
                    "trio": trio,
                    "cosine_min": min(pair_cosines),
                    "cosine_max": max(pair_cosines),
                    "diff_norm_min": min(pair_diffs),
                    "diff_norm_max": max(pair_diffs),
                    "diff_max_abs_min": min(pair_max_abs_diffs),
                    "diff_max_abs_max": max(pair_max_abs_diffs),
                }
        if best is not None:
            results.append(best)
    return results


def save_special_triplets(path, triplets, case_defs):
    fieldnames = [
        "family",
        "case_a",
        "case_b",
        "case_c",
        "label_a",
        "label_b",
        "label_c",
        "cosine_min",
        "cosine_max",
        "diff_norm_min",
        "diff_norm_max",
        "diff_max_abs_min",
        "diff_max_abs_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for triplet in triplets:
            a, b, c = triplet["trio"]
            writer.writerow(
                {
                    "family": triplet["family"],
                    "case_a": case_defs[a].case_id,
                    "case_b": case_defs[b].case_id,
                    "case_c": case_defs[c].case_id,
                    "label_a": case_defs[a].display_label,
                    "label_b": case_defs[b].display_label,
                    "label_c": case_defs[c].display_label,
                    "cosine_min": f"{triplet['cosine_min']:.8f}",
                    "cosine_max": f"{triplet['cosine_max']:.8f}",
                    "diff_norm_min": f"{triplet['diff_norm_min']:.10e}",
                    "diff_norm_max": f"{triplet['diff_norm_max']:.10e}",
                    "diff_max_abs_min": f"{triplet['diff_max_abs_min']:.10e}",
                    "diff_max_abs_max": f"{triplet['diff_max_abs_max']:.10e}",
                }
            )


def example_case_ids(edges):
    lookup = build_lookup(edges)
    corner = lookup[tuple(sorted((0, 1)))]
    edge = lookup[tuple(sorted((7, 15)))]
    center = lookup[tuple(sorted((27, 28)))]
    center_v = lookup[tuple(sorted((28, 36)))]
    center_h = lookup[tuple(sorted((28, 29)))]
    triple_rids = tuple(sorted((center, center_v, center_h)))
    return {
        "single_corner": f"single_r{corner:03d}_p10",
        "single_edge": f"single_r{edge:03d}_p10",
        "single_center": f"single_r{center:03d}_p10",
        "pair_center": f"pair_same_r{min(center, center_v):03d}_r{max(center, center_v):03d}_p10",
        "triple_center": f"triple_same_r{triple_rids[0]:03d}_r{triple_rids[1]:03d}_r{triple_rids[2]:03d}_p10",
    }


def fallback_example_index(case_defs, preferred_id, family, pattern, amplitude_abs, location_prefs):
    case_index = {case_def.case_id: idx for idx, case_def in enumerate(case_defs)}
    if preferred_id in case_index:
        return case_index[preferred_id]
    for location in location_prefs:
        for idx, case_def in enumerate(case_defs):
            if (
                case_def.family == family
                and case_def.pattern == pattern
                and case_def.amplitude_abs == amplitude_abs
                and case_def.location_label == location
            ):
                return idx
    for idx, case_def in enumerate(case_defs):
        if case_def.family == family and case_def.pattern == pattern and case_def.amplitude_abs == amplitude_abs:
            return idx
    raise KeyError(preferred_id)


def plot_examples(fig_path, case_defs, delta_matrices, ext_nodes, excitations, figure_excitation, edges):
    excitation_to_index = {exc: idx for idx, exc in enumerate(excitations)}
    primary_idx = excitation_to_index[figure_excitation]
    selected_ids = example_case_ids(edges)
    selected_indices = {
        "single_corner": fallback_example_index(case_defs, selected_ids["single_corner"], "single", "single_plus", 10.0, ("corner",)),
        "single_edge": fallback_example_index(case_defs, selected_ids["single_edge"], "single", "single_plus", 10.0, ("edge",)),
        "single_center": fallback_example_index(case_defs, selected_ids["single_center"], "single", "single_plus", 10.0, ("interior",)),
        "pair_center": fallback_example_index(case_defs, selected_ids["pair_center"], "pair", "pair_same", 10.0, ("center", "interior")),
        "triple_center": fallback_example_index(case_defs, selected_ids["triple_center"], "triple", "triple_same", 10.0, ("center", "interior")),
    }

    plt.figure(figsize=(11, 5.5))
    x = np.arange(len(ext_nodes))
    plt.axhline(0.0, color="#808080", linestyle="--", linewidth=1.0, label="Base (delta = 0)")

    color_map = {
        "single_corner": "#c0392b",
        "single_edge": "#e67e22",
        "single_center": "#2980b9",
        "pair_center": "#16a085",
        "triple_center": "#8e44ad",
    }
    for key in ("single_corner", "single_edge", "single_center", "pair_center", "triple_center"):
        idx = selected_indices[key]
        curve = delta_matrices[idx][primary_idx]
        label = case_defs[idx].display_label
        plt.plot(x, curve, marker="o", markersize=3.5, linewidth=1.8, color=color_map[key], label=label)

    plt.xticks(x)
    plt.xlabel("Boundary node index (clockwise, 28 nodes)")
    plt.ylabel("Delta V (V)")
    plt.title("Boundary-voltage delta under 0->63 excitation")
    plt.grid(alpha=0.25, linestyle=":")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()


def plot_norm_vs_amplitude(fig_path, case_defs, metrics, noise_norm_threshold):
    plt.figure(figsize=(10.5, 5.8))
    categories = ("corner", "edge", "interior")
    colors = {
        "corner": "#c0392b",
        "edge": "#e67e22",
        "interior": "#2980b9",
    }

    for category in categories:
        xs = []
        ys = []
        med_x = []
        med_y = []
        for amplitude in SINGLE_AMPLITUDES:
            amp_values = [
                item["norm"]
                for case_def, item in zip(case_defs, metrics)
                if case_def.family == "single"
                and case_def.location_label == category
                and case_def.amplitude_abs == amplitude
            ]
            xs.extend([amplitude] * len(amp_values))
            ys.extend(amp_values)
            if amp_values:
                med_x.append(amplitude)
                med_y.append(float(np.median(np.array(amp_values))))
        jitter = np.linspace(-0.15, 0.15, num=len(xs)) if xs else []
        jittered_x = [x + j for x, j in zip(xs, jitter)] if xs else []
        plt.scatter(jittered_x, ys, s=16, alpha=0.38, color=colors[category], label=f"{category} samples")
        plt.plot(med_x, med_y, color=colors[category], linewidth=2.4, marker="o", label=f"{category} median")

    plt.axhline(noise_norm_threshold, color="#2c3e50", linestyle="--", linewidth=1.5, label="Noise norm threshold")
    plt.yscale("log")
    plt.xlabel("Absolute resistor change (ohm)")
    plt.ylabel("Aggregated Euclidean norm of Delta V")
    plt.title("Boundary-voltage sensitivity vs change amplitude")
    plt.grid(alpha=0.25, linestyle=":")
    handles, labels = plt.gca().get_legend_handles_labels()
    dedup = {}
    for handle, label in zip(handles, labels):
        if label not in dedup:
            dedup[label] = handle
    plt.legend(dedup.values(), dedup.keys(), fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()


def select_heatmap_cases(pairs, max_sim, limit):
    selected = []
    for item in pairs:
        for idx in (item["idx_i"], item["idx_j"]):
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= limit:
                return selected
    ranked = np.argsort(np.nan_to_num(max_sim, nan=-1.0))[::-1]
    for idx in ranked.tolist():
        if idx not in selected:
            selected.append(int(idx))
        if len(selected) >= limit:
            break
    return selected


def plot_similarity_heatmap(fig_path, selected_indices, case_defs, unit_vectors):
    if not selected_indices:
        return
    ordered = sorted(
        selected_indices,
        key=lambda idx: (case_defs[idx].family, case_defs[idx].pattern, case_defs[idx].amplitude_abs, case_defs[idx].case_id),
    )
    selected = unit_vectors[ordered]
    sim_matrix = selected @ selected.T

    plt.figure(figsize=(9.5, 8.2))
    im = plt.imshow(sim_matrix, cmap="viridis", vmin=0.85, vmax=1.0, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Cosine similarity")
    ticks = np.arange(len(ordered))
    tick_labels = [case_defs[idx].family[0].upper() for idx in ordered]
    plt.xticks(ticks, tick_labels, fontsize=5, rotation=90)
    plt.yticks(ticks, tick_labels, fontsize=5)
    plt.title("Cosine-similarity heatmap of high-ambiguity cases")
    plt.xlabel("Selected cases (S/P/T)")
    plt.ylabel("Selected cases (S/P/T)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()


def write_analysis_summary(path, args, excitations, meta_counts, detection, pairs, triplets):
    cross_family_counts = {}
    for item in pairs:
        family_pair = tuple(sorted((item["family_i"], item["family_j"])))
        key = " vs ".join(family_pair)
        cross_family_counts[key] = cross_family_counts.get(key, 0) + 1

    lines = []
    lines.append("# Inverse Identifiability Summary")
    lines.append("")
    lines.append("## Setup")
    lines.append("- Grid: 8x8, 64 nodes, 112 resistors.")
    lines.append("- Boundary measurements follow the project clockwise boundary-node order.")
    lines.append("- Unique boundary nodes: 28.")
    lines.append(f"- Excitation mode: {args.excitation_mode}.")
    lines.append(f"- Excitations used here: {', '.join(f'{src}->{gnd}' for src, gnd in excitations)}.")
    lines.append(f"- Noise threshold per measurement: {args.noise_threshold:.3e} V.")
    lines.append(f"- Aggregated vector dimension: {detection['vector_dim']}.")
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"- Adjacent resistor pairs: {meta_counts['adjacent_pairs_total']}.")
    lines.append(f"- Connected resistor triples available: {meta_counts['connected_triples_total']}.")
    lines.append(f"- Connected resistor triples used: {meta_counts['connected_triples_used']}.")
    lines.append("")
    lines.append("## Detectability")
    lines.append(f"- Optimistic single-edge threshold: {detection['optimistic_detectable_threshold_ohm']} ohm.")
    lines.append(
        f"- Robust single-edge threshold at 90% detectability: "
        f"{detection['robust_detectable_threshold_ohm_at_90pct']} ohm."
    )
    lines.append(f"- Noise-equivalent norm threshold: {detection['noise_norm_threshold_v']:.6e}.")
    for category, stats in detection["category_stats"].items():
        lines.append(
            f"- {category}: min/median/max threshold = "
            f"{stats['threshold_min_ohm']} / {stats['threshold_median_ohm']} / {stats['threshold_max_ohm']} ohm."
        )
    lines.append("")
    lines.append("## Non-Uniqueness")
    lines.append(
        f"- Different-support high-similarity pairs saved: {len(pairs)} "
        f"(cosine >= {args.cosine_threshold:.3f})."
    )
    if cross_family_counts:
        for key, value in sorted(cross_family_counts.items()):
            lines.append(f"- {key}: {value} pairs.")
    for triplet in triplets:
        lines.append(
            f"- {triplet['family']} best triplet cosine range: "
            f"{triplet['cosine_min']:.6f} to {triplet['cosine_max']:.6f}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def attach_family_info_to_pairs(pairs, case_defs):
    enriched = []
    for item in pairs:
        i = item["idx_i"]
        j = item["idx_j"]
        enriched.append(
            {
                **item,
                "family_i": case_defs[i].family,
                "family_j": case_defs[j].family,
            }
        )
    return enriched


def main():
    parser = argparse.ArgumentParser(description="Run inverse identifiability study for the 64Nodes resistor grid.")
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "outputs"))
    parser.add_argument("--current-a", type=float, default=0.005)
    parser.add_argument("--noise-threshold", type=float, default=1e-3)
    parser.add_argument("--excitation-mode", choices=("focus4", "full32"), default="focus4")
    parser.add_argument("--max-triple-bases", type=int, default=0, help="0 means use all connected triples.")
    parser.add_argument("--cosine-threshold", type=float, default=0.99)
    parser.add_argument("--heatmap-case-limit", type=int, default=96)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    edges = build_edges()
    ext_nodes = external_nodes_clockwise()
    excitations = choose_excitations(args.excitation_mode, ext_nodes)
    keep_idx = {gnd: np.array([i for i in range(NUM_NODES) if i != gnd], dtype=np.int64) for gnd in range(NUM_NODES)}

    baseline_case = CaseDef(
        case_id="baseline",
        family="baseline",
        pattern="baseline",
        amplitude_abs=0.0,
        location_label="baseline",
        changed_rids=tuple(),
        delta_values=tuple(),
        display_label="Baseline",
    )
    baseline_response = simulate_case(baseline_case, excitations, ext_nodes, edges, keep_idx, args.current_a)

    case_defs, meta_counts = build_cases(edges, args.max_triple_bases)
    delta_vectors = []
    delta_matrices = []
    metrics = []

    vector_dim = len(excitations) * len(ext_nodes)
    figure_excitation = (0, 63)
    primary_idx = {exc: idx for idx, exc in enumerate(excitations)}[figure_excitation]

    for idx, case_def in enumerate(case_defs):
        response = simulate_case(case_def, excitations, ext_nodes, edges, keep_idx, args.current_a)
        delta = response - baseline_response
        delta_vector = delta.reshape(-1).astype(np.float32)
        delta_vectors.append(delta_vector)
        delta_matrices.append(delta.astype(np.float32))
        metrics.append(
            {
                "norm": float(np.linalg.norm(delta_vector)),
                "max_abs": float(np.max(np.abs(delta_vector))),
                "primary_norm": float(np.linalg.norm(delta[primary_idx])),
                "primary_max_abs": float(np.max(np.abs(delta[primary_idx]))),
            }
        )
        if (idx + 1) % 500 == 0:
            print(f"Simulated {idx + 1}/{len(case_defs)} cases")

    delta_vectors = np.stack(delta_vectors, axis=0)

    pairs, max_sim, unit_vectors = compute_similarity_pairs(
        vectors=delta_vectors,
        case_defs=case_defs,
        threshold=args.cosine_threshold,
        limit=TOP_SIMILARITY_LIMIT,
    )
    enriched_pairs = attach_family_info_to_pairs(pairs, case_defs)
    triplets = find_special_triplets(case_defs, delta_vectors, unit_vectors)
    detection = detection_summary(case_defs, metrics, edges, vector_dim, args.noise_threshold)
    selected_heatmap_cases = select_heatmap_cases(pairs, max_sim, args.heatmap_case_limit)

    case_metrics_path = output_dir / "case_metrics.csv"
    similarity_pairs_path = output_dir / "high_similarity_pairs.csv"
    triplets_path = output_dir / "special_case_triplets.csv"
    detection_path = output_dir / "detection_summary.json"
    summary_path = output_dir / "analysis_summary.md"

    write_case_metrics(case_metrics_path, case_defs, metrics, edges)
    save_similarity_pairs(similarity_pairs_path, pairs, case_defs, metrics)
    save_special_triplets(triplets_path, triplets, case_defs)
    detection_path.write_text(json.dumps(detection, indent=2), encoding="utf-8")

    plot_examples(
        output_dir / "fig1_delta_v_examples.png",
        case_defs,
        delta_matrices,
        ext_nodes,
        excitations,
        figure_excitation,
        edges,
    )
    plot_norm_vs_amplitude(output_dir / "fig2_norm_vs_amplitude.png", case_defs, metrics, detection["noise_norm_threshold_v"])
    plot_similarity_heatmap(output_dir / "fig3_cosine_similarity_heatmap.png", selected_heatmap_cases, case_defs, unit_vectors)

    write_analysis_summary(summary_path, args, excitations, meta_counts, detection, enriched_pairs, triplets)

    manifest = {
        "boundary_nodes_clockwise": ext_nodes,
        "excitations_used": excitations,
        "case_count": len(case_defs),
        "vector_dim": vector_dim,
        "files": {
            "case_metrics": str(case_metrics_path),
            "high_similarity_pairs": str(similarity_pairs_path),
            "special_case_triplets": str(triplets_path),
            "detection_summary": str(detection_path),
            "analysis_summary": str(summary_path),
            "fig1": str(output_dir / "fig1_delta_v_examples.png"),
            "fig2": str(output_dir / "fig2_norm_vs_amplitude.png"),
            "fig3": str(output_dir / "fig3_cosine_similarity_heatmap.png"),
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Completed identifiability study for {len(case_defs)} cases.")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
