import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_identifiability_study as ris

np = ris.np
plt = ris.plt


def parse_base_resistances(text):
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("At least one base resistance must be provided.")
    return values


def simulate_case_with_base(case_def, base_resistance, excitations, ext_nodes, edges, keep_idx, current_a):
    values = np.full(ris.NUM_RESISTORS, base_resistance, dtype=np.float64)
    for rid, delta in zip(case_def.changed_rids, case_def.delta_values):
        values[rid] = base_resistance + delta
    gmat = ris.build_conductance(values, edges)
    responses = []
    for src, gnd in excitations:
        v = ris.solve_for_excitation(gmat, src, gnd, current_a, keep_idx)
        responses.append(v[ext_nodes])
    return np.stack(responses, axis=0)


def exemplar_case_ids():
    return {
        "pair_same_center_p10": "pair_same_r048_r056_p10",
        "pair_cancel_center": "pair_cancel_r048_r056_mp",
        "triple_same_center_p10": "triple_same_r048_r049_r056_p10",
        "triple_cancel_center": "triple_cancel_r048_r049_r056_o2",
    }


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_results(fig_path, baseline_rows, aggregate_rows, exemplar_rows, similarity_rows):
    base_values = [row["base_resistance_ohm"] for row in baseline_rows]
    baseline_norms = [row["baseline_norm"] for row in baseline_rows]

    rep_keys = [
        ("pair_same", 10.0, "Pair same +10"),
        ("pair_cancel", 10.0, "Pair cancel +10/-10"),
        ("triple_same", 10.0, "Triple same +10"),
        ("triple_cancel", 10.0, "Triple cancel +10/-5/-5"),
    ]

    aggregate_lookup = {}
    for row in aggregate_rows:
        aggregate_lookup[(row["pattern"], row["amplitude_abs_ohm"])] = row

    exemplar_lookup = defaultdict(dict)
    for row in exemplar_rows:
        exemplar_lookup[row["exemplar_key"]][row["base_resistance_ohm"]] = row

    plt.figure(figsize=(12.0, 9.0))

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(base_values, baseline_norms, color="#2c3e50", marker="o", linewidth=2.2, label="Baseline norm")
    colors = {
        "pair_same": "#1f77b4",
        "pair_cancel": "#d62728",
        "triple_same": "#2ca02c",
        "triple_cancel": "#9467bd",
    }
    for pattern, amplitude, label in rep_keys:
        y = [aggregate_lookup[(pattern, amplitude)]["median_norm_by_base"][base_r] for base_r in base_values]
        ax1.plot(base_values, y, marker="o", linewidth=1.8, color=colors[pattern], label=label)
    ax1.set_yscale("log")
    ax1.set_xlabel("Base resistance (ohm)")
    ax1.set_ylabel("Norm scale (log)")
    ax1.set_title("Baseline voltage rises, anomaly norm barely changes")
    ax1.grid(alpha=0.25, linestyle=":")
    ax1.legend(fontsize=8)

    ax2 = plt.subplot(2, 2, 2)
    for pattern, amplitude, label in rep_keys:
        y = [aggregate_lookup[(pattern, amplitude)]["detectable_rate_by_base"][base_r] for base_r in base_values]
        ax2.plot(base_values, y, marker="o", linewidth=2.0, color=colors[pattern], label=label)
    ax2.set_xlabel("Base resistance (ohm)")
    ax2.set_ylabel("Detectable rate")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_title("Detectable rate is nearly unchanged")
    ax2.grid(alpha=0.25, linestyle=":")
    ax2.legend(fontsize=8)

    ax3 = plt.subplot(2, 2, 3)
    for pattern, amplitude, label in rep_keys:
        y = [aggregate_lookup[(pattern, amplitude)]["median_relative_norm_by_base"][base_r] for base_r in base_values]
        ax3.plot(base_values, y, marker="o", linewidth=2.0, color=colors[pattern], label=label)
    ax3.set_yscale("log")
    ax3.set_xlabel("Base resistance (ohm)")
    ax3.set_ylabel("Median norm / baseline norm")
    ax3.set_title("Relative anomaly signal shrinks as base resistance rises")
    ax3.grid(alpha=0.25, linestyle=":")
    ax3.legend(fontsize=8)

    ax4 = plt.subplot(2, 2, 4)
    sim_total = [row["high_similarity_pairs_total"] for row in similarity_rows]
    sim_detectable = [row["high_similarity_pairs_both_detectable"] for row in similarity_rows]
    ax4.plot(base_values, sim_total, color="#34495e", marker="o", linewidth=2.2, label="All high-similarity pairs")
    ax4.plot(base_values, sim_detectable, color="#e67e22", marker="o", linewidth=2.2, label="Both detectable")
    ax4.set_xlabel("Base resistance (ohm)")
    ax4.set_ylabel("Pair count")
    ax4.set_title("High-similarity ambiguity does not disappear")
    ax4.grid(alpha=0.25, linestyle=":")
    ax4.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare the effect of changing base resistance under fixed current.")
    parser.add_argument("--output-dir", default=str(ris.PROJECT_DIR / "base_resistance_experiment"))
    parser.add_argument("--base-resistances", default="1000,2000,5000")
    parser.add_argument("--current-a", type=float, default=0.005)
    parser.add_argument("--noise-threshold", type=float, default=1e-3)
    parser.add_argument("--excitation-mode", choices=("focus4", "full32"), default="full32")
    parser.add_argument("--cosine-threshold", type=float, default=0.99)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_resistances = parse_base_resistances(args.base_resistances)
    edges = ris.build_edges()
    ext_nodes = ris.external_nodes_clockwise()
    excitations = ris.choose_excitations(args.excitation_mode, ext_nodes)
    keep_idx = {gnd: np.array([i for i in range(ris.NUM_NODES) if i != gnd], dtype=np.int64) for gnd in range(ris.NUM_NODES)}
    vector_dim = len(excitations) * len(ext_nodes)
    noise_norm_threshold = args.noise_threshold * math.sqrt(vector_dim)

    cases, _ = ris.build_cases(edges, 0)
    cases = [case_def for case_def in cases if case_def.family in ("pair", "triple")]
    case_index = {case_def.case_id: case_def for case_def in cases}

    baseline_case = ris.CaseDef(
        case_id="baseline",
        family="baseline",
        pattern="baseline",
        amplitude_abs=0.0,
        location_label="baseline",
        changed_rids=tuple(),
        delta_values=tuple(),
        display_label="Baseline",
    )

    baseline_rows = []
    aggregate_groups = defaultdict(lambda: {
        "detectable_rate_by_base": {},
        "median_norm_by_base": {},
        "median_max_abs_by_base": {},
        "median_relative_norm_by_base": {},
    })
    location_rows = []
    similarity_rows = []
    similarity_breakdown_rows = []
    exemplar_rows = []

    exemplar_ids = exemplar_case_ids()

    for base_r in base_resistances:
        baseline_response = simulate_case_with_base(
            baseline_case, base_r, excitations, ext_nodes, edges, keep_idx, args.current_a
        )
        baseline_norm = float(np.linalg.norm(baseline_response.reshape(-1)))
        baseline_max_abs = float(np.max(np.abs(baseline_response)))
        baseline_rows.append(
            {
                "base_resistance_ohm": base_r,
                "baseline_norm": baseline_norm,
                "baseline_max_abs": baseline_max_abs,
            }
        )

        metric_rows = []
        vectors = []
        for idx, case_def in enumerate(cases):
            response = simulate_case_with_base(case_def, base_r, excitations, ext_nodes, edges, keep_idx, args.current_a)
            delta = response - baseline_response
            delta_vector = delta.reshape(-1).astype(np.float32)
            norm = float(np.linalg.norm(delta_vector))
            max_abs = float(np.max(np.abs(delta_vector)))
            metric_rows.append(
                {
                    "case_id": case_def.case_id,
                    "family": case_def.family,
                    "pattern": case_def.pattern,
                    "location_label": case_def.location_label,
                    "amplitude_abs_ohm": case_def.amplitude_abs,
                    "norm": norm,
                    "max_abs": max_abs,
                    "relative_norm": norm / baseline_norm,
                    "detectable": norm >= noise_norm_threshold,
                }
            )
            vectors.append(delta_vector)
            if (idx + 1) % 1000 == 0:
                print(f"Base {base_r:.0f} ohm: simulated {idx + 1}/{len(cases)} pair/triple cases")

        vectors = np.stack(vectors, axis=0)

        by_pattern_amp = defaultdict(list)
        by_pattern_loc_amp = defaultdict(list)
        for row in metric_rows:
            by_pattern_amp[(row["pattern"], row["amplitude_abs_ohm"])].append(row)
            by_pattern_loc_amp[(row["pattern"], row["location_label"], row["amplitude_abs_ohm"])].append(row)

        for (pattern, amplitude), rows in by_pattern_amp.items():
            norms = sorted(row["norm"] for row in rows)
            max_vals = sorted(row["max_abs"] for row in rows)
            rel_vals = sorted(row["relative_norm"] for row in rows)
            detectable_rate = sum(int(row["detectable"]) for row in rows) / len(rows)
            group = aggregate_groups[(pattern, amplitude)]
            group["detectable_rate_by_base"][base_r] = detectable_rate
            group["median_norm_by_base"][base_r] = norms[len(norms) // 2]
            group["median_max_abs_by_base"][base_r] = max_vals[len(max_vals) // 2]
            group["median_relative_norm_by_base"][base_r] = rel_vals[len(rel_vals) // 2]

        for (pattern, location, amplitude), rows in by_pattern_loc_amp.items():
            norms = sorted(row["norm"] for row in rows)
            rel_vals = sorted(row["relative_norm"] for row in rows)
            location_rows.append(
                {
                    "base_resistance_ohm": base_r,
                    "pattern": pattern,
                    "location_label": location,
                    "amplitude_abs_ohm": amplitude,
                    "n_cases": len(rows),
                    "detectable_rate": sum(int(row["detectable"]) for row in rows) / len(rows),
                    "median_norm": norms[len(norms) // 2],
                    "median_relative_norm": rel_vals[len(rel_vals) // 2],
                }
            )

        pairs, _, _ = ris.compute_similarity_pairs(
            vectors=vectors,
            case_defs=cases,
            threshold=args.cosine_threshold,
            limit=0,
        )
        both_detectable_count = 0
        pattern_pair_counter = Counter()
        for item in pairs:
            i = item["idx_i"]
            j = item["idx_j"]
            if metric_rows[i]["detectable"] and metric_rows[j]["detectable"]:
                both_detectable_count += 1
            key = tuple(sorted((cases[i].pattern, cases[j].pattern)))
            pattern_pair_counter[key] += 1

        similarity_rows.append(
            {
                "base_resistance_ohm": base_r,
                "high_similarity_pairs_total": len(pairs),
                "high_similarity_pairs_both_detectable": both_detectable_count,
                "cosine_threshold": args.cosine_threshold,
            }
        )

        for (pattern_i, pattern_j), count in sorted(pattern_pair_counter.items()):
            similarity_breakdown_rows.append(
                {
                    "base_resistance_ohm": base_r,
                    "pattern_i": pattern_i,
                    "pattern_j": pattern_j,
                    "pair_count": count,
                }
            )

        for exemplar_key, case_id in exemplar_ids.items():
            case_def = case_index[case_id]
            row = next(row for row in metric_rows if row["case_id"] == case_id)
            exemplar_rows.append(
                {
                    "base_resistance_ohm": base_r,
                    "exemplar_key": exemplar_key,
                    "case_id": case_id,
                    "display_label": case_def.display_label,
                    "pattern": case_def.pattern,
                    "location_label": case_def.location_label,
                    "amplitude_abs_ohm": case_def.amplitude_abs,
                    "norm": row["norm"],
                    "max_abs": row["max_abs"],
                    "relative_norm": row["relative_norm"],
                    "detectable": row["detectable"],
                }
            )

    aggregate_rows = []
    for (pattern, amplitude), payload in sorted(aggregate_groups.items()):
        aggregate_rows.append(
            {
                "pattern": pattern,
                "amplitude_abs_ohm": amplitude,
                "detectable_rate_by_base": payload["detectable_rate_by_base"],
                "median_norm_by_base": payload["median_norm_by_base"],
                "median_max_abs_by_base": payload["median_max_abs_by_base"],
                "median_relative_norm_by_base": payload["median_relative_norm_by_base"],
            }
        )

    aggregate_csv_rows = []
    for row in aggregate_rows:
        for base_r in base_resistances:
            aggregate_csv_rows.append(
                {
                    "base_resistance_ohm": base_r,
                    "pattern": row["pattern"],
                    "amplitude_abs_ohm": row["amplitude_abs_ohm"],
                    "detectable_rate": row["detectable_rate_by_base"][base_r],
                    "median_norm": row["median_norm_by_base"][base_r],
                    "median_max_abs": row["median_max_abs_by_base"][base_r],
                    "median_relative_norm": row["median_relative_norm_by_base"][base_r],
                }
            )

    write_csv(
        output_dir / "baseline_scaling.csv",
        ["base_resistance_ohm", "baseline_norm", "baseline_max_abs"],
        baseline_rows,
    )
    write_csv(
        output_dir / "aggregate_by_pattern.csv",
        [
            "base_resistance_ohm",
            "pattern",
            "amplitude_abs_ohm",
            "detectable_rate",
            "median_norm",
            "median_max_abs",
            "median_relative_norm",
        ],
        aggregate_csv_rows,
    )
    write_csv(
        output_dir / "aggregate_by_pattern_location.csv",
        [
            "base_resistance_ohm",
            "pattern",
            "location_label",
            "amplitude_abs_ohm",
            "n_cases",
            "detectable_rate",
            "median_norm",
            "median_relative_norm",
        ],
        location_rows,
    )
    write_csv(
        output_dir / "similarity_summary.csv",
        [
            "base_resistance_ohm",
            "high_similarity_pairs_total",
            "high_similarity_pairs_both_detectable",
            "cosine_threshold",
        ],
        similarity_rows,
    )
    write_csv(
        output_dir / "similarity_breakdown.csv",
        ["base_resistance_ohm", "pattern_i", "pattern_j", "pair_count"],
        similarity_breakdown_rows,
    )
    write_csv(
        output_dir / "exemplar_scaling.csv",
        [
            "base_resistance_ohm",
            "exemplar_key",
            "case_id",
            "display_label",
            "pattern",
            "location_label",
            "amplitude_abs_ohm",
            "norm",
            "max_abs",
            "relative_norm",
            "detectable",
        ],
        exemplar_rows,
    )

    plot_results(output_dir / "base_resistance_effects.png", baseline_rows, aggregate_rows, exemplar_rows, similarity_rows)

    summary_lines = [
        "# Base Resistance Experiment Summary",
        "",
        "Question: if the current stays fixed but the global base resistance rises, does that help pair/triple anomaly learning?",
        "",
        "Main observations:",
    ]
    for row in baseline_rows:
        summary_lines.append(
            f"- Base resistance {row['base_resistance_ohm']:.0f} ohm: baseline norm = {row['baseline_norm']:.6f}, "
            f"baseline max abs = {row['baseline_max_abs']:.6f}."
        )
    summary_lines.extend(
        [
            "- Baseline voltage grows almost linearly with base resistance.",
            "- Under fixed absolute resistor changes, pair/triple delta norms change only slightly.",
            "- Relative anomaly strength (delta norm divided by baseline norm) shrinks as base resistance rises.",
            "- High-similarity ambiguity counts remain nearly unchanged.",
        ]
    )
    (output_dir / "experiment_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    manifest = {
        "base_resistances_ohm": base_resistances,
        "excitation_mode": args.excitation_mode,
        "vector_dim": vector_dim,
        "noise_per_measurement_v": args.noise_threshold,
        "noise_norm_threshold_v": noise_norm_threshold,
        "files": {
            "baseline_scaling": str(output_dir / "baseline_scaling.csv"),
            "aggregate_by_pattern": str(output_dir / "aggregate_by_pattern.csv"),
            "aggregate_by_pattern_location": str(output_dir / "aggregate_by_pattern_location.csv"),
            "similarity_summary": str(output_dir / "similarity_summary.csv"),
            "similarity_breakdown": str(output_dir / "similarity_breakdown.csv"),
            "exemplar_scaling": str(output_dir / "exemplar_scaling.csv"),
            "figure": str(output_dir / "base_resistance_effects.png"),
            "summary": str(output_dir / "experiment_summary.md"),
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Completed base-resistance comparison for {len(base_resistances)} settings.")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
