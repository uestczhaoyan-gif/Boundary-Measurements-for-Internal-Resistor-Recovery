from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "gnn" / "GNN_NOISE" / "rand_boundary_robustness_curve.svg"
X_LABELS = ["Clean", "40 dB", "30 dB", "20 dB"]
METRIC_FILES = [
    ROOT / "gnn" / "GNN_CMEI_INFERENCE" / "outputs" / "gnn_cmei_noiseft_rand_boundary_clean_20260401" / "training_data64Nodes_2_noiseft_rand_boundary_20260401" / "cmei_metrics.json",
    ROOT / "gnn" / "GNN_CMEI_INFERENCE" / "outputs" / "gnn_cmei_noiseft_rand_boundary_40dB_20260402" / "training_data64Nodes_2_noiseft_rand_boundary_20260401" / "cmei_metrics.json",
    ROOT / "gnn" / "GNN_CMEI_INFERENCE" / "outputs" / "gnn_cmei_noiseft_rand_boundary_30dB_20260402" / "training_data64Nodes_2_noiseft_rand_boundary_20260401" / "cmei_metrics.json",
    ROOT / "gnn" / "GNN_CMEI_INFERENCE" / "outputs" / "gnn_cmei_noiseft_rand_boundary_20db_20260401" / "training_data64Nodes_2_noiseft_rand_boundary_20260401" / "cmei_metrics.json",
]


def esc(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def load_series():
    import json

    rows = [json.loads(path.read_text(encoding="utf-8")) for path in METRIC_FILES]
    return [
        ("CMEI", "#0b84a5", [row["scores"]["CMEI"] for row in rows]),
        ("Macro F1", "#f28e2b", [row["macro_f1"] * 100.0 for row in rows]),
        ("Num Accuracy", "#59a14f", [row["num_accuracy"] * 100.0 for row in rows]),
        ("ID Recall", "#c0392b", [row["id_recall"] * 100.0 for row in rows]),
    ]


def build_svg():
    series = load_series()
    width, height = 1280, 820
    left, right, top, bottom = 110, 70, 95, 100
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_min, y_max = 70.0, 92.5

    def x_to_px(i):
        if len(X_LABELS) == 1:
            return left + plot_w / 2
        return left + i * plot_w / (len(X_LABELS) - 1)

    def y_to_px(v):
        return top + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        "<style>",
        'text { font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif; fill: #24313d; }',
        ".title { font-size: 34px; font-weight: 700; }",
        ".subtitle { font-size: 18px; fill: #5a6774; }",
        ".axis { font-size: 15px; fill: #566370; }",
        ".legend { font-size: 15px; font-weight: 600; }",
        ".value { font-size: 13px; font-weight: 600; }",
        "</style>",
        "</defs>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fbfaf6" />',
        f'<rect x="{left-35}" y="{top-35}" width="{plot_w+70}" height="{plot_h+70}" fill="#fffdf9" stroke="#e7ddd0" stroke-width="2" rx="24" />',
        f'<text x="{left-10}" y="58" class="title">Rand-Boundary Joint Robustness Curves</text>',
        f'<text x="{left-10}" y="86" class="subtitle">training_data64Nodes_2_noiseft_rand_boundary_20260401</text>',
    ]

    for tick in range(70, 95, 5):
        y = y_to_px(float(tick))
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#e8edf2" stroke-width="1" />')
        parts.append(f'<text x="{left-10}" y="{y+5:.2f}" text-anchor="end" class="axis">{tick}</text>')

    for i, label in enumerate(X_LABELS):
        x = x_to_px(i)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_h}" stroke="#f0f3f6" stroke-width="1" />')
        parts.append(f'<text x="{x:.2f}" y="{top+plot_h+34}" text-anchor="middle" class="axis">{esc(label)}</text>')

    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#8694a0" stroke-width="2" />')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#8694a0" stroke-width="2" />')
    parts.append(f'<text x="{left+plot_w/2:.2f}" y="{height-38}" text-anchor="middle" class="axis">Noise condition</text>')
    parts.append(f'<text x="40" y="{top+plot_h/2:.2f}" text-anchor="middle" class="axis" transform="rotate(-90 40 {top+plot_h/2:.2f})">Score (%)</text>')

    legend_x = left + plot_w - 220
    legend_y = top + 28
    parts.append(f'<rect x="{legend_x-18}" y="{legend_y-22}" width="205" height="136" fill="#fff7ea" stroke="#efd7b0" stroke-width="1" rx="18" />')

    for idx, (label, color, values) in enumerate(series):
        points = " ".join(f"{x_to_px(i):.2f},{y_to_px(v):.2f}" for i, v in enumerate(values))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" points="{points}" />')
        for i, v in enumerate(values):
            x = x_to_px(i)
            y = y_to_px(v)
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6.5" fill="{color}" stroke="#ffffff" stroke-width="2" />')
            dy = -12 if idx % 2 == 0 else 22
            parts.append(f'<text x="{x:.2f}" y="{y+dy:.2f}" text-anchor="middle" class="value" fill="{color}">{v:.1f}</text>')
        ly = legend_y + idx * 28
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x+28}" y2="{ly}" stroke="{color}" stroke-width="4" />')
        parts.append(f'<circle cx="{legend_x+14}" cy="{ly}" r="5" fill="{color}" stroke="#ffffff" stroke-width="1.5" />')
        parts.append(f'<text x="{legend_x+38}" y="{ly+5}" class="legend">{esc(label)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build_svg(), encoding="utf-8")
    print(f"Saved SVG to {OUTPUT}")


if __name__ == "__main__":
    main()
