from __future__ import annotations

import argparse
import json
from pathlib import Path


PALETTE = [
    "#0b84a5",
    "#f6c85f",
    "#6f4e7c",
    "#9dd866",
    "#ca472f",
    "#ffa056",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot robustness curves from archived JSON metrics.")
    parser.add_argument("--metric-key", required=True, help="JSON key to plot, e.g. cmei / test_macro_f1 / count_macro_f1")
    parser.add_argument("--metric-label", default="", help="Y-axis label shown on the chart.")
    parser.add_argument("--title", default="Noise Robustness Curves")
    parser.add_argument("--x-label", default="SNR (dB)")
    parser.add_argument("--x-values", nargs="+", type=float, required=True, help="X-axis values, e.g. 40 30 20")
    parser.add_argument("--multiply", type=float, default=1.0, help="Multiply each metric by this factor before plotting.")
    parser.add_argument(
        "--series",
        action="append",
        default=[],
        help="Series definition: label=path1,path2,...  Number of paths must match --x-values.",
    )
    parser.add_argument("--out-svg", required=True)
    return parser.parse_args()


def load_metric(path: Path, metric_key: str, multiply: float) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    if metric_key not in data:
        raise KeyError(f"{path} missing metric key: {metric_key}")
    return float(data[metric_key]) * multiply


def parse_series(items: list[str], x_count: int, metric_key: str, multiply: float) -> list[tuple[str, list[float]]]:
    parsed: list[tuple[str, list[float]]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --series value: {item}")
        label, raw_paths = item.split("=", 1)
        paths = [Path(p.strip()) for p in raw_paths.split(",") if p.strip()]
        if len(paths) != x_count:
            raise ValueError(f"{label}: expected {x_count} files, got {len(paths)}")
        values = [load_metric(path, metric_key, multiply) for path in paths]
        parsed.append((label, values))
    if not parsed:
        raise ValueError("At least one --series is required.")
    return parsed


def esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def fmt_value(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 10:
        return f"{v:.2f}"
    return f"{v:.3f}"


def build_svg(
    title: str,
    x_label: str,
    y_label: str,
    x_values: list[float],
    series: list[tuple[str, list[float]]],
) -> str:
    width, height = 1280, 820
    left, right, top, bottom = 110, 80, 110, 120
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_y = [v for _, values in series for v in values]
    y_min = min(all_y)
    y_max = max(all_y)
    if y_min == y_max:
        pad = max(abs(y_min) * 0.1, 1.0)
        y_min -= pad
        y_max += pad
    else:
        pad = (y_max - y_min) * 0.12
        y_min -= pad
        y_max += pad

    if len(x_values) == 1:
        x_min = x_values[0] - 1.0
        x_max = x_values[0] + 1.0
    else:
        x_min = min(x_values)
        x_max = max(x_values)
        if x_min == x_max:
            x_min -= 1.0
            x_max += 1.0

    def x_to_px(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def y_to_px(y: float) -> float:
        return top + plot_h - (y - y_min) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        "<style>",
        'text { font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif; fill: #22313f; }',
        ".title { font-size: 34px; font-weight: 700; }",
        ".axis { font-size: 15px; fill: #51606d; }",
        ".legend { font-size: 16px; font-weight: 600; }",
        ".value { font-size: 13px; font-weight: 600; }",
        "</style>",
        "</defs>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fbfaf6" />',
        f'<rect x="{left-35}" y="{top-35}" width="{plot_w+70}" height="{plot_h+70}" fill="#fffdf9" stroke="#e7ddd0" stroke-width="2" rx="24" />',
        f'<text x="{left-20}" y="62" class="title">{esc(title)}</text>',
    ]

    # grid
    ticks = 5
    for i in range(ticks + 1):
        y = top + plot_h * i / ticks
        value = y_max - (y_max - y_min) * i / ticks
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#e6ecf1" stroke-width="1" />')
        parts.append(f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" class="axis">{esc(fmt_value(value))}</text>')

    for x in x_values:
        px = x_to_px(x)
        parts.append(f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top+plot_h}" stroke="#eef2f5" stroke-width="1" />')
        parts.append(f'<text x="{px:.2f}" y="{top+plot_h+30}" text-anchor="middle" class="axis">{esc(fmt_value(x))}</text>')

    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#8694a0" stroke-width="2" />')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#8694a0" stroke-width="2" />')
    parts.append(f'<text x="{left+plot_w/2:.2f}" y="{height-42}" text-anchor="middle" class="axis">{esc(x_label)}</text>')
    parts.append(
        f'<text x="38" y="{top+plot_h/2:.2f}" text-anchor="middle" class="axis" transform="rotate(-90 38 {top+plot_h/2:.2f})">{esc(y_label)}</text>'
    )

    legend_x = left + plot_w - 10
    legend_y = top + 18

    for idx, (label, values) in enumerate(series):
        color = PALETTE[idx % len(PALETTE)]
        points = " ".join(f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for x, y in zip(x_values, values))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" points="{points}" />')
        for x, y in zip(x_values, values):
            px = x_to_px(x)
            py = y_to_px(y)
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="6.5" fill="{color}" stroke="#ffffff" stroke-width="2" />')
            parts.append(f'<text x="{px:.2f}" y="{py-12:.2f}" text-anchor="middle" class="value" fill="{color}">{esc(fmt_value(y))}</text>')
        ly = legend_y + idx * 28
        parts.append(f'<line x1="{legend_x-170}" y1="{ly}" x2="{legend_x-140}" y2="{ly}" stroke="{color}" stroke-width="4" />')
        parts.append(f'<circle cx="{legend_x-155}" cy="{ly}" r="5" fill="{color}" stroke="#ffffff" stroke-width="1.5" />')
        parts.append(f'<text x="{legend_x-130}" y="{ly+5}" class="legend">{esc(label)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    y_label = args.metric_label or args.metric_key
    series = parse_series(args.series, len(args.x_values), args.metric_key, args.multiply)
    svg = build_svg(args.title, args.x_label, y_label, args.x_values, series)
    out_path = Path(args.out_svg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    print(f"Saved SVG to {out_path}")


if __name__ == "__main__":
    main()
