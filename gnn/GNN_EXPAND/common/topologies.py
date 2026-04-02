from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TopologySpec:
    key: str
    title: str
    num_nodes: int
    resistor_edges: tuple[tuple[int, int], ...]
    message_edges: tuple[tuple[int, int], ...]
    node_coords: tuple[tuple[float, float], ...]
    boundary_nodes_clockwise: tuple[int, ...]
    notes: str = ""

    @property
    def num_resistors(self) -> int:
        return len(self.resistor_edges)

    @property
    def num_boundary_nodes(self) -> int:
        return len(self.boundary_nodes_clockwise)


def _normalize_positions(coord_to_id: dict[tuple[int, int], int], positions: dict[tuple[int, int], tuple[float, float]]):
    xs = [xy[0] for xy in positions.values()]
    ys = [xy[1] for xy in positions.values()]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    node_coords = [None] * len(coord_to_id)
    for coord, node_id in coord_to_id.items():
        x_raw, y_raw = positions[coord]
        node_coords[node_id] = ((x_raw - min_x) / span_x, (y_raw - min_y) / span_y)
    return tuple(node_coords)


def _select_boundary_nodes_clockwise(
    coord_to_id: dict[tuple[int, int], int],
    active_coords: set[tuple[int, int]],
    positions: dict[tuple[int, int], tuple[float, float]],
):
    boundary_coords = set()
    for coord in active_coords:
        r, c = coord
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            if (r + dr, c + dc) not in active_coords:
                boundary_coords.add(coord)
                break

    cx = sum(positions[coord][0] for coord in active_coords) / max(len(active_coords), 1)
    cy = sum(positions[coord][1] for coord in active_coords) / max(len(active_coords), 1)

    def angle_key(coord: tuple[int, int]):
        return (
            -math.atan2(positions[coord][1] - cy, positions[coord][0] - cx),
            coord[0],
            coord[1],
        )

    ordered = sorted(boundary_coords, key=angle_key)

    if ordered:
        start_coord = min(ordered, key=lambda coord: (coord[0], coord[1]))
        start_idx = ordered.index(start_coord)
        ordered = ordered[start_idx:] + ordered[:start_idx]

    return tuple(coord_to_id[coord] for coord in ordered)


def _build_grid_edges(rows: int, cols: int, active_coords: set[tuple[int, int]] | None = None):
    if active_coords is None:
        active_coords = {(r, c) for r in range(rows) for c in range(cols)}
    coord_to_id = {coord: idx for idx, coord in enumerate(sorted(active_coords))}
    resistor_edges: list[tuple[int, int]] = []
    message_edges: list[tuple[int, int]] = []

    for r, c in sorted(active_coords):
        src = coord_to_id[(r, c)]
        for dr, dc in ((0, 1), (1, 0)):
            nbr = (r + dr, c + dc)
            if nbr in coord_to_id:
                dst = coord_to_id[nbr]
                resistor_edges.append((src, dst))
                message_edges.append((src, dst))
                message_edges.append((dst, src))
    positions = {coord: (float(coord[1]), float(-coord[0])) for coord in active_coords}
    node_coords = _normalize_positions(coord_to_id, positions)
    boundary_nodes = _select_boundary_nodes_clockwise(coord_to_id, active_coords, positions)
    return coord_to_id, tuple(resistor_edges), tuple(message_edges), node_coords, boundary_nodes


def make_grid_topology(key: str, title: str, rows: int, cols: int, notes: str = "") -> TopologySpec:
    _coord_to_id, resistor_edges, message_edges, node_coords, boundary_nodes = _build_grid_edges(rows, cols)
    return TopologySpec(
        key=key,
        title=title,
        num_nodes=rows * cols,
        resistor_edges=resistor_edges,
        message_edges=message_edges,
        node_coords=node_coords,
        boundary_nodes_clockwise=boundary_nodes,
        notes=notes,
    )


def make_corner_cut_topology(
    key: str,
    title: str,
    rows: int,
    cols: int,
    cut_coords: set[tuple[int, int]],
    notes: str = "",
) -> TopologySpec:
    active_coords = {(r, c) for r in range(rows) for c in range(cols)} - set(cut_coords)
    _coord_to_id, resistor_edges, message_edges, node_coords, boundary_nodes = _build_grid_edges(
        rows,
        cols,
        active_coords=active_coords,
    )
    return TopologySpec(
        key=key,
        title=title,
        num_nodes=len(active_coords),
        resistor_edges=resistor_edges,
        message_edges=message_edges,
        node_coords=node_coords,
        boundary_nodes_clockwise=boundary_nodes,
        notes=notes,
    )


def make_honeycomb_topology(key: str, title: str, rows: int, cols: int, notes: str = "") -> TopologySpec:
    coord_to_id = {(r, c): r * cols + c for r in range(rows) for c in range(cols)}
    undirected: set[tuple[int, int]] = set()
    message_edges: list[tuple[int, int]] = []
    active_coords = set(coord_to_id)

    for r in range(rows):
        for c in range(cols):
            src = coord_to_id[(r, c)]
            if r % 2 == 0:
                nbrs = [(r, c + 1), (r + 1, c), (r + 1, c - 1)]
            else:
                nbrs = [(r, c + 1), (r + 1, c), (r + 1, c + 1)]
            for rr, cc in nbrs:
                if (rr, cc) not in coord_to_id:
                    continue
                dst = coord_to_id[(rr, cc)]
                edge = (min(src, dst), max(src, dst))
                if edge not in undirected:
                    undirected.add(edge)
                message_edges.append((src, dst))
                message_edges.append((dst, src))

    resistor_edges = tuple(sorted(undirected))
    positions = {
        coord: (float(coord[1]) + 0.5 * float(coord[0] % 2), -float(coord[0]) * 0.8660254037844386)
        for coord in active_coords
    }
    node_coords = _normalize_positions(coord_to_id, positions)
    boundary_nodes = _select_boundary_nodes_clockwise(coord_to_id, active_coords, positions)
    return TopologySpec(
        key=key,
        title=title,
        num_nodes=rows * cols,
        resistor_edges=resistor_edges,
        message_edges=tuple(message_edges),
        node_coords=node_coords,
        boundary_nodes_clockwise=boundary_nodes,
        notes=notes,
    )


_STAGE4_CUT_COORDS = {
    (0, 0), (0, 1), (1, 0),
    (0, 7), (0, 8), (1, 8),
    (7, 0), (8, 0), (8, 1),
    (7, 8), (8, 7), (8, 8),
}


TOPOLOGY_REGISTRY: dict[str, TopologySpec] = {
    "square_10x10": make_grid_topology(
        key="square_10x10",
        title="Stage1 Square 10x10",
        rows=10,
        cols=10,
        notes="纯规模扩展：10x10 正方形网格，100 节点。",
    ),
    "rect_6x10": make_grid_topology(
        key="rect_6x10",
        title="Stage2 Rectangular 6x10",
        rows=6,
        cols=10,
        notes="打破对称性：6x10 长方形网格，60 节点。",
    ),
    "honeycomb_63": make_honeycomb_topology(
        key="honeycomb_63",
        title="Stage3 Honeycomb 7x9",
        rows=7,
        cols=9,
        notes="复杂拓扑：7x9 蜂窝状邻接，63 节点。",
    ),
    "circlecut_69": make_corner_cut_topology(
        key="circlecut_69",
        title="Stage4 Circle-Cut 9x9",
        rows=9,
        cols=9,
        cut_coords=_STAGE4_CUT_COORDS,
        notes="迁移目标拓扑：9x9 角点裁切近似圆形，69 节点。",
    ),
}


def get_topology(key: str) -> TopologySpec:
    if key not in TOPOLOGY_REGISTRY:
        raise KeyError(f"Unknown topology key: {key}")
    return TOPOLOGY_REGISTRY[key]
