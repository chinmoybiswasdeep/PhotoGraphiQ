"""Deterministic two-dimensional layouts for open graphs.

Coordinates are visualization metadata only: a :class:`Layout` never changes the
quantum semantics of an :class:`~photographiq.ir.OpenGraph`. All layouts use
mathematical axes (y grows upward) and are deterministic; the force-directed layout
is deterministic for a fixed ``seed``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import networkx as nx

from ..ir import OpenGraph


def _network(graph: OpenGraph | nx.Graph) -> nx.Graph:
    return graph.graph if isinstance(graph, OpenGraph) else nx.Graph(graph)


def node_key(node: Hashable) -> str:
    """JSON key of a node label (``str(node)``), the convention of ``OpenGraph.to_dict``."""
    return str(node)


def keyed(nodes: Iterable[Hashable]) -> dict[str, Hashable]:
    """Map ``str(node)`` to node, rejecting labels whose strings collide."""
    table: dict[str, Hashable] = {}
    for node in nodes:
        key = node_key(node)
        if key in table:
            raise ValueError(f"Node labels {table[key]!r} and {node!r} share the string {key!r}")
        table[key] = node
    return table


@dataclass(frozen=True)
class Layout:
    """Node positions ``{node: (x, y)}`` in arbitrary units, y upward."""

    positions: Mapping

    def __post_init__(self):
        clean = {}
        for node, xy in self.positions.items():
            x, y = (float(v) for v in xy)
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError(f"Position of {node!r} must be finite")
            clean[node] = (x, y)
        object.__setattr__(self, "positions", clean)

    def __getitem__(self, node: Hashable) -> tuple[float, float]:
        return self.positions[node]

    def require(self, nodes: Iterable[Hashable]) -> Layout:
        """Return self if every node has a position, else raise ``ValueError``."""
        missing = [n for n in nodes if n not in self.positions]
        if missing:
            raise ValueError(f"Layout lacks positions for {missing!r}")
        return self

    def to_dict(self) -> dict[str, list[float]]:
        """JSON-compatible ``{str(node): [x, y]}``."""
        keyed(self.positions)
        return {node_key(n): [x, y] for n, (x, y) in self.positions.items()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)

    @classmethod
    def from_dict(cls, data: Mapping, nodes: Iterable[Hashable]) -> Layout:
        """Inverse of :meth:`to_dict`; keys are matched to ``nodes`` by ``str``."""
        table = keyed(nodes)
        unknown = set(map(str, data)) - set(table)
        if unknown:
            raise ValueError(f"Layout names unknown nodes: {sorted(unknown)}")
        return cls({table[str(k)]: v for k, v in data.items()})

    @classmethod
    def from_json(cls, text: str, nodes: Iterable[Hashable]) -> Layout:
        return cls.from_dict(json.loads(text), nodes)


def grid(graph: OpenGraph | nx.Graph, columns: int | None = None) -> Layout:
    """Row-major grid in graph node order."""
    nodes = list(_network(graph))
    width = columns or max(1, math.ceil(math.sqrt(len(nodes))))
    return Layout({n: (i % width, -(i // width)) for i, n in enumerate(nodes)})


def circular(graph: OpenGraph | nx.Graph) -> Layout:
    """Nodes on the unit circle, counter-clockwise from (1, 0) in graph order."""
    nodes = list(_network(graph))
    count = max(len(nodes), 1)
    return Layout(
        {
            n: (math.cos(2 * math.pi * i / count), math.sin(2 * math.pi * i / count))
            for i, n in enumerate(nodes)
        }
    )


def _depths(graph: OpenGraph | nx.Graph) -> dict:
    """Breadth-first depth from the inputs (or each component's first node)."""
    network = _network(graph)
    sources = list(graph.inputs) if isinstance(graph, OpenGraph) else []
    depth: dict = {}
    order = {n: i for i, n in enumerate(network)}  # graph order; labels need not be comparable
    for component in sorted(
        nx.connected_components(network), key=lambda c: min(order[n] for n in c)
    ):
        roots = [n for n in sources if n in component] or [_first(network, component)]
        depth.update(nx.multi_source_dijkstra_path_length(network, roots, weight=_hop))
    return depth


def _hop(u, v, data) -> int:
    return 1


def _first(network: nx.Graph, nodes: Iterable[Hashable]) -> Hashable:
    order = {n: i for i, n in enumerate(network)}
    return min(nodes, key=order.__getitem__)


def _columns(network: nx.Graph, column: Mapping) -> list[list]:
    count = max(column.values(), default=-1) + 1
    columns: list[list] = [[] for _ in range(count)]
    for node in network:  # graph order inside each column
        columns[column[node]].append(node)
    return columns


def _place(columns: Sequence[Sequence]) -> Layout:
    """Column x = index, nodes centred vertically with unit spacing, top to bottom."""
    return Layout(
        {
            node: (x, (len(column) - 1) / 2 - y)
            for x, column in enumerate(columns)
            for y, node in enumerate(column)
        }
    )


def hierarchical(graph: OpenGraph | nx.Graph) -> Layout:
    """Columns by breadth-first distance from the inputs, left to right."""
    network = _network(graph)
    return _place(_columns(network, _depths(graph)))


def _flow_columns(graph: OpenGraph, order=None, layers=None) -> dict:
    """Column of every node from flow layers or a total order; unlisted nodes go last."""
    if layers is not None and order is not None:
        raise ValueError("Pass either measurement layers or a measurement order")
    if layers is None and order is None:
        raise ValueError("A layered layout needs measurement layers or an order")
    groups = [list(layer) for layer in layers] if layers is not None else [[n] for n in order]
    column = {}
    for i, group in enumerate(groups):
        for node in group:
            if node not in graph.graph or node in column:
                raise ValueError(f"Layer entry {node!r} is unknown or repeated")
            column[node] = i
    for node in graph.graph:
        column.setdefault(node, len(groups))
    return column


def layered(graph: OpenGraph, *, order=None, layers=None) -> Layout:
    """Columns by measurement layer (or position in a total order); outputs last."""
    return _place(_columns(graph.graph, _flow_columns(graph, order, layers)))


def force_directed(graph: OpenGraph | nx.Graph, *, seed: int = 0, iterations: int = 200) -> Layout:
    """NetworkX spring layout (unweighted), deterministic for a given seed."""
    network = _network(graph)
    return Layout(nx.spring_layout(network, seed=seed, iterations=iterations, weight=None))


def publication(graph: OpenGraph, *, order=None, layers=None, sweeps: int = 4) -> Layout:
    """Aligned integer grid for manuscripts.

    Columns come from flow layers/order when given, otherwise from breadth-first
    depth from the inputs. Inside each column nodes are ordered by the barycentre of
    their neighbours in adjacent columns (alternating sweeps; ties keep graph order),
    which reduces but does not minimise edge crossings.
    """
    network = graph.graph
    column = (
        _flow_columns(graph, order, layers)
        if order is not None or layers is not None
        else _depths(graph)
    )
    columns = _columns(network, column)
    rank = {n: i for i, n in enumerate(network)}
    for sweep in range(sweeps):
        indices = range(1, len(columns)) if sweep % 2 == 0 else range(len(columns) - 2, -1, -1)
        for c in indices:
            ref = columns[c - 1] if sweep % 2 == 0 else columns[c + 1]
            position = {n: i for i, n in enumerate(ref)}

            def centre(node, position=position):
                near = [position[m] for m in network[node] if m in position]
                return (sum(near) / len(near) if near else math.inf, rank[node])

            columns[c] = sorted(columns[c], key=centre)
    return _place(columns)


def manual(graph: OpenGraph | nx.Graph, positions: Mapping) -> Layout:
    """User-supplied positions; every node must be present."""
    return Layout(dict(positions)).require(_network(graph))


CLEARANCE = 0.34  # layout units an edge keeps from non-endpoint node centres
_BEND_STEP = 0.16
_BEND_TRIES = 10


def _clearance(a, b, offset: float, others) -> float:
    """Smallest distance of the quadratic curve a -> b (control offset ``offset``) to ``others``."""
    (x1, y1), (x2, y2) = a, b
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    cx, cy = (x1 + x2) / 2 + offset * dy / length, (y1 + y2) / 2 - offset * dx / length
    best = math.inf
    for i in range(1, 40):
        t = i / 40
        u = 1 - t
        x = u * u * x1 + 2 * t * u * cx + t * t * x2
        y = u * u * y1 + 2 * t * u * cy + t * t * y2
        for ox, oy in others:
            best = min(best, math.hypot(x - ox, y - oy))
    return best


def route(
    layout: Layout, source: Hashable, target: Hashable, base: float = 0.0, *, clearance=CLEARANCE
) -> float:
    """Control-point offset of a quadratic edge ``source -> target`` that avoids other nodes.

    The control point is the chord midpoint plus ``offset`` times the unit normal
    ``(dy, -dx) / |d|`` (y up); this is Matplotlib's ``arc3`` convention (``rad =
    offset / |d|``) and Cytoscape's ``control-point-distances`` (pixels, y down). The
    curve peaks at ``offset / 2``. Starting from ``base`` (0 for straight CZ edges, the
    signed default arc of overlay arrows) the offset grows away from the chord, on
    both sides when ``base`` is 0 (arrows try their own side first, then the mirrored
    side, then flatter arcs), until the curve stays ``clearance`` away from every other node; if no tried
    offset is clear, the clearest one is returned.
    """
    pos = layout.positions
    a, b = pos[source], pos[target]
    others = [xy for n, xy in pos.items() if n not in (source, target)]
    if not others:
        return base
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    if length == 0:
        return base
    signs = (1.0, -1.0) if base == 0 else (math.copysign(1.0, base),)
    candidates = [base] + [
        base + s * j * _BEND_STEP * 2 for j in range(1, _BEND_TRIES) for s in signs
    ]
    if base != 0:  # the preferred side is blocked: mirror the arc to the other side
        candidates += [-c for c in candidates] + [base / 2, -base / 2, base / 4, -base / 4]
    scored = []
    for offset in candidates:
        gap = _clearance(a, b, offset, others)
        if gap >= clearance:
            return offset
        scored.append((gap, -abs(offset), offset))
    return max(scored)[2]


LAYOUTS: dict[str, Callable[..., Layout]] = {
    "grid": grid,
    "circular": circular,
    "hierarchical": hierarchical,
    "layered": layered,
    "force": force_directed,
    "publication": publication,
    "manual": manual,
}


def compute_layout(graph: OpenGraph, method: str = "publication", **options) -> Layout:
    """Dispatch to a named layout in :data:`LAYOUTS`."""
    if method not in LAYOUTS:
        raise ValueError(f"Unknown layout {method!r}; choose from {sorted(LAYOUTS)}")
    return LAYOUTS[method](graph, **options).require(_network(graph))
