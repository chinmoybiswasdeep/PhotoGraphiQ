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
