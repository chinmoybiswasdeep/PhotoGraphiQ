"""Set-level helpers shared by the qubit flow finders and verifiers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ..ir import OpenGraph, QubitMeasurement


def odd_neighbourhood(og: OpenGraph, nodes: Iterable) -> frozenset:
    """Odd(K): vertices adjacent to an odd number of vertices of K."""
    result: set = set()
    for n in nodes:
        result ^= set(og.graph.neighbors(n))
    return frozenset(result)


def qubit_specs(og: OpenGraph) -> dict:
    """Measurement spec of every measured node; reject missing specs instead of guessing."""
    missing = [n for n in og.measured if n not in og.measurements]
    if missing:
        raise ValueError(f"Missing measurement specification for nodes {missing}")
    specs = {n: og.measurements[n] for n in og.measured}
    if not all(isinstance(m, QubitMeasurement) for m in specs.values()):
        raise TypeError("Qubit flows need QubitMeasurement specs")
    return specs


def layer_problems(og: OpenGraph, layers) -> tuple[list[str], dict]:
    """Check that ``layers`` partition the measured nodes; return problems and ranks.

    Rank r(u) is the layer index (earliest first); outputs get rank len(layers), so
    the induced strict order is u < v  iff  r(u) < r(v).
    """
    problems = []
    rank: dict = {}
    for i, layer in enumerate(layers):
        for n in layer:
            if n in rank:
                problems.append(f"node {n!r} appears in two layers")
            rank[n] = i
    if set(rank) != set(og.measured):
        problems.append("layers do not partition the measured (non-output) nodes")
    for n in og.outputs:
        rank[n] = len(layers)
    return problems, rank


def correction_map(og: OpenGraph, sets: Mapping, order: tuple) -> dict:
    """Pauli corrections of each outcome on nodes measured later and on outputs.

    Stabilizer components on nodes measured earlier (allowed by Pauli flow only for
    compatible Pauli measurements) act as phases there and are omitted.
    """
    position = {n: i for i, n in enumerate(order)}
    later_than = {
        u: {n for n in og.nodes if position.get(n, len(order)) > position[u]} for u in order
    }
    return {
        u: {
            "x": frozenset((set(sets[u]) - {u}) & later_than[u]),
            "z": frozenset((odd_neighbourhood(og, sets[u]) - {u}) & later_than[u]),
        }
        for u in order
    }


def flatten(layers) -> tuple:
    return tuple(n for layer in layers for n in layer)
