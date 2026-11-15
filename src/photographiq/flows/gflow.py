"""Plane-aware generalised flow (gflow) of qubit open graphs.

Definition (Browne, Kashefi, Mhalla, Perdrix, NJP 9, 250, 2007; three-plane form
of Backens et al., Quantum 5, 421, 2021, "There and back again", Def. of gflow):
g: O^c -> P(I^c) with strict partial order < such that for every measured i
(G1) j in g(i), j != i  =>  i < j;
(G2) j in Odd(g(i)), j != i  =>  i < j;
(G3) XY: i not in g(i), i in Odd(g(i));  XZ: i in g(i), i in Odd(g(i));
     YZ: i in g(i), i not in Odd(g(i)).

Finder: the backward layered GF(2) search of Mhalla and Perdrix (ICALP 2008),
extended to XZ/YZ by Backens et al. (2021). For each still-unsolved node one GF(2)
system is solved; it is complete (if it stops early no gflow exists) and returns
maximally delayed layers. Complexity O(n^5) worst case for n nodes (n rounds, n
nodes, O(n^3) elimination) - adequate for the graph sizes simulated here.

The same layered search with Pauli labels implements Pauli flow (see ``pauli.py``).
"""

from __future__ import annotations

from collections.abc import Mapping

import networkx as nx
import numpy as np

from ..ir import Model, OpenGraph
from ._common import correction_map, flatten, layer_problems, odd_neighbourhood, qubit_specs
from .result import FlowAnalysisResult, unsupported

ALGORITHM = "gflow (Browne et al. 2007; Mhalla-Perdrix 2008 / Backens et al. 2021 search)"

# Self conditions: (x_u forced value or None if free, required parity of
# Odd(K)_u + [label == 'Y'] * x_u, or None if unconstrained).
_SELF = {
    "XY": (0, 1),
    "XZ": (1, 1),
    "YZ": (1, 0),
    "X": (None, 1),
    "Y": (None, 1),
    "Z": (1, None),
}


def solve_gf2(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray | None:
    """One solution of matrix @ x = rhs over GF(2) (free variables 0), or None."""
    rows, cols = matrix.shape
    aug = np.concatenate([matrix, rhs[:, None]], axis=1).astype(np.uint8) & 1
    pivots, r = [], 0
    for c in range(cols):
        if r == rows:
            break
        hits = np.flatnonzero(aug[r:, c])
        if not hits.size:
            continue
        p = r + hits[0]
        aug[[r, p]] = aug[[p, r]]
        mask = aug[:, c].astype(bool)
        mask[r] = False
        aug[mask] ^= aug[r]
        pivots.append(c)
        r += 1
    if aug[r:, -1].any():
        return None
    x = np.zeros(cols, dtype=np.uint8)
    for i, c in enumerate(pivots):
        x[c] = aug[i, -1]
    return x


def layered_search(og: OpenGraph, labels: Mapping) -> tuple[dict, tuple, tuple]:
    """Backward layered search; ``labels`` maps measured nodes to XY/XZ/YZ/X/Y/Z.

    With plane labels only this is the gflow search; Pauli labels X/Y/Z switch on the
    Pauli-flow relaxations (P1-P3 of Simmons 2021). Returns (correction sets, layers
    earliest first, unsolved nodes); the search failed iff unsolved is non-empty.
    """
    nodes = list(og.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    adj = nx.to_numpy_array(og.graph, nodelist=nodes, weight=None).astype(np.uint8)
    inputs = set(og.inputs)
    solved = set(og.outputs)
    sets: dict = {}
    layers: list[tuple] = []
    while unsolved := [n for n in nodes if n not in solved]:
        layer = []
        for u in unsolved:
            found = _solve_node(u, unsolved, solved, labels, inputs, nodes, index, adj)
            if found is not None:
                sets[u] = found
                layer.append(u)
        if not layer:
            return sets, tuple(layers), tuple(unsolved)
        solved.update(layer)
        layers.insert(0, tuple(layer))
    return sets, tuple(layers), ()


def _solve_node(u, unsolved, solved, labels, inputs, nodes, index, adj) -> frozenset | None:
    forced, parity = _SELF[labels[u]]
    if u in inputs:  # corrections live in I^c
        if forced == 1:
            return None
        forced = 0
    candidates = [
        w
        for w in nodes
        if w not in inputs and w != u and (w in solved or labels.get(w) in ("X", "Y"))
    ]
    if forced is None:
        candidates.append(u)
    cols = [index[w] for w in candidates]
    rows, rhs = [], []
    for v in unsolved:
        lab = labels[v]
        if v == u:
            if parity is None:
                continue
            target = parity
        elif lab == "Z":
            continue  # Z-measured nodes tolerate Z corrections (P2) and have no P3
        else:
            target = 0
        row = adj[index[v], cols].copy()
        if lab == "Y" and v in candidates:  # P3 / P9: x_v + Odd_v
            row[candidates.index(v)] ^= 1
        if forced == 1:
            target ^= int(adj[index[v], index[u]])
        rows.append(row)
        rhs.append(target)
    matrix = np.array(rows, dtype=np.uint8).reshape(len(rows), len(cols))
    x = solve_gf2(matrix, np.array(rhs, dtype=np.uint8))
    if x is None:
        return None
    chosen = {w for w, bit in zip(candidates, x) if bit}
    if forced == 1:
        chosen.add(u)
    return frozenset(chosen)


def verify_gflow(og: OpenGraph, sets: Mapping, layers) -> list[str]:
    """Check domain, codomain and (G1)-(G3) with plain set operations; [] means valid."""
    problems, rank = layer_problems(og, layers)
    specs = qubit_specs(og)
    if set(sets) != set(og.measured):
        problems.append("correction-set domain differs from the measured nodes")
    for u, g in sets.items():
        g = set(g)
        if g & set(og.inputs) or not g <= set(og.nodes):
            problems.append(f"g({u!r}) is not a subset of the non-input nodes")
        odd = odd_neighbourhood(og, g)
        for j in g - {u}:
            if not rank.get(u, -1) < rank.get(j, -1):
                problems.append(f"G1: {j!r} in g({u!r}) is not after {u!r}")
        for j in odd - {u}:
            if not rank.get(u, -1) < rank.get(j, -1):
                problems.append(f"G2: {j!r} in Odd(g({u!r})) is not after {u!r}")
        plane = specs[u].plane.value
        want = {"XY": (False, True), "XZ": (True, True), "YZ": (True, False)}[plane]
        if (u in g, u in odd) != want:
            problems.append(f"G3: plane condition {plane} fails at {u!r}")
    return problems


def find_gflow(og: OpenGraph) -> FlowAnalysisResult:
    """Maximally delayed plane-aware gflow, verified before returning."""
    if og.model is not Model.QUBIT:
        return unsupported(og.model, ALGORITHM, "gflow is a qubit-model notion")
    labels = {n: m.plane.value for n, m in qubit_specs(og).items()}
    sets, layers, unsolved = layered_search(og, labels)
    if unsolved:
        return FlowAnalysisResult(
            "not_certified",
            og.model,
            ALGORITHM,
            diagnostics={
                "reason": "no gflow exists (the layered GF(2) search is complete for gflow)",
                "complete": True,
                "uncorrectable": unsolved,
            },
        )
    problems = verify_gflow(og, sets, layers)
    if problems:  # pragma: no cover - would be an implementation bug
        raise AssertionError(f"gflow failed verification: {problems}")
    order = flatten(layers)
    return FlowAnalysisResult(
        "certified",
        og.model,
        ALGORITHM,
        order,
        layers,
        correction_map(og, sets, order),
        {"correction_sets": sets, "checks": ("domain", "codomain", "G1", "G2", "G3")},
        ("ideal qubits and noiseless CZ graph state", "planar (XY/XZ/YZ) measurements"),
        {"complete": True},
    )
