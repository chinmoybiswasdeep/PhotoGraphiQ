"""Scalar-exact qubit ZX diagrams: tensor semantics and a sound subset of rewrites.

Semantics (Coecke-Kissinger conventions with exact scalars):

* Z spider, phase ``a``, ``n`` legs: ``|0..0><0..0| + e^{ia} |1..1><1..1|`` (as an n-leg tensor).
* X spider: the same with ``|+>, |->`` (normalized), i.e. ``H`` on every leg of a Z spider.
* Simple edge: identity; Hadamard edge: ``H = [[1,1],[1,-1]]/sqrt(2)``.
* Boundary vertices have exactly one edge. :meth:`Diagram.matrix` returns the
  ``2**n_out x 2**n_in`` linear map (first boundary most significant) times ``Diagram.scalar``.

Every rewrite checks its precondition, multiplies ``scalar`` by the exact factor and appends a
:class:`RewriteStep` to ``Diagram.trace``. Rules: spider fusion, self-loop removal, identity
removal (with simple/Hadamard edge composition, which includes Hadamard-Hadamard
cancellation), colour change, state copy (Pauli states), bialgebra (2x2), Hopf, and
cancellation of parallel Hadamard edges. Pi-copy, local complementation/pivoting and any
CV-ZX identities are not implemented. Tensor evaluation is dense and exponential.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .qubit import E, M, N, QubitPattern, S, X, Z, _parity, effective_angle

__all__ = [
    "Diagram",
    "EdgeType",
    "RewriteError",
    "RewriteStep",
    "VertexType",
    "bialgebra",
    "cancel_hadamard_pair",
    "color_change",
    "copy_state",
    "fuse",
    "graph_state_diagram",
    "hopf",
    "pattern_branch_diagram",
    "remove_identity",
    "remove_self_loops",
    "simplify_spiders",
]

_TWO_PI = 2 * math.pi
_H = np.array([[1, 1], [1, -1]], complex) / math.sqrt(2)


class VertexType(str, enum.Enum):
    BOUNDARY = "B"
    Z = "Z"
    X = "X"


class EdgeType(str, enum.Enum):
    SIMPLE = "S"
    HADAMARD = "H"

    def then(self, other: EdgeType) -> EdgeType:
        """Composition of two edges through an identity: H then H = S."""
        return EdgeType.SIMPLE if self is other else EdgeType.HADAMARD


class RewriteError(ValueError):
    """Rewrite precondition is not satisfied."""


@dataclass(frozen=True)
class RewriteStep:
    """One applied rule: name, involved vertices and the exact scalar factor introduced."""

    rule: str
    vertices: tuple
    factor: complex


def _phase(a: float) -> float:
    return float(a) % _TWO_PI


def _is_zero(a: float, tol: float = 1e-12) -> bool:
    a = _phase(a)
    return min(a, _TWO_PI - a) < tol


@dataclass
class Diagram:
    """Open ZX multigraph. Vertices and edges have integer ids; self-loops are allowed."""

    vertices: dict = field(default_factory=dict)  # id -> [VertexType, phase]
    edges: dict = field(default_factory=dict)  # id -> (u, v, EdgeType)
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    scalar: complex = 1.0 + 0j
    trace: list = field(default_factory=list)
    _next: int = 0

    # ----------------------------------------------------------------- construction
    def _id(self) -> int:
        self._next += 1
        return self._next - 1

    def add_vertex(self, kind: VertexType | str, phase: float = 0.0) -> int:
        kind = VertexType(kind)
        if kind is VertexType.BOUNDARY and phase:
            raise ValueError("Boundaries carry no phase")
        v = self._id()
        self.vertices[v] = [kind, _phase(phase)]
        return v

    def add_edge(self, u: int, v: int, kind: EdgeType | str = EdgeType.SIMPLE) -> int:
        if u not in self.vertices or v not in self.vertices:
            raise KeyError("Unknown vertex")
        e = self._id()
        self.edges[e] = (u, v, EdgeType(kind))
        return e

    def add_input(self) -> int:
        b = self.add_vertex(VertexType.BOUNDARY)
        self.inputs.append(b)
        return b

    def add_output(self) -> int:
        b = self.add_vertex(VertexType.BOUNDARY)
        self.outputs.append(b)
        return b

    def copy(self) -> Diagram:
        return Diagram(
            {v: list(d) for v, d in self.vertices.items()},
            dict(self.edges),
            list(self.inputs),
            list(self.outputs),
            self.scalar,
            list(self.trace),
            self._next,
        )

    # ----------------------------------------------------------------- queries
    def kind(self, v: int) -> VertexType:
        return self.vertices[v][0]

    def phase(self, v: int) -> float:
        return self.vertices[v][1]

    def incident(self, v: int) -> list[int]:
        return [e for e, (a, b, _) in self.edges.items() if v in (a, b)]

    def degree(self, v: int) -> int:
        return sum((a == v) + (b == v) for a, b, _ in self.edges.values())

    def between(self, u: int, v: int, kind: EdgeType | None = None) -> list[int]:
        return [
            e
            for e, (a, b, t) in self.edges.items()
            if {a, b} == {u, v} and u != v and (kind is None or t is kind)
        ]

    def loops(self, v: int) -> list[int]:
        return [e for e, (a, b, _) in self.edges.items() if a == b == v]

    def other(self, e: int, v: int) -> int:
        a, b, _ = self.edges[e]
        return b if a == v else a

    def spiders(self) -> list[int]:
        return [v for v, (k, _) in self.vertices.items() if k is not VertexType.BOUNDARY]

    def to_networkx(self) -> nx.MultiGraph:
        """Visualization view: vertex attrs ``kind``/``phase``, edge attr ``kind``."""
        g = nx.MultiGraph()
        for v, (k, p) in self.vertices.items():
            g.add_node(v, kind=k.value, phase=p)
        for e, (a, b, t) in self.edges.items():
            g.add_edge(a, b, key=e, kind=t.value)
        return g

    def validate(self) -> Diagram:
        boundaries = [v for v in self.vertices if self.kind(v) is VertexType.BOUNDARY]
        if sorted(boundaries) != sorted(self.inputs + self.outputs):
            raise ValueError("Every boundary must be listed once as an input or output")
        for b in boundaries:
            if self.degree(b) != 1:
                raise ValueError(f"Boundary {b} must have exactly one edge")
        return self

    # ----------------------------------------------------------------- semantics
    def matrix(self) -> np.ndarray:
        """Exact linear map ``2**len(outputs) x 2**len(inputs)`` including ``scalar``."""
        self.validate()
        tensors: list[tuple[np.ndarray, list]] = []
        ends: dict[int, list] = {v: [] for v in self.vertices}
        for e, (a, b, t) in self.edges.items():
            la, lb = (e, 0), (e, 1)
            ends[a].append(la)
            ends[b].append(lb)
            tensors.append((np.eye(2, dtype=complex) if t is EdgeType.SIMPLE else _H, [la, lb]))
        for v, (k, p) in self.vertices.items():
            if k is not VertexType.BOUNDARY:
                tensors.append((_spider(k, p, len(ends[v])), ends[v]))
        open_labels = [ends[b][0] for b in self.outputs + self.inputs]
        result = _contract(tensors, open_labels)
        return self.scalar * result.reshape(2 ** len(self.outputs), 2 ** len(self.inputs))

    # ----------------------------------------------------------------- combinators
    def _embed(self, other: Diagram) -> dict:
        mapping = {}
        for v, (k, p) in other.vertices.items():
            mapping[v] = self.add_vertex(k, p)
        for a, b, t in other.edges.values():
            self.add_edge(mapping[a], mapping[b], t)
        self.scalar *= other.scalar
        return mapping

    def tensor(self, other: Diagram) -> Diagram:
        """Parallel composition: inputs and outputs of ``self`` first."""
        result = self.copy()
        mapping = result._embed(other)
        result.inputs += [mapping[b] for b in other.inputs]
        result.outputs += [mapping[b] for b in other.outputs]
        return result

    def then(self, other: Diagram) -> Diagram:
        """Sequential composition: ``self`` first, its outputs plugged into ``other``'s inputs."""
        if len(self.outputs) != len(other.inputs):
            raise ValueError("Output/input counts differ")
        result = self.copy()
        mapping = result._embed(other)
        for out, inp in zip(list(result.outputs), [mapping[b] for b in other.inputs]):
            (e1,), (e2,) = result.incident(out), result.incident(inp)
            u, v = result.other(e1, out), result.other(e2, inp)
            kind = result.edges[e1][2].then(result.edges[e2][2])
            for e in (e1, e2):
                del result.edges[e]
            del result.vertices[out], result.vertices[inp]
            result.add_edge(u, v, kind)
        result.inputs = list(self.inputs)
        result.outputs = [mapping[b] for b in other.outputs]
        return result


def _spider(kind: VertexType, phase: float, n: int) -> np.ndarray:
    t = np.zeros((2,) * n, complex)
    if n == 0:
        return np.array(1 + np.exp(1j * phase))
    t[(0,) * n] = 1
    t[(1,) * n] += np.exp(1j * phase)
    if kind is VertexType.X:
        for axis in range(n):
            t = np.moveaxis(np.tensordot(_H, t, axes=([1], [axis])), 0, axis)
    return t


def _contract(tensors: list, open_labels: list) -> np.ndarray:
    """Greedy pairwise contraction of a labelled tensor network (each label used <= twice)."""
    tensors = list(tensors)
    if not tensors:
        return np.ones(())
    while len(tensors) > 1:
        best = None
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                shared = set(tensors[i][1]) & set(tensors[j][1])
                if shared:
                    size = len(set(tensors[i][1]) ^ set(tensors[j][1]))
                    if best is None or size < best[0]:
                        best = (size, i, j)
        if best is None:  # disconnected pieces: outer product
            i, j = 0, 1
        else:
            _, i, j = best
        (a, la), (b, lb) = tensors[i], tensors[j]
        out = [x for x in la if x not in lb] + [x for x in lb if x not in la]
        letters = {x: k for k, x in enumerate(dict.fromkeys(la + lb))}
        c = np.einsum(
            a, [letters[x] for x in la], b, [letters[x] for x in lb], [letters[x] for x in out]
        )
        tensors = [t for k, t in enumerate(tensors) if k not in (i, j)] + [(c, out)]
    t, labels = tensors[0]
    return np.transpose(t, [labels.index(x) for x in open_labels]) if open_labels else t


# --------------------------------------------------------------------------- rewrites


def _spider_check(d: Diagram, *vs: int) -> None:
    for v in vs:
        if v not in d.vertices or d.kind(v) is VertexType.BOUNDARY:
            raise RewriteError(f"{v} is not a spider")


def _record(d: Diagram, rule: str, vertices: tuple, factor: complex) -> Diagram:
    d.scalar *= factor
    d.trace.append(RewriteStep(rule, vertices, complex(factor)))
    return d


def fuse(d: Diagram, u: int, v: int) -> Diagram:
    """(f) Merge same-colour spiders joined by a simple edge; phases add. Factor 1.

    Further u-v edges become self-loops of the merged spider (remove with
    :func:`remove_self_loops`).
    """
    _spider_check(d, u, v)
    if d.kind(u) is not d.kind(v):
        raise RewriteError("Fusion needs spiders of the same colour")
    simple = d.between(u, v, EdgeType.SIMPLE)
    if not simple:
        raise RewriteError("Fusion needs a simple edge between the spiders")
    d = d.copy()
    del d.edges[simple[0]]
    for e, (a, b, t) in list(d.edges.items()):
        d.edges[e] = (u if a == v else a, u if b == v else b, t)
    d.vertices[u][1] = _phase(d.phase(u) + d.phase(v))
    del d.vertices[v]
    return _record(d, "fuse", (u, v), 1)


def remove_self_loops(d: Diagram, v: int) -> Diagram:
    """Delete self-loops of ``v``: simple loops factor 1; each Hadamard loop adds pi, 1/sqrt 2."""
    _spider_check(d, v)
    loops = d.loops(v)
    if not loops:
        raise RewriteError("No self-loop")
    d = d.copy()
    hadamards = sum(d.edges[e][2] is EdgeType.HADAMARD for e in loops)
    for e in loops:
        del d.edges[e]
    d.vertices[v][1] = _phase(d.phase(v) + math.pi * hadamards)
    return _record(d, "self-loop", (v,), 2 ** (-hadamards / 2))


def remove_identity(d: Diagram, v: int) -> Diagram:
    """(id) Remove a phase-0 spider with two distinct non-loop edges; edges compose. Factor 1.

    Two Hadamard edges compose to a simple edge (Hadamard cancellation, ``H H = I``).
    """
    _spider_check(d, v)
    edges = d.incident(v)
    if not _is_zero(d.phase(v)) or len(edges) != 2 or d.loops(v):
        raise RewriteError("Identity removal needs a phase-0 spider with two plain edges")
    (e1, e2), d = edges, d.copy()
    a, b = d.other(e1, v), d.other(e2, v)
    kind = d.edges[e1][2].then(d.edges[e2][2])
    del d.edges[e1], d.edges[e2], d.vertices[v]
    d.add_edge(a, b, kind)
    return _record(d, "identity", (v,), 1)


def color_change(d: Diagram, v: int) -> Diagram:
    """(h) Swap the colour of ``v`` and toggle every non-loop incident edge. Factor 1."""
    _spider_check(d, v)
    d = d.copy()
    d.vertices[v][0] = VertexType.X if d.kind(v) is VertexType.Z else VertexType.Z
    for e in d.incident(v):
        a, b, t = d.edges[e]
        if a != b:
            d.edges[e] = (a, b, EdgeType.SIMPLE if t is EdgeType.HADAMARD else EdgeType.HADAMARD)
    return _record(d, "color-change", (v,), 1)


def copy_state(d: Diagram, state: int, v: int) -> Diagram:
    """Copy a one-legged Pauli state (phase 0 or pi) through an opposite-colour spider.

    Factor ``e^{i a alpha} 2^{(1-n)/2}`` for state phase ``a pi``, spider phase ``alpha`` and
    ``n`` remaining legs of ``v`` (which must have no self-loops).
    """
    _spider_check(d, state, v)
    edges = d.incident(state)
    if d.kind(state) is d.kind(v) or len(edges) != 1 or d.edges[edges[0]][2] is EdgeType.HADAMARD:
        raise RewriteError("State copy needs an opposite-colour one-leg state on a simple edge")
    if d.other(edges[0], state) != v or d.loops(v):
        raise RewriteError("State must be attached to v, which must have no loops")
    a = d.phase(state) / math.pi
    if not (_is_zero(d.phase(state)) or _is_zero(d.phase(state) - math.pi)):
        raise RewriteError("Only Pauli (phase 0 or pi) states copy")
    bit = round(a) % 2
    color = d.kind(state)
    d = d.copy()
    alpha = d.phase(v)
    del d.edges[edges[0]], d.vertices[state]
    rest = d.incident(v)
    for e in rest:
        w, t = d.other(e, v), d.edges[e][2]
        new = d.add_vertex(color, bit * math.pi)
        del d.edges[e]
        d.add_edge(new, w, t)
    del d.vertices[v]
    factor = np.exp(1j * bit * alpha) * 2 ** ((1 - len(rest)) / 2)
    return _record(d, "copy", (state, v), factor)


def bialgebra(d: Diagram, u: int, v: int) -> Diagram:
    """(b) Phase-0 Z spider u and X spider v, one simple edge, two other legs each. Factor sqrt 2.

    The pair becomes X spiders on u's legs and Z spiders on v's legs, fully connected.
    """
    _spider_check(d, u, v)
    if d.kind(u) is not VertexType.Z or d.kind(v) is not VertexType.X:
        raise RewriteError("Bialgebra needs (Z, X)")
    link = d.between(u, v)
    if not (_is_zero(d.phase(u)) and _is_zero(d.phase(v))):
        raise RewriteError("Bialgebra needs phase-0 spiders")
    if len(link) != 1 or d.edges[link[0]][2] is not EdgeType.SIMPLE or d.loops(u) or d.loops(v):
        raise RewriteError("Bialgebra needs exactly one simple u-v edge and no loops")
    legs_u = [e for e in d.incident(u) if e != link[0]]
    legs_v = [e for e in d.incident(v) if e != link[0]]
    if len(legs_u) != 2 or len(legs_v) != 2:
        raise RewriteError("Only the 2x2 bialgebra is implemented")
    d = d.copy()
    xs: list[int] = []
    zs: list[int] = []
    for legs, side, kind in ((legs_u, u, VertexType.X), (legs_v, v, VertexType.Z)):
        for e in legs:
            w, t = d.other(e, side), d.edges[e][2]
            new = d.add_vertex(kind)
            del d.edges[e]
            d.add_edge(new, w, t)
            (xs if kind is VertexType.X else zs).append(new)
    del d.edges[link[0]], d.vertices[u], d.vertices[v]
    for x in xs:
        for z in zs:
            d.add_edge(x, z)
    return _record(d, "bialgebra", (u, v), math.sqrt(2))


def hopf(d: Diagram, u: int, v: int) -> Diagram:
    """(Hopf) Remove two parallel simple edges between a Z and an X spider. Factor 1/2."""
    _spider_check(d, u, v)
    if {d.kind(u), d.kind(v)} != {VertexType.Z, VertexType.X}:
        raise RewriteError("Hopf needs one Z and one X spider")
    simple = d.between(u, v, EdgeType.SIMPLE)
    if len(simple) < 2:
        raise RewriteError("Hopf needs two parallel simple edges")
    d = d.copy()
    del d.edges[simple[0]], d.edges[simple[1]]
    return _record(d, "hopf", (u, v), 0.5)


def cancel_hadamard_pair(d: Diagram, u: int, v: int) -> Diagram:
    """Remove two parallel Hadamard edges between same-colour spiders. Factor 1/2.

    (Colour change of ``v`` followed by Hopf.)
    """
    _spider_check(d, u, v)
    if d.kind(u) is not d.kind(v):
        raise RewriteError("Hadamard-pair cancellation needs same-colour spiders")
    had = d.between(u, v, EdgeType.HADAMARD)
    if len(had) < 2:
        raise RewriteError("Need two parallel Hadamard edges")
    d = d.copy()
    del d.edges[had[0]], d.edges[had[1]]
    return _record(d, "hadamard-pair", (u, v), 0.5)


def simplify_spiders(d: Diagram, *, max_steps: int = 10_000) -> Diagram:
    """Apply fusion, self-loop removal and identity removal to a fixed point."""
    for _ in range(max_steps):
        for v in d.spiders():
            if d.loops(v):
                d = remove_self_loops(d, v)
                break
            fusable = [
                w
                for e in d.incident(v)
                if d.edges[e][2] is EdgeType.SIMPLE
                and (w := d.other(e, v)) != v
                and d.kind(w) is d.kind(v)
            ]
            if fusable:
                d = fuse(d, v, fusable[0])
                break
            try:
                d = remove_identity(d, v)
                break
            except RewriteError:
                continue
        else:
            return d
    raise RuntimeError("simplify_spiders did not terminate")


# --------------------------------------------------------------------------- conversions


def graph_state_diagram(graph: nx.Graph, order: Sequence | None = None) -> Diagram:
    """Diagram equal to ``|G>`` (outputs in ``order``): Z spiders, Hadamard edges, exact scalar.

    The bare network evaluates to ``2^{(n-|E|)/2} |G>``, hence ``scalar = 2^{(|E|-n)/2}``.
    """
    order = tuple(graph.nodes if order is None else order)
    d = Diagram()
    spider = {v: d.add_vertex(VertexType.Z) for v in order}
    for v in order:
        d.add_edge(spider[v], d.add_output())
    for a, b in graph.edges:
        d.add_edge(spider[a], spider[b], EdgeType.HADAMARD)
    d.scalar = 2 ** ((graph.number_of_edges() - len(order)) / 2)
    return d


def _effect(d: Diagram, wire: int, plane, angle: float) -> None:
    """Attach the bra of the outcome-0 state of ``O(plane, angle)`` with exact scalar."""
    from ..ir import Plane

    plane = Plane(plane)
    if plane is Plane.XY:  # (<0| + e^{-ia}<1|)/sqrt2 = Z(-a)/sqrt2
        d.add_edge(wire, d.add_vertex(VertexType.Z, -angle))
        d.scalar /= math.sqrt(2)
        return
    if plane is Plane.XZ:  # <XZ,a| = <YZ,a| S with S = Z(pi/2)
        s = d.add_vertex(VertexType.Z, math.pi / 2)
        d.add_edge(wire, s)
        wire = s
    # <YZ,a| = e^{-ia/2}/sqrt2 * X(a) effect
    d.add_edge(wire, d.add_vertex(VertexType.X, angle))
    d.scalar *= np.exp(-0.5j * angle) / math.sqrt(2)


def pattern_branch_diagram(pattern: QubitPattern, outcomes: Mapping) -> Diagram:
    """Diagram whose matrix equals the branch Kraus operator of :func:`qubit.run_branch`.

    ``N`` -> ``Z(0)/sqrt2``; ``E`` -> two phase-free Z spiders joined by a Hadamard edge times
    ``sqrt 2``; ``M`` -> effect of the effective (domain-adjusted) basis state for the given
    outcome; active ``X``/``Z`` -> phase-pi X/Z spiders.
    """
    pattern.validate()
    d = Diagram()
    wire: dict = {}
    for q in pattern.inputs:
        wire[q] = d.add_input()
    signals: dict = {}

    def extend(q, kind: VertexType, phase: float = 0.0, edge: EdgeType = EdgeType.SIMPLE) -> int:
        v = d.add_vertex(kind, phase)
        d.add_edge(wire[q], v, edge)
        wire[q] = v
        return v

    for c in pattern.commands:
        if isinstance(c, N):
            wire[c.node] = d.add_vertex(VertexType.Z)
            d.scalar /= math.sqrt(2)
        elif isinstance(c, E):
            a, b = extend(c.u, VertexType.Z), extend(c.v, VertexType.Z)
            d.add_edge(a, b, EdgeType.HADAMARD)
            d.scalar *= math.sqrt(2)
        elif isinstance(c, M):
            r = int(outcomes[c.node]) & 1
            angle = effective_angle(
                c.plane, c.angle, _parity(c.s_domain, signals), _parity(c.t_domain, signals)
            )
            # outcome r is the outcome-0 state of angle + r*pi (see measurement_basis)
            _effect(d, wire.pop(c.node), c.plane, angle + math.pi * r)
            signals[c.node] = r
        elif isinstance(c, (X, Z)) and _parity(c.domain, signals):
            extend(c.node, VertexType.X if isinstance(c, X) else VertexType.Z, math.pi)
        elif isinstance(c, S):
            signals[c.node] ^= _parity(c.domain, signals)
    for q in pattern.outputs:
        d.add_edge(wire[q], d.add_output())
    return d
