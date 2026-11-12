"""ZX diagrams: exact tensor semantics and scalar-exact soundness of every rewrite."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest
from calculus_random import random_pattern

from photographiq.calculus import zx
from photographiq.calculus.clifford import graph_state
from photographiq.calculus.qubit import E, M, N, QubitPattern, X, enumerate_branches
from photographiq.calculus.zx import Diagram, EdgeType, RewriteError, VertexType
from photographiq.ir import Plane

ATOL = 1e-11  # dense contractions of <= ~25 tensors; observed deviations <= 1e-14
H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)


def spider_matrix(kind, phase, n_in, n_out):
    d = Diagram()
    s = d.add_vertex(kind, phase)
    for _ in range(n_in):
        d.add_edge(d.add_input(), s)
    for _ in range(n_out):
        d.add_edge(s, d.add_output())
    return d.matrix()


def test_spider_semantics():
    a = 0.73
    assert np.allclose(spider_matrix("Z", a, 0, 1), [[1], [np.exp(1j * a)]])
    assert np.allclose(spider_matrix("Z", a, 1, 1), np.diag([1, np.exp(1j * a)]))
    assert np.allclose(spider_matrix("X", math.pi, 1, 1), [[0, 1], [1, 0]])
    plus = np.array([1, 1]) / math.sqrt(2)
    minus = np.array([1, -1]) / math.sqrt(2)
    expected = np.kron(plus, plus)[:, None] * plus[None, :] + np.exp(1j * a) * (
        np.kron(minus, minus)[:, None] * minus[None, :]
    )
    assert np.allclose(spider_matrix("X", a, 1, 2), expected)
    d = Diagram()
    d.add_edge(d.add_input(), d.add_output(), EdgeType.HADAMARD)
    assert np.allclose(d.matrix(), H)
    assert np.isclose(spider_matrix("Z", a, 0, 0)[0, 0], 1 + np.exp(1j * a))


def test_composition_and_tensor_product():
    a = Diagram()
    s = a.add_vertex("Z", 0.4)
    a.add_edge(a.add_input(), s)
    a.add_edge(s, a.add_output(), EdgeType.HADAMARD)
    b = Diagram()
    t = b.add_vertex("X", 1.1)
    b.add_edge(b.add_input(), t, EdgeType.HADAMARD)
    b.add_edge(t, b.add_output())
    b.scalar = 0.5j
    assert np.allclose(a.then(b).matrix(), b.matrix() @ a.matrix(), atol=ATOL)
    assert np.allclose(a.tensor(b).matrix(), np.kron(a.matrix(), b.matrix()), atol=ATOL)


def random_diagram(rng: np.random.Generator, n_spiders: int = 6, n_edges: int = 7) -> Diagram:
    d = Diagram()
    spiders = [
        d.add_vertex(VertexType.Z if rng.random() < 0.5 else VertexType.X, rng.uniform(0, 6.3))
        for _ in range(n_spiders)
    ]

    def kind():
        return EdgeType.HADAMARD if rng.random() < 0.4 else EdgeType.SIMPLE

    for _ in range(int(rng.integers(1, 3))):
        d.add_edge(d.add_input(), spiders[rng.integers(n_spiders)], kind())
    for _ in range(int(rng.integers(1, 3))):
        d.add_edge(spiders[rng.integers(n_spiders)], d.add_output(), kind())
    for _ in range(n_edges):
        u, v = rng.choice(n_spiders, 2, replace=False)
        d.add_edge(spiders[u], spiders[v], kind())
    d.scalar = complex(rng.normal(), rng.normal())
    return d


def pick(rng, d, kind=None, exclude=()):
    choices = [
        v
        for v in d.spiders()
        if (kind is None or d.kind(v) is VertexType(kind)) and v not in exclude
    ]
    return int(choices[rng.integers(len(choices))])


def other_colour(kind):
    return VertexType.X if kind is VertexType.Z else VertexType.Z


def assert_sound(before: Diagram, after: Diagram, rule: str):
    assert after.trace[-1].rule == rule
    a, b = before.matrix(), after.matrix()
    assert np.max(np.abs(a - b)) <= ATOL * max(1.0, np.max(np.abs(a))), rule


def setups(rng, d):
    """Yield (rule name, diagram with the redex planted, rewrite thunk)."""
    # fusion: plant a simple edge between two same-colour spiders
    u = pick(rng, d)
    v = d.add_vertex(d.kind(u), rng.uniform(0, 6))
    d.add_edge(u, v)
    d.add_edge(v, pick(rng, d), EdgeType.HADAMARD)
    yield "fuse", d, lambda x: zx.fuse(x, u, v)
    # self-loops (simple + Hadamard)
    e = d.copy()
    w = pick(rng, e)
    e.add_edge(w, w, EdgeType.HADAMARD)
    e.add_edge(w, w)
    yield "self-loop", e, lambda x: zx.remove_self_loops(x, w)
    # identity: subdivide an edge with a phase-0 spider
    f = d.copy()
    eid = next(iter(k for k, (a, b, _) in f.edges.items() if a != b))
    a, b, t = f.edges.pop(eid)
    mid = f.add_vertex(VertexType.Z if rng.random() < 0.5 else VertexType.X)
    f.add_edge(a, mid, t)
    f.add_edge(mid, b, EdgeType.HADAMARD)
    yield "identity", f, lambda x: zx.remove_identity(x, mid)
    # colour change
    g = d.copy()
    c = pick(rng, g)
    yield "color-change", g, lambda x: zx.color_change(x, c)
    # state copy
    h = d.copy()
    target = pick(rng, h)
    state = h.add_vertex(other_colour(h.kind(target)), math.pi * int(rng.integers(2)))
    h.add_edge(state, target)
    for loop in h.loops(target):
        del h.edges[loop]
    yield "copy", h, lambda x: zx.copy_state(x, state, target)
    # bialgebra
    k = d.copy()
    z, xs = k.add_vertex(VertexType.Z), k.add_vertex(VertexType.X)
    k.add_edge(z, xs)
    for spider in (z, z, xs, xs):
        k.add_edge(
            spider,
            pick(rng, k, exclude=(z, xs)),
            EdgeType.HADAMARD if rng.random() < 0.3 else EdgeType.SIMPLE,
        )
    yield "bialgebra", k, lambda x: zx.bialgebra(x, z, xs)
    # Hopf
    m = d.copy()
    zz = (
        pick(rng, m, "Z")
        if any(m.kind(s) is VertexType.Z for s in m.spiders())
        else m.add_vertex("Z")
    )
    xx = m.add_vertex(VertexType.X, rng.uniform(0, 6))
    m.add_edge(zz, xx)
    m.add_edge(xx, zz)
    m.add_edge(xx, pick(rng, m))
    yield "hopf", m, lambda x: zx.hopf(x, zz, xx)
    # parallel Hadamard pair between same colours
    n = d.copy()
    p = pick(rng, n)
    r = n.add_vertex(n.kind(p), rng.uniform(0, 6))
    n.add_edge(p, r, EdgeType.HADAMARD)
    n.add_edge(r, p, EdgeType.HADAMARD)
    n.add_edge(r, pick(rng, n))
    yield "hadamard-pair", n, lambda x: zx.cancel_hadamard_pair(x, p, r)


def test_every_rewrite_is_scalar_exact_on_random_diagrams():
    rng = np.random.default_rng(17)
    counts: dict = {}
    for _ in range(25):
        base = random_diagram(rng)
        for rule, diagram, rewrite in setups(rng, base.copy()):
            after = rewrite(diagram)
            assert_sound(diagram, after, rule)
            counts[rule] = counts.get(rule, 0) + 1
    assert len(counts) == 8 and min(counts.values()) == 25


def test_rewrite_preconditions_are_enforced():
    d = Diagram()
    z, x = d.add_vertex("Z", 0.3), d.add_vertex("X")
    d.add_edge(z, x)
    d.add_edge(d.add_input(), z)
    d.add_edge(x, d.add_output())
    with pytest.raises(RewriteError):
        zx.fuse(d, z, x)
    with pytest.raises(RewriteError):
        zx.remove_identity(d, z)
    with pytest.raises(RewriteError):
        zx.hopf(d, z, x)
    with pytest.raises(RewriteError):
        zx.bialgebra(d, z, x)
    with pytest.raises(RewriteError):
        zx.copy_state(d, z, x)
    with pytest.raises(RewriteError):
        zx.remove_self_loops(d, z)


def test_simplify_spiders_preserves_tensor_and_records_trace():
    rng = np.random.default_rng(3)
    for _ in range(10):
        d = random_diagram(rng)
        s = zx.simplify_spiders(d)
        assert np.allclose(s.matrix(), d.matrix(), atol=ATOL * max(1, np.abs(d.matrix()).max()))
        assert all(isinstance(step, zx.RewriteStep) for step in s.trace)
        assert len(s.spiders()) <= len(d.spiders())


def test_graph_state_diagram_equals_statevector():
    rng = np.random.default_rng(5)
    for _ in range(10):
        g = nx.gnp_random_graph(int(rng.integers(1, 6)), 0.5, seed=int(rng.integers(1 << 30)))
        d = zx.graph_state_diagram(g)
        assert np.allclose(d.matrix()[:, 0], graph_state(g), atol=ATOL)


def test_pattern_branch_diagrams_equal_kraus_operators():
    rng = np.random.default_rng(23)
    for _ in range(12):
        pattern = random_pattern(rng, n_aux=2, steps=12)
        for branch in enumerate_branches(pattern)[:4]:
            d = zx.pattern_branch_diagram(pattern, branch.outcomes)
            assert np.allclose(d.matrix(), branch.operator, atol=ATOL)
            simplified = zx.simplify_spiders(d)
            assert np.allclose(simplified.matrix(), branch.operator, atol=ATOL)


def test_j_pattern_simplifies_to_two_spiders():
    alpha = 0.6
    pattern = QubitPattern((1,), [N(2), E(1, 2), M(1, Plane.XY, -alpha), X(2, {1})])
    for r in (0, 1):
        d = zx.simplify_spiders(zx.pattern_branch_diagram(pattern, {1: r}))
        expected = H @ np.diag([1, np.exp(1j * alpha)]) / math.sqrt(2)  # branch amplitude 1/sqrt2
        k = d.matrix()
        phase = np.vdot(expected, k) / abs(np.vdot(expected, k))
        assert np.allclose(k, phase * expected, atol=ATOL)
        assert len(d.spiders()) <= 3
