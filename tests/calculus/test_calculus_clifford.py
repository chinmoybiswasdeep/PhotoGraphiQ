"""Single-qubit Clifford group, local complementation and LC equivalence."""

from __future__ import annotations

import itertools
import math

import networkx as nx
import numpy as np
import pytest

from photographiq.calculus.clifford import (
    PAULIS,
    Clifford,
    apply_local_cliffords,
    cv_local_complementation,
    graph_state,
    lc_cliffords,
    lc_equivalence,
    local_complement,
)

ATOL = 1e-12


def proportional(a: np.ndarray, b: np.ndarray) -> bool:
    overlap = np.vdot(b, a)
    return abs(abs(overlap) - np.linalg.norm(a) * np.linalg.norm(b)) < 1e-10 and abs(overlap) > 0


def test_group_has_24_distinct_unitaries_and_closed_table():
    group = Clifford.all()
    mats = [c.matrix for c in group]
    for a, b in itertools.combinations(mats, 2):
        assert not proportional(a.reshape(-1), b.reshape(-1))
    for a, b in itertools.product(group, repeat=2):
        assert proportional((a @ b).matrix.reshape(-1), (a.matrix @ b.matrix).reshape(-1))
    for a, b, c in itertools.islice(itertools.product(group, repeat=3), 0, None, 37):
        assert (a @ b) @ c == a @ (b @ c)
    for c in group:
        assert c @ c.inverse() == Clifford(0) == c.inverse() @ c
        assert Clifford.from_matrix(np.exp(0.37j) * c.matrix) == c
        assert Clifford.named(c.name) == c


def test_pauli_conjugation_is_a_signed_permutation():
    for c in Clifford.all():
        images = {p: c.conjugate(p) for p in "XYZ"}
        assert sorted(name for _, name in images.values()) == ["X", "Y", "Z"]
        u = c.matrix
        for p, (sign, name) in images.items():
            assert np.allclose(u @ PAULIS[p] @ u.conj().T, sign * PAULIS[name], atol=ATOL)
    h = Clifford.named("H")
    assert h.conjugate("X") == (1, "Z") and h.conjugate("Y") == (-1, "Y")


def test_non_clifford_is_rejected():
    with pytest.raises(ValueError, match="not a Clifford"):
        Clifford.from_matrix(np.diag([1, np.exp(1j * math.pi / 4)]))


def random_graph(rng: np.random.Generator, n: int) -> nx.Graph:
    g = nx.gnp_random_graph(n, 0.5, seed=int(rng.integers(1 << 30)))
    return g


def stabilizer(graph: nx.Graph, v, order) -> np.ndarray:
    ops = [PAULIS["X"] if w == v else PAULIS["Z"] if w in graph[v] else PAULIS["I"] for w in order]
    out = np.ones((1, 1))
    for op in ops:
        out = np.kron(out, op)
    return out


def test_graph_state_is_stabilized_by_generators():
    rng = np.random.default_rng(0)
    for _ in range(10):
        g = random_graph(rng, 5)
        psi = graph_state(g)
        for v in g:
            assert np.allclose(stabilizer(g, v, tuple(g)) @ psi, psi, atol=ATOL)


def test_local_complementation_unitary_maps_graph_states():
    rng = np.random.default_rng(1)
    for _ in range(20):
        g = random_graph(rng, int(rng.integers(2, 7)))
        order = tuple(g)
        for v in g:
            tau = local_complement(g, v)
            out = apply_local_cliffords(graph_state(g, order), lc_cliffords(g, v), order)
            assert proportional(out, graph_state(tau, order))
            for w in tau:
                assert np.allclose(stabilizer(tau, w, order) @ out, out, atol=ATOL)
            assert nx.utils.edges_equal(local_complement(tau, v).edges, g.edges)


def test_lc_equivalence_recovers_random_sequences():
    rng = np.random.default_rng(2)
    for _ in range(15):
        g1 = random_graph(rng, int(rng.integers(3, 7)))
        g2 = g1
        for v in rng.choice(list(g1), size=4):
            g2 = local_complement(g2, int(v))
        witness = lc_equivalence(g1, g2)
        assert witness is not None and witness.residual < 1e-12
        order = tuple(g1)
        out = apply_local_cliffords(graph_state(g1, order), witness.cliffords, order)
        for w in g2:
            assert np.allclose(stabilizer(g2, w, order) @ out, out, atol=ATOL)


def cut_rank(graph: nx.Graph, part) -> int:
    """GF(2) rank of the cut adjacency submatrix (an LC invariant: log2 Schmidt rank)."""
    rows = [[int(graph.has_edge(a, b)) for b in graph if b not in part] for a in part]
    m = np.array(rows, dtype=np.uint8) % 2
    rank = 0
    for col in range(m.shape[1]):
        pivot = next((r for r in range(rank, m.shape[0]) if m[r, col]), None)
        if pivot is None:
            continue
        m[[rank, pivot]] = m[[pivot, rank]]
        for r in range(m.shape[0]):
            if r != rank and m[r, col]:
                m[r] ^= m[rank]
        rank += 1
    return rank


def test_lc_inequivalence_agrees_with_cut_rank_invariant():
    star, path, complete = nx.star_graph(3), nx.path_graph(4), nx.complete_graph(4)
    assert cut_rank(star, {0, 2}) != cut_rank(path, {0, 2})
    assert lc_equivalence(star, path) is None
    witness = lc_equivalence(star, complete)
    assert witness is not None and witness.sequence == (0,)
    with pytest.raises(ValueError, match="same labelled"):
        lc_equivalence(star, nx.path_graph(5))


def test_cv_local_complementation_is_explicitly_unsupported():
    with pytest.raises(NotImplementedError, match="does not transfer"):
        cv_local_complementation()
