"""Gaussian-graph calculus (MFvL 2011) checked against direct hbar=2 Gaussian numerics.

Tolerances: every check is a few dense O(N^3) linear-algebra steps on states built from
squeezing |r| <= 0.8 per gate, so condition numbers stay below ~1e4 and the expected error is
~eps * cond ~ 1e-12. RTOL = 1e-10 (max-norm, relative to max(1, |reference|)) leaves a 100x
margin; the observed maxima are ~1e-14 (see docs/user-guide/gaussian-graphical-calculus.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from photographiq import gaussian as gm
from photographiq.backends.gaussian import GaussianBackend
from photographiq.calculus import GaussianGraph
from photographiq.measurements import Homodyne
from photographiq.states import GaussianInput, GaussianState

RTOL = 1e-10
SEEDS = range(20)


def err(actual, reference) -> float:
    reference = np.asarray(reference)
    return float(np.abs(np.asarray(actual) - reference).max() / max(1.0, np.abs(reference).max()))


def random_symplectic(rng, n, depth=None):
    """Product of random squeezers, rotations, beam splitters and CZ gates on n modes."""
    s = np.eye(2 * n)
    for _ in range(depth or 3 * n):
        kind = rng.integers(4) if n > 1 else rng.integers(2)
        if kind == 0:
            j, g = [rng.integers(n)], gm.squeezing(rng.uniform(-0.8, 0.8))
        elif kind == 1:
            j, g = [rng.integers(n)], gm.rotation(rng.uniform(0, 2 * np.pi))
        else:
            j = list(rng.choice(n, 2, replace=False))
            g = gm.beamsplitter(rng.uniform(0, np.pi)) if kind == 2 else gm.cz(rng.normal())
        s = gm.embed(g, [q for m in j for q in (2 * m, 2 * m + 1)], 2 * n) @ s
    return s


def random_state(seed, n=None) -> GaussianState:
    """Random pure state S S^T (S applied to vacuum) with a random mean."""
    rng = np.random.default_rng(seed)
    n = n or int(rng.integers(1, 6))
    s = random_symplectic(rng, n)
    return GaussianState(
        rng.normal(scale=2.0, size=2 * n), s @ s.T, tuple(f"m{i}" for i in range(n))
    )


def backend_with(state: GaussianState) -> GaussianBackend:
    backend = GaussianBackend()
    backend.set_state(state)
    return backend


def assert_same_state(graph: GaussianGraph, state: GaussianState, rtol=RTOL):
    assert graph.nodes == state.nodes
    assert err(graph.to_covariance(), state.covariance) < rtol
    assert err(graph.mean, state.mean) < rtol


# ---------------------------------------------------------------- Z <-> covariance


def test_vacuum_and_resource_conventions():
    assert np.allclose(GaussianGraph.vacuum("ab").to_covariance(), np.eye(4), rtol=0, atol=0)
    r = 0.6
    backend = GaussianBackend()
    backend.prepare("x", r)  # p-squeezed resource diag(e^{2r}, e^{-2r})
    graph = GaussianGraph.squeezed(("x",), r)
    assert graph.Z[0, 0] == 1j * np.exp(-2 * r)
    assert_same_state(graph, backend.get_state())
    coherent = GaussianInput.coherent(0.3 - 0.2j).state("c")
    assert_same_state(GaussianGraph.from_state(coherent), coherent)
    assert np.allclose(GaussianGraph.from_state(coherent).Z, 1j)


@pytest.mark.parametrize("seed", SEEDS)
def test_round_trips_on_random_pure_states(seed):
    state = random_state(seed)
    graph = GaussianGraph.from_state(state)
    assert_same_state(graph, state)
    again = GaussianGraph(graph.Z, graph.nodes).to_covariance()
    assert err(GaussianGraph.from_covariance(again).Z, graph.Z) < RTOL


def test_correspondence_formula_explicitly():
    """cov (qq/pp) = [[U^-1, U^-1 V], [V U^-1, U + V U^-1 V]] = 2 x MFvL Eq. 2.19."""
    rng = np.random.default_rng(7)
    m = rng.normal(size=(3, 3))
    u, v = m @ m.T + np.eye(3), rng.normal(size=(3, 3))
    v = v + v.T
    cov = GaussianGraph(v + 1j * u).to_covariance()
    ui = np.linalg.inv(u)
    assert err(cov[0::2, 0::2], ui) < RTOL
    assert err(cov[1::2, 0::2], v @ ui) < RTOL
    assert err(cov[1::2, 1::2], u + v @ ui @ v) < RTOL
    # pure: det = 1 and symplectic eigenvalues 1, i.e. (Omega V)^2 = -I
    om = gm.omega(3)
    assert err(om @ cov @ om @ cov, -np.eye(6)) < RTOL


def test_cluster_matches_backend_preparation():
    a = np.array([[0, 1.0, 0, 0.5], [1.0, 0, -2.0, 0], [0, -2.0, 0, 1.0], [0.5, 0, 1.0, 0]])
    r = 0.9
    backend = GaussianBackend()
    for node in range(4):
        backend.prepare(node, r)
    for u in range(4):
        for v in range(u + 1, 4):
            if a[u, v]:
                backend.entangle(u, v, a[u, v])
    graph = GaussianGraph.cluster(a, r, range(4))
    assert np.array_equal(graph.Z, a + 1j * np.exp(-2 * r) * np.eye(4))
    assert_same_state(graph, backend.get_state())


# ---------------------------------------------------------------- Gaussian unitaries


GATES = {
    "cz": lambda g, b, u, v, x: (g.cz(u, v, x), b.entangle(u, v, x)),
    "rotate": lambda g, b, u, v, x: (g.rotate(u, x), b.rotate(u, x)),
    "squeeze": lambda g, b, u, v, x: (g.squeeze(u, x / 3), b.squeeze(u, x / 3)),
    "beamsplitter": lambda g, b, u, v, x: (g.beamsplitter(u, v, x), b.beamsplitter(u, v, x)),
    "quadratic_phase": lambda g, b, u, v, x: (g.quadratic_phase(u, x), b.quadratic_phase(u, x)),
}


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("seed", SEEDS)
def test_unitary_rules_match_symplectic_action(gate, seed):
    state = random_state(seed, n=3)
    rng = np.random.default_rng(1000 + seed)
    u, v = rng.choice(state.nodes, 2, replace=False)
    backend = backend_with(state)
    graph, _ = GATES[gate](GaussianGraph.from_state(state), backend, u, v, rng.normal())
    assert_same_state(graph, backend.get_state())


@pytest.mark.parametrize("seed", SEEDS)
def test_generic_symplectic_on_subset(seed):
    rng = np.random.default_rng(seed)
    state = random_state(seed, n=4)
    nodes = tuple(rng.choice(state.nodes, 2, replace=False))
    s = random_symplectic(rng, 2)
    backend = backend_with(state)
    backend.transform(nodes, s)
    assert_same_state(GaussianGraph.from_state(state).apply(s, nodes), backend.get_state())


def test_paper_closed_form_local_rules():
    """MFvL Eqs. 3.14, 3.16, 3.18, 3.20, 3.23, 3.12 in the repo's sign conventions."""
    state = random_state(3, n=3)
    g = GaussianGraph.from_state(state)
    z = g.Z
    t, rr, w = z[0, 0], z[1:, 0], z[1:, 1:]

    def parts(h):
        return h.Z[0, 0], h.Z[1:, 0], h.Z[1:, 1:]

    # shear / quadratic phase (3.14): t -> t + s
    assert err(g.quadratic_phase("m0", 0.4).Z, z + np.diag([0.4, 0, 0])) < RTOL
    # CZ (3.23): Z_uv += g
    added = np.zeros((3, 3))
    added[0, 2] = added[2, 0] = -1.3
    assert err(g.cz("m0", "m2", -1.3).Z, z + added) < RTOL
    # squeezing (3.16): repo squeezing(r) = MFvL S_squeeze(-r): t e^{2r}, r e^{r}, W fixed
    sq = 0.35
    for got, want in zip(parts(g.squeeze("m0", sq)), (np.exp(2 * sq) * t, np.exp(sq) * rr, w)):
        assert err(got, want) < RTOL
    # phase (3.18): repo rotation(a) = MFvL S_phase(-a)
    a = 0.7
    c, s = np.cos(-a), np.sin(-a)
    den = c + t * s
    want = ((-s + t * c) / den, rr / den, w - s * np.outer(rr, rr) / den)
    for got, ref in zip(parts(g.rotate("m0", a)), want):
        assert err(got, ref) < RTOL
    # Fourier F = MFvL S_phase(-pi/2) = repo rotation(pi/2) (3.20)
    want = (-1 / t, -rr / t, w - np.outer(rr, rr) / t)
    for got, ref in zip(parts(g.rotate("m0", np.pi / 2)), want):
        assert err(got, ref) < RTOL
    # beam splitter (3.12): Z' = (R ⊕ I) Z (R ⊕ I)^T with R = [[c, -s], [s, c]]
    th = 0.45
    big = np.eye(3)
    big[:2, :2] = gm.rotation(th)
    assert err(g.beamsplitter("m0", "m1", th).Z, big @ z @ big.T) < RTOL


# ---------------------------------------------------------------- measurements


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("quadrature", ["q", "p", "angle"])
def test_homodyne_rules_match_backend_conditioning(seed, quadrature):
    rng = np.random.default_rng(seed)
    state = random_state(seed, n=int(rng.integers(2, 6)))
    node = state.nodes[int(rng.integers(len(state.nodes)))]
    outcome = float(rng.normal(scale=3.0))
    graph = GaussianGraph.from_state(state)
    if quadrature == "q":
        angle, result = 0.0, graph.measure_q(node, outcome)
        # vertex deletion: Z' is Z without row/column k, independent of the outcome
        keep = [i for i, n in enumerate(graph.nodes) if n != node]
        assert np.array_equal(result.Z, graph.Z[np.ix_(keep, keep)])
    elif quadrature == "p":
        angle, result = np.pi / 2, graph.measure_p(node, outcome)
    else:
        angle = float(rng.uniform(0, 2 * np.pi))
        result = graph.rotate(node, -angle).measure_q(node, outcome)
    backend = backend_with(state)
    backend.measure(node, Homodyne(), angle, outcome=outcome)
    assert_same_state(result, backend.get_state())


def test_measuring_last_mode_and_sequences():
    state = random_state(11, n=3)
    graph = GaussianGraph.from_state(state)
    backend = backend_with(state)
    for node, meth, angle, m in (("m1", "measure_p", np.pi / 2, 0.8), ("m0", "measure_q", 0, -1.1)):
        graph = getattr(graph, meth)(node, m)
        backend.measure(node, Homodyne(), angle, outcome=m)
    assert_same_state(graph, backend.get_state())
    empty = graph.measure_q("m2", 0.3)
    assert empty.nodes == () and empty.Z.shape == (0, 0)


def test_measurement_rejects_bad_outcome():
    with pytest.raises(ValueError, match="finite real"):
        GaussianGraph.vacuum("a").measure_q("a", float("nan"))
    with pytest.raises(KeyError):
        GaussianGraph.vacuum("a").measure_p("b", 0.0)


# ---------------------------------------------------------------- ideal limits


def test_wire_shortening_ideal_limit():
    """p-measuring the two inner nodes of a 4-node unit-weight wire joins its ends.

    Exact Schur complement: Z' = -X / (1 + e^2) + i e (2 + e^2) / (1 + e^2) I with e = e^{-2r},
    so Z' -> -X (edge weight -1, MFvL Sec. III.E p rule applied twice) with error O(e^{-2r}).
    The intermediate graph has Im Z_00 ~ e^{2r}; the second Schur step cancels it down to
    ~2 e^{-2r}, so the rounding error is ~eps e^{2r} (1.6e-9 at r = 8), not eps: the tolerance
    is 64 eps e^{2r}. The tiny Im part (finite-squeezing noise) is thus only ~1% accurate at r = 8.
    """
    a = np.diag([1.0, 1.0, 1.0], 1)
    a = a + a.T
    x = np.array([[0, 1.0], [1.0, 0]])
    errors = []
    for r in (0.5, 1.0, 2.0, 4.0, 8.0):
        e = np.exp(-2 * r)
        g = GaussianGraph.cluster(a, r).measure_p(1, 0.0).measure_p(2, 0.0)
        exact = -x / (1 + e**2) + 1j * e * (2 + e**2) / (1 + e**2) * np.eye(2)
        assert np.abs(g.Z - exact).max() < 64 * np.finfo(float).eps * np.exp(2 * r)
        errors.append(np.abs(g.Z - (-x)).max())
        assert errors[-1] <= 2.5 * e
    assert all(b < a for a, b in zip(errors, errors[1:]))
    assert errors[-1] < 1e-6


def test_cluster_converges_to_adjacency():
    a = np.array([[0, 1.0], [1.0, 0]])
    for r in (1.0, 4.0, 8.0):
        assert np.abs(GaussianGraph.cluster(a, r).Z - a).max() == pytest.approx(np.exp(-2 * r))


# ---------------------------------------------------------------- composition


def test_tensor_relabel_reorder():
    s1, s2 = random_state(1, n=2), random_state(2, n=1)
    s2 = GaussianState(s2.mean, s2.covariance, ("z",))
    g = GaussianGraph.from_state(s1).tensor(GaussianGraph.from_state(s2))
    backend = backend_with(s1)
    backend.prepare("z", state=GaussianInput(tuple(s2.mean), tuple(map(tuple, s2.covariance))))
    assert_same_state(g, backend.get_state())
    renamed = g.relabel({"m0": "a", "z": "c"})
    assert renamed.nodes == ("a", "m1", "c") and renamed.Z is not None
    assert np.array_equal(renamed.Z, g.Z)
    perm = g.reorder(("z", "m0", "m1"))
    assert_same_state(perm, backend.get_state().reduced(("z", "m0", "m1")))
    with pytest.raises(ValueError, match="disjoint"):
        g.tensor(g)
    with pytest.raises(ValueError, match="permutation"):
        g.reorder(("z", "m0"))


def test_edges_view():
    g = GaussianGraph.cluster([[0, 2.0], [2.0, 0]], 0.0, "ab")
    assert g.edges() == {("a", "a"): 1j, ("a", "b"): 2.0, ("b", "b"): 1j}


# ---------------------------------------------------------------- negative tests


def test_mixed_states_rejected():
    with pytest.raises(ValueError, match="not a pure Gaussian state"):
        GaussianGraph.from_covariance(2 * np.eye(2))  # thermal
    backend = GaussianBackend()
    backend.prepare("a", 0.5)
    backend.prepare("b", 0.5)
    backend.entangle("a", "b")
    backend.loss("a", 0.9)
    with pytest.raises(ValueError, match="not a pure Gaussian state"):
        GaussianGraph.from_state(backend.get_state())
    # reduced state of an entangled pure state is mixed
    with pytest.raises(ValueError, match="mixed states have no Gaussian graph"):
        GaussianGraph.from_state(random_state(4, n=3).reduced(("m0",)))


def test_invalid_Z_rejected():
    with pytest.raises(ValueError, match="positive definite"):
        GaussianGraph([[1.0]])  # ideal (unnormalizable) state, Im Z = 0
    with pytest.raises(ValueError, match="positive definite"):
        GaussianGraph(np.diag([1j, -1j]))
    with pytest.raises(ValueError, match="symmetric"):
        GaussianGraph([[1j, 1 + 1j], [1 - 1j, 1j]])  # Hermitian but not symmetric
    with pytest.raises(ValueError, match="square"):
        GaussianGraph(np.ones((2, 3)) * 1j)
    with pytest.raises(ValueError, match="unique"):
        GaussianGraph(np.eye(2) * 1j, ("a", "a"))
    with pytest.raises(ValueError, match="positive definite"):
        GaussianGraph.from_covariance(-np.eye(2))
    with pytest.raises(ValueError, match="zero diagonal"):
        GaussianGraph.cluster([[1.0]], 1.0)


def test_non_symplectic_rejected():
    g = GaussianGraph.vacuum("ab")
    with pytest.raises(ValueError, match="symplectic"):
        g.apply(np.diag([2.0, 2.0]), ("a",))
    with pytest.raises(ValueError, match="symplectic"):
        g.apply(np.eye(2) * 1j, ("a",))
    with pytest.raises(ValueError, match="Repeated"):
        g.apply(np.eye(4), ("a", "a"))
    with pytest.raises(ValueError, match="distinct"):
        g.cz("a", "a")
