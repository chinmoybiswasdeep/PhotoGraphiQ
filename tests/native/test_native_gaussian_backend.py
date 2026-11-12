"""NativeGaussianBackend reproduces GaussianBackend trajectories, densities and errors.

Tolerances: both backends evaluate the same formulas; only floating-point summation order
differs (local O(nk) updates versus dense S V S^T). Over the random patterns below the
largest observed relative deviations are 4e-15 (outcomes), 5e-15 (densities) and 2e-15
(moments), so rtol=1e-12 leaves a margin of ~200 while any formula error is O(1).
"""

import numpy as np
import pytest

import photographiq as pg
from photographiq.backends.gaussian import GaussianBackend
from photographiq.gaussian import omega

ng = pytest.importorskip(
    "photographiq.backends.native_gaussian",
    reason="native 'gaussian' component not built",
    exc_type=ImportError,
)
NativeGaussianBackend = ng.NativeGaussianBackend

RTOL, ATOL = 1e-12, 1e-12


def random_pattern(rng, modes=6, steps=60):
    """Random adaptive Gaussian pattern: gates, loss, displacements and all detector types."""
    pattern = pg.Pattern(inputs=(0, 1))
    active, scalar_keys, vector_keys = [0, 1], [], []
    fresh = 2

    def prepare():
        nonlocal fresh
        pattern.append(pg.Prepare(fresh, float(rng.uniform(0.1, 1.5))))
        active.append(fresh)
        fresh += 1

    def adaptive(base):
        if scalar_keys and rng.random() < 0.5:
            return pg.Outcome(scalar_keys[rng.integers(len(scalar_keys))]) * 0.3 + base
        if vector_keys and rng.random() < 0.5:
            return pg.Outcome(vector_keys[rng.integers(len(vector_keys))], 1) * 0.2 + base
        return base

    while len(active) < modes:
        prepare()
    for _ in range(steps):
        while len(active) < 3:
            prepare()
        u, v = (int(x) for x in rng.choice(active, 2, replace=False))
        kind = rng.integers(10)
        if kind == 0:
            pattern.append(pg.Entangle(u, v, float(rng.normal())))
        elif kind == 1:
            pattern.append(pg.Rotate(u, adaptive(float(rng.uniform(-3, 3)))))
        elif kind == 2:
            pattern.append(pg.Squeeze(u, float(rng.uniform(-0.8, 0.8))))
        elif kind == 3:
            pattern.append(pg.BeamSplitter(u, v, adaptive(float(rng.uniform(-3, 3)))))
        elif kind == 4:
            pattern.append(pg.Displace(u, adaptive(float(rng.normal())), float(rng.normal())))
        elif kind == 5:
            pattern.append(pg.QuadraticPhase(u, float(rng.normal())))
        elif kind == 6:
            pattern.append(pg.Loss(u, float(rng.uniform(0.3, 1)), float(rng.uniform(0, 0.5))))
        else:
            detector = rng.integers(3)
            if detector == 0:
                angle = adaptive(float(rng.uniform(-3, 3)))
                eta, noise = (1.0, 0.0) if rng.random() < 0.5 else (0.7, 0.05)
                pattern.append(pg.Measure(u, pg.Homodyne(angle, eta, noise)))
                scalar_keys.append(u)
            elif detector == 1:
                pattern.append(pg.Measure(u, pg.Heterodyne()))
                vector_keys.append(u)
            else:
                pattern.append(pg.Measure(u, pg.Generaldyne([[0.5, 0.1], [0.1, 2.5]])))
                vector_keys.append(u)
            active.remove(u)
    pattern.append(pg.Output(tuple(active)))
    return pattern


def assert_same(a, b):
    assert a.outcomes.keys() == b.outcomes.keys()
    for key in a.outcomes:
        np.testing.assert_allclose(b.outcomes[key], a.outcomes[key], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(
            b.measurement_statistics[key]["value"],
            a.measurement_statistics[key]["value"],
            rtol=RTOL,
            atol=0,
        )
        assert (
            b.measurement_statistics[key]["postselected"]
            == a.measurement_statistics[key]["postselected"]
        )
    assert a.state.nodes == b.state.nodes
    np.testing.assert_allclose(b.state.mean, a.state.mean, rtol=RTOL, atol=ATOL)
    scale = max(1.0, np.abs(a.state.covariance).max())
    np.testing.assert_allclose(b.state.covariance, a.state.covariance, rtol=RTOL, atol=ATOL * scale)
    assert b.physical_displacements == a.physical_displacements
    assert b.backend == "NativeGaussianBackend"


def assert_physical(state, tol=1e-9):
    n = len(state.nodes)
    if n:
        lowest = np.linalg.eigvalsh(state.covariance + 1j * omega(n)).min()
        assert lowest > -tol * max(1.0, np.abs(state.covariance).max())


@pytest.mark.parametrize("frame", [False, True])
@pytest.mark.parametrize("case", range(8))
def test_random_patterns_match_numpy_backend(case, frame):
    rng = np.random.default_rng(100 + case)
    pattern = random_pattern(rng)
    inputs = {0: pg.GaussianInput.coherent(0.3 - 0.2j), 1: pg.GaussianInput.squeezed(0.4, 0.7)}
    for seed in (case, 10_000 + case):
        ref = pg.simulate(pattern, backend=GaussianBackend(), inputs=inputs, seed=seed, frame=frame)
        out = pg.simulate(
            pattern, backend=NativeGaussianBackend(), inputs=inputs, seed=seed, frame=frame
        )
        assert_same(ref, out)
        assert_physical(out.state)


@pytest.mark.parametrize("frame", [False, True])
def test_postselection_matches_numpy_backend(frame):
    rng = np.random.default_rng(7)
    pattern = random_pattern(rng, modes=5, steps=40)
    sampled = pg.simulate(pattern, backend=GaussianBackend(), seed=3, frame=frame)
    keys = list(sampled.outcomes)[::2]
    assert keys
    fixed = {k: sampled.outcomes[k] for k in keys}
    ref = pg.simulate(
        pattern, backend=GaussianBackend(), seed=4, frame=frame, measurement_outcomes=fixed
    )
    out = pg.simulate(
        pattern, backend=NativeGaussianBackend(), seed=4, frame=frame, measurement_outcomes=fixed
    )
    assert_same(ref, out)
    for key in keys:
        assert out.measurement_statistics[key]["postselected"]
        np.testing.assert_allclose(out.outcomes[key], fixed[key], rtol=0, atol=1e-12)


def test_correlated_initial_state_and_run_shots():
    rng = np.random.default_rng(11)
    pattern = random_pattern(rng, modes=4, steps=30)
    tmsv = pg.simulate(_tmsv_pattern(0.6), backend="gaussian", seed=0).state
    initial = pg.GaussianState(tmsv.mean + [0.1, 0, 0, -0.2], tmsv.covariance, (0, 1))
    ref = pg.run_shots(pattern, 5, seed=21, backend=GaussianBackend(), initial_state=initial)
    out = pg.run_shots(pattern, 5, seed=21, backend=NativeGaussianBackend(), initial_state=initial)
    for a, b in zip(ref.trajectories, out.trajectories, strict=True):
        assert_same(a, b)


def _tmsv_pattern(r):
    pattern = pg.Pattern()
    pattern.append(pg.Prepare(0, state=pg.GaussianInput.squeezed(r)))
    pattern.append(pg.Prepare(1, state=pg.GaussianInput.squeezed(-r)))
    pattern.append(pg.BeamSplitter(0, 1, np.pi / 4))
    pattern.append(pg.Output((0, 1)))
    return pattern


def test_near_singular_conditioning_large_squeezing():
    """r = 8 resources: anti-squeezed variance e^16 ~ 9e6, squeezed e^-16 ~ 1e-7."""
    r = 8.0
    pattern = pg.Pattern()
    for node in range(4):
        pattern.append(pg.Prepare(node, r))
    for u, v in ((0, 1), (1, 2), (2, 3)):
        pattern.append(pg.Entangle(u, v))
    pattern.append(pg.Measure(0, pg.Homodyne(np.pi / 2)))  # measures the squeezed quadrature
    pattern.append(pg.Measure(1, pg.Homodyne(0.0)))
    pattern.append(pg.Measure(2, pg.Heterodyne()))
    pattern.append(pg.Output((3,)))
    for seed in range(4):
        ref = pg.simulate(pattern, backend=GaussianBackend(), seed=seed)
        out = pg.simulate(pattern, backend=NativeGaussianBackend(), seed=seed)
        # Observed agreement is bitwise here, but O(1) conditional entries arise from
        # cancellations of O(e^16) terms (E - E^2/(E+1)), whose legitimate rounding is
        # ~eps e^16 ~ 2e-9 per operation; rtol=1e-8 is that bound with a small factor.
        for key in ref.outcomes:
            np.testing.assert_allclose(out.outcomes[key], ref.outcomes[key], rtol=1e-8, atol=0)
            np.testing.assert_allclose(
                out.measurement_statistics[key]["value"],
                ref.measurement_statistics[key]["value"],
                rtol=1e-8,
            )
        np.testing.assert_allclose(out.state.covariance, ref.state.covariance, rtol=1e-8)
        np.testing.assert_allclose(out.state.mean, ref.state.mean, rtol=1e-8, atol=1e-8)
        assert_physical(out.state)


def test_tiny_variance_homodyne_closed_form():
    """p_a + q_b of a p-squeezed a and q-squeezed b after CZ has variance 2 e^-2r ~ 3e-8.

    Cov(p_a', q_b) = e^-2r, so q_b | x has mean x/2 and variance e^-2r/2 (exact Schur
    complement); the outcome density is N(x; 0, 2 e^-2r).
    """
    r = 9.0
    backend = NativeGaussianBackend()
    backend.prepare("a", r)
    backend.prepare("b", state=pg.GaussianInput.squeezed(r))
    backend.entangle("a", "b")
    x = 3e-5
    backend.measure("a", pg.Homodyne(np.pi / 2), np.pi / 2, outcome=x)
    small = np.exp(-2 * r)
    expected = np.exp(-(x**2) / (4 * small)) / np.sqrt(4 * np.pi * small)
    assert np.isclose(backend.last_measurement["value"], expected, rtol=1e-12)
    state = backend.get_state()
    assert np.isclose(state.covariance[0, 0], small / 2, rtol=1e-12)
    assert np.isclose(state.mean[0], x / 2, rtol=1e-12)
    assert_physical(state)


def test_errors_match_numpy_backend():
    for backend in (GaussianBackend(), NativeGaussianBackend()):
        backend.prepare("a", 0.5)
        backend.prepare("b", 0.5)
        with pytest.raises(ValueError, match="symplectic"):
            backend.transform(("a",), 2 * np.eye(2))
        with pytest.raises(ValueError, match="Repeated"):
            backend.entangle("a", "a")
        with pytest.raises(ValueError, match="already prepared"):
            backend.prepare("a")
        with pytest.raises(ValueError):
            backend.rotate("zz", 0.1)
        with pytest.raises(ValueError, match="Invalid loss"):
            backend.loss("a", 1.2)
        with pytest.raises(ValueError, match="Nonfinite"):
            backend.displace("a", np.inf, 0)
        with pytest.raises(ValueError, match="zero numerical density"):
            backend.measure("a", pg.Homodyne(), outcome=1e6)
        with pytest.raises(ValueError, match="finite real pair"):
            backend.measure("a", pg.Heterodyne(), outcome=(1.0, 2.0, 3.0))
        with pytest.raises(NotImplementedError):
            backend.measure("a", pg.PhotonNumber())
        assert backend.get_state().nodes == ("a", "b")  # failed operations left the state


def test_snapshots_are_independent_of_backend_storage():
    backend = NativeGaussianBackend()
    backend.prepare(0, 0.3)
    snapshot = backend.get_state()
    backend.squeeze(0, 0.5)
    backend.displace(0, 1.0, 0.0)
    assert snapshot.mean[0] == 0.0
    assert np.isclose(snapshot.covariance[0, 0], np.exp(0.6))
    assert np.isclose(backend.get_state().covariance[0, 0], np.exp(0.6) * np.exp(-1.0))
