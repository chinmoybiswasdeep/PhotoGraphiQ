"""Hybrid Gaussian-Fock CV-MBQC executor: bridge, homodyne instrument, loss, feed-forward,
planner and cross-backend agreement (Gaussian, native-fock, native-mixed-fock).

The real-CUDA-Q test runs only where ``cudaq`` is importable (Linux); elsewhere the explicit
``engine="cudaq"`` request must raise instead of falling back to NumPy.
"""

from __future__ import annotations

import importlib.util
import warnings
from math import comb

import numpy as np
import pytest

import photographiq as pg
import photographiq.cudaq_dynamics as cq
from photographiq import native
from photographiq.hybrid import homodyne as hd
from photographiq.hybrid.executor import _loss_superoperator, ensemble_statistics, run, run_shots
from photographiq.hybrid.planner import classify, plan
from photographiq.hybrid.states import (
    fock_input,
    from_total_photon,
    gaussian_to_fock,
    moment_matched_gaussian,
    to_total_photon,
    williamson,
)

HAS_CUDAQ = importlib.util.find_spec("cudaq") is not None
needs_native = pytest.mark.skipif(
    not native.AVAILABLE or "fock" not in native.components(), reason="native fock missing"
)


def _infidelity(a, b):
    n = max(len(a), len(b))
    A, B = np.zeros(n, complex), np.zeros(n, complex)
    A[: len(a)], B[: len(b)] = a, b
    return 1 - abs(np.vdot(A, B)) ** 2 / np.vdot(A, A).real / np.vdot(B, B).real


def _trace_distance(a, b):
    n = max(len(a), len(b))
    A, B = np.zeros((n, n), complex), np.zeros((n, n), complex)
    A[: len(a), : len(a)], B[: len(b), : len(b)] = a, b
    return 0.5 * np.abs(np.linalg.eigvalsh(A - B)).sum()


def _wire(theta=1.1, r=0.6):
    p = pg.Pattern(inputs=(0,))
    p.extend([pg.Prepare(1, r), pg.Entangle(0, 1)])
    p.measure(0, pg.Homodyne(theta))
    p.displace(1, q=-pg.Outcome(0) / np.sin(theta))
    return p.append(pg.Output((1,))).validate()


def _three_mode_cubic(gamma=0.15, r=0.4, theta=1.1):
    """Wire step plus cubic injection: three live modes, two homodynes, nonlinear feed-forward."""
    p = pg.Pattern(inputs=(0,))
    p.extend(
        [pg.Prepare(1, r), pg.Prepare(2, state=pg.CubicPhaseResource(gamma, r)), pg.Entangle(0, 1)]
    )
    p.measure(0, pg.Homodyne(theta))
    p.displace(1, q=-pg.Outcome(0) / np.sin(theta))
    p.extend([pg.Rotate(2, np.pi / 2), pg.Entangle(1, 2, -1.0), pg.Rotate(2, -np.pi / 2)])
    p.append(pg.Measure(2, pg.Homodyne(0.0), "m"))
    p.extend(
        [
            pg.QuadraticPhase(1, -2 * gamma * pg.Outcome("m")),
            pg.Displace(1, p=-gamma * pg.Outcome("m") ** 2),
        ]
    )
    return p.append(pg.Output((1,))).validate()


# ------------------------------------------------------------------ representation + bridge
def test_williamson_reconstructs_mixed_two_mode_covariance():
    rng = np.random.default_rng(1)
    S = np.diag([np.exp(-0.4), np.exp(0.4)])
    big = np.kron(np.eye(2), S)
    V = big @ np.diag([1.6, 1.6, 1.2, 1.2]) @ big.T
    Q = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    passive = np.kron(Q, np.eye(2))  # orthogonal and symplectic in interleaved order
    V = passive @ V @ passive.T
    nu, Sw = williamson(V)
    assert np.allclose(sorted(nu), [1.2, 1.6])
    assert np.allclose(Sw @ np.diag(np.repeat(nu, 2)) @ Sw.T, V)


def test_bridge_mixed_single_mode_moments_and_weight():
    mean = np.array([0.4, -0.3])
    V = np.array([[1.8, 0.3], [0.3, 1.1]])
    rep = gaussian_to_fock(mean, V, (20,))
    m, c = rep.moments()
    assert rep.representation == "mixed-fock"
    assert np.abs(m - mean).max() < 1e-5 and np.abs(c - V).max() < 1e-4
    assert 0.999 < rep.retained_weight <= 1
    assert np.linalg.eigvalsh(rep.data).min() > -1e-12


def test_bridge_two_mode_pure_entangled_state():
    p = (
        pg.Pattern(inputs=(0, 1))
        .extend([pg.Squeeze(0, 0.3), pg.Entangle(0, 1, 0.7)])
        .append(pg.Output((0, 1)))
    )
    g = pg.simulate(
        p.validate(), backend="gaussian", inputs={0: pg.GaussianInput((0.2, 0.1))}
    ).state
    rep = gaussian_to_fock(g.mean, g.covariance, (16, 16))
    m, c = rep.moments()
    assert rep.representation == "pure-fock"
    assert (
        np.abs(m - g.mean).max() < 1e-5 and np.abs(c - g.covariance).max() < 1e-4
    )  # dim-16 truncation


def test_contract_fields_and_no_silent_renormalisation():
    rep = fock_input([1, 1, 1], "a", 2)
    contract = rep.contract()
    for key in (
        "mode_order",
        "hilbert_space_dimensions",
        "total_photon_cutoff",
        "representation_type",
        "normalization",
        "retained_weight",
        "physical_parameters",
        "numerical_precision",
    ):
        assert key in contract
    assert np.isclose(rep.retained_weight, 2 / 3)  # truncation probability stays visible


def test_total_photon_round_trip_reports_weight():
    rep = gaussian_to_fock([3.0, 0, 0, 0], np.eye(4), (8, 8))
    data, weight = to_total_photon(rep, 4)
    assert 0.5 < weight < 0.99
    back = from_total_photon(data, rep.modes, rep.dims, 4)
    assert np.isclose(back.normalization, weight)


def test_moment_matching_is_labelled_approximate():
    assert moment_matched_gaussian(fock_input([0, 1], 0, 4))["approximate"] is True


# ------------------------------------------------------------------ homodyne instrument
def test_quadrature_wavefunctions_closed_forms_and_density_normalisation():
    x = np.linspace(-12, 12, 4001)
    vac = gaussian_to_fock([0, 0], np.eye(2), (24,), modes=("a",))
    coh = gaussian_to_fock([0.8, -0.4], np.eye(2), (24,), modes=("a",))
    one = fock_input([0, 1], "a", 24)
    g = np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)
    assert np.abs(hd.density(vac, "a", 0.7, x) - g).max() < 1e-12
    for theta in (0.0, 0.7, 2.0):
        mu = 0.8 * np.cos(theta) - 0.4 * np.sin(theta)
        want = np.exp(-((x - mu) ** 2) / 2) / np.sqrt(2 * np.pi)
        assert np.abs(hd.density(coh, "a", theta, x) - want).max() < 1e-7
    assert np.abs(hd.density(one, "a", 0.3, x) - x**2 * g).max() < 1e-12
    assert abs(np.trapezoid(hd.density(coh, "a", 1.0, x), x) - 1) < 1e-9


def test_number_state_is_phase_insensitive_and_squeezed_variance():
    x = np.linspace(-8, 8, 801)
    n2 = fock_input([0, 0, 1], 0, 12)
    assert np.allclose(hd.density(n2, 0, 0.0, x), hd.density(n2, 0, 1.3, x))
    sq = gaussian_to_fock([0, 0], np.diag([np.exp(-0.8), np.exp(0.8)]), (30,))
    p = hd.density(sq, 0, 0.0, x)
    assert abs(np.trapezoid(x**2 * p, x) - np.exp(-0.8)) < 1e-6


@pytest.mark.parametrize("mixed", [False, True])
def test_conditional_state_matches_gaussian_backend(mixed):
    p = _wire()
    inputs = {0: pg.GaussianInput((0.4, -0.2))}
    g = pg.simulate(p, backend="gaussian", inputs=inputs, measurement_outcomes={0: 0.3})
    h = run(p, dims=24, inputs=inputs, measurement_outcomes={0: 0.3}, mixed=mixed)
    m, c = h.state.moments()
    assert h.state.representation == ("mixed-fock" if mixed else "pure-fock")
    assert np.abs(m - g.state.mean).max() < 1e-5 and np.abs(c - g.state.covariance).max() < 1e-3
    assert abs(h.densities[0] - np.exp(g.log_likelihood)) < 1e-6


def test_binned_homodyne_probability_and_fine_limit():
    rep = gaussian_to_fock([0.3, 0.0, 0.0, 0.0], np.eye(4), (10, 10))
    x = np.linspace(-0.25, 0.25, 2001)
    out = hd.measure(rep, 0, 0.0, outcome=0.0, resolution=0.5)
    assert abs(out.density - np.trapezoid(hd.density(rep, 0, 0.0, x), x)) < 1e-6
    assert "coarse graining" in out.convention and out.resolution == 0.5
    fine = hd.measure(rep, 0, 0.0, outcome=0.0, resolution=1e-4).state
    ideal = hd.measure(rep, 0, 0.0, outcome=0.0).state
    assert _trace_distance(fine.data, ideal.density_matrix()) < 1e-6


def test_sampling_is_reproducible_and_matches_density():
    p = _wire()
    a = run_shots(p, 3, seed=7, dims=12)
    b = run_shots(p, 3, seed=7, dims=12)
    assert [r.records for r in a] == [r.records for r in b]
    rep = gaussian_to_fock([0.0, 0.0], np.diag([0.5, 2.0]), (16,))
    rng = np.random.default_rng(0)
    xs = np.array([hd.measure(rep, 0, 0.0, rng=rng, grid=2001).outcome for _ in range(2000)])
    assert abs(xs.mean()) < 4 * np.sqrt(0.5 / 2000) and abs(xs.var() - 0.5) < 0.1
    # replaying a sampled outcome as a fixed outcome reproduces the trajectory
    replay = run(p, dims=12, measurement_outcomes=a[0].records)
    assert np.allclose(replay.state.data, a[0].state.data)


def test_ensemble_statistics_are_labelled_estimates():
    stats = ensemble_statistics(run_shots(_wire(), 4, seed=1, dims=10), cq.number(0))
    assert stats["shots"] == 4 and "estimate" in stats["note"]


# ------------------------------------------------------------------ loss
@pytest.mark.parametrize("eta", [0.3, 0.8])
def test_loss_superoperator_is_the_exact_kraus_channel(eta):
    d = 8
    kraus = []
    for k in range(d):
        K = np.zeros((d, d))
        for n in range(k, d):
            K[n - k, n] = np.sqrt(comb(n, k) * eta ** (n - k) * (1 - eta) ** k)
        kraus.append(K)
    want = sum(np.einsum("ac,bd->abcd", K, K) for K in kraus)
    assert np.abs(_loss_superoperator(eta, 0.0, d) - want).max() < 1e-10


def test_thermal_loss_superoperator_matches_lindblad_reference():
    d, eta, nth = 6, 0.7, 0.2
    rho = np.zeros((d, d), complex)
    rho[1, 1] = rho[2, 2] = 0.5
    rho[1, 2] = rho[2, 1] = 0.3
    jumps = [np.sqrt(1 + nth) * cq.annihilate(0), np.sqrt(nth) * cq.create(0)]
    ref = cq.reference_evolve(
        cq.BosonicHamiltonian(), {0: d}, [0.0, -np.log(eta)], rho, collapse_operators=jumps
    )
    got = np.einsum("abij,ij->ab", _loss_superoperator(eta, nth, d), rho)
    assert np.abs(got - ref.state).max() < 1e-8


@needs_native
def test_lossy_kerr_teleportation_matches_native_mixed_fock():
    p = pg.Pattern(inputs=(0,))
    p.extend(
        [pg.Prepare(1, 0.4), pg.Entangle(0, 1), pg.Kerr(0, 0.2), pg.Loss(0, 0.8), pg.Loss(1, 0.9)]
    )
    p.measure(0, pg.Homodyne(0.5))
    p.displace(1, q=-pg.Outcome(0) / np.sin(0.5))
    p = p.append(pg.Output((1,))).validate()
    inputs, out = {0: pg.GaussianInput((0.3, 0.1))}, {0: 0.2}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = pg.simulate(
            p, backend="native-mixed-fock", cutoff=24, inputs=inputs, measurement_outcomes=out
        )
    h = run(p, dims=16, inputs=inputs, measurement_outcomes=out)
    assert h.state.representation == "mixed-fock"
    assert _trace_distance(h.state.data, ref.state.density_matrix) < 2e-3
    assert abs(h.densities[0] - np.exp(ref.log_likelihood)) < 1e-4


# ------------------------------------------------------------------ non-Gaussian vs native-fock
@needs_native
def test_three_mode_cubic_mbqc_converges_to_native_fock():
    p = _three_mode_cubic()
    inputs, out = {0: pg.GaussianInput((0.3, 0.1))}, {0: 0.2, "m": -0.3}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = pg.simulate(
            p, backend="native-fock", cutoff=32, inputs=inputs, measurement_outcomes=out
        )
    errors = [
        _infidelity(
            run(p, dims=d, inputs=inputs, measurement_outcomes=out).state.data,
            ref.state.state_vector,
        )
        for d in (12, 20)
    ]
    assert errors[1] < errors[0] and errors[1] < 2e-5
    h = run(p, dims=20, inputs=inputs, measurement_outcomes=out)
    assert abs(np.exp(h.log_likelihood) - np.exp(ref.log_likelihood)) < 1e-5
    assert set(h.records) == {0, "m"} and h.plan.peak_dimension == 20**3


@needs_native
def test_photon_counting_matches_native_fock():
    p = pg.non_gaussian.photon_subtraction(0.4)
    inputs = {"in": pg.FockInput((0, 0, 1))}
    for k in (0, 1, 2):
        ref = pg.simulate(
            p, backend="native-fock", cutoff=6, inputs=inputs, measurement_outcomes={"count": k}
        )
        h = run(p, dims=6, inputs=inputs, measurement_outcomes={"count": k})
        assert (
            abs(
                h.probabilities["count"]
                - comb(2, k) * np.sin(0.4) ** (2 * k) * np.cos(0.4) ** (4 - 2 * k)
            )
            < 1e-12
        )
        assert np.isclose(h.probabilities["count"], np.exp(ref.log_likelihood))
        assert _infidelity(h.state.data, ref.state.state_vector) < 1e-12


# ------------------------------------------------------------------ planner, causality, engines
def test_planner_dry_run_segments_and_memory():
    p = _three_mode_cubic()
    pl = plan(p, 10)
    kinds = [s.kind for s in pl.segments]
    assert kinds[:3] == ["gaussian-preparation", "non-gaussian-preparation", "gaussian-unitary"]
    assert kinds.count("homodyne") == 2 and kinds.count("feedforward") == 3
    assert pl.executable and pl.peak_dimension == 1000 and pl.peak_bytes == 16 * 1000
    assert plan(p, 10, mixed=True).peak_bytes == 16 * 1000**2
    assert pl.to_dict()["segments"][3]["writes"] == ("0",)
    assert classify(pg.Measure(0, pg.Homodyne(0.0))) == "homodyne"


def test_unsupported_commands_are_reported_not_executed():
    p = (
        pg.Pattern(inputs=(0,))
        .append(pg.Prepare(1, state=pg.CatResource(1.0)))
        .append(pg.Output((0, 1)))
    )
    assert not plan(p.validate(), 8).executable
    with pytest.raises(NotImplementedError):
        run(p, dims=8)


def test_feedforward_causality_is_enforced():
    p = pg.Pattern(inputs=(0,)).extend(
        [pg.Prepare(1, 0.5), pg.Entangle(0, 1), pg.Displace(1, q=pg.Outcome(0))]
    )
    p.measure(0, pg.Homodyne(0.0))
    with pytest.raises(ValueError, match="future outcome"):
        run(p.append(pg.Output((1,))), dims=8)


def test_feedforward_gain_is_applied_to_the_record():
    p = pg.Pattern(inputs=(0,)).extend([pg.Prepare(1, 0.5), pg.Entangle(0, 1)])
    p.measure(0, pg.Homodyne(np.pi / 2))
    p.displace(1, q=pg.Parameter("g") * pg.Outcome(0))
    p = p.append(pg.Output((1,))).validate()
    base = run(p, dims=16, measurement_outcomes={0: 0.4}, parameters={"g": 0.0}).state.moments()[0]
    gain = run(p, dims=16, measurement_outcomes={0: 0.4}, parameters={"g": 1.5}).state.moments()[0]
    assert abs((gain - base)[0] - 0.6) < 1e-3


def test_engine_selection_is_explicit():
    with pytest.raises(ValueError):
        run(_wire(), dims=8, engine="gpu")


@pytest.mark.skipif(HAS_CUDAQ, reason="cudaq is installed")
def test_cudaq_engine_never_falls_back_to_numpy():
    with pytest.raises(ImportError, match="Linux"):
        run(_wire(), dims=8, engine="cudaq", measurement_outcomes={0: 0.1})


@pytest.mark.skipif(not HAS_CUDAQ, reason="requires CUDA-Q (Linux only)")
def test_real_cudaq_engine_matches_numpy_reference():
    p = _three_mode_cubic()
    out = {0: 0.2, "m": -0.3}
    ref = run(p, dims=10, measurement_outcomes=out)
    got = run(p, dims=10, measurement_outcomes=out, engine="cudaq")
    assert all(
        r.get("engine") in ("cudaq", None) for r in got.execution_record if r["op"] == "unitary"
    )
    assert _infidelity(got.state.data, ref.state.data) < 1e-6
    lossy = pg.Pattern(inputs=(0,)).extend([pg.Prepare(1, 0.4), pg.Entangle(0, 1), pg.Loss(1, 0.8)])
    lossy.measure(0, pg.Homodyne(0.5))
    lossy = lossy.append(pg.Output((1,))).validate()
    a = run(lossy, dims=10, measurement_outcomes={0: 0.1})
    b = run(lossy, dims=10, measurement_outcomes={0: 0.1}, engine="cudaq")
    assert _trace_distance(a.state.data, b.state.data) < 1e-6
