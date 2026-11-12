"""Gaussian-backend postselection and outcome densities, validated independently.

Oracles: closed-form Gaussian conditioning, and Fock/mixed-Fock postselection at converged
cutoffs (independent Hermite-function projection code paths).
"""

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

import photographiq as pg
from photographiq.expressions import atan2
from photographiq.fock_analysis import quadrature_moment

SOURCE = {"L": pg.GaussianInput.coherent(0.3 - 0.2j)}


def _fock_moments(state, node):
    mean = np.array([quadrature_moment(state, node, 1, a) for a in (0, np.pi / 2)])
    second = [quadrature_moment(state, node, 2, a) for a in (0, np.pi / 2, np.pi / 4)]
    vq, vp = second[0] - mean[0] ** 2, second[1] - mean[1] ** 2
    cqp = second[2] - (mean[0] + mean[1]) ** 2 / 2 - (vq + vp) / 2
    return mean, np.array([[vq, cqp], [cqp, vp]])


def _adaptive_wire(r, efficiency=1.0, noise=0.0):
    # Adapted from photonic-qrc-eoc (https://github.com/chinmoybiswasdeep/photonic-qrc-eoc,
    # branch CV, commit 493f4c3f186ee314796964d6ddb6b2749171b7bb), MIT License,
    # Copyright (c) 2026 Chinmoy Biswas; notebook
    # 2b_adaptive_cv_mbqrc_within_pattern_lyapunov_preflight_colab, cell 17 (RefInterpreter
    # validation pattern): two wire steps, the second basis atan2(1, k + c * Outcome).
    first = np.arctan2(1, 0.3)
    second = atan2(1.0, 0.2 + 0.5 * pg.Outcome("a"))
    detector = {"efficiency": efficiency, "noise": noise}
    return (
        pg.Pattern(inputs=("L",))
        .extend(
            [
                pg.Prepare("C1", r),
                pg.Entangle("L", "C1", 1.0),
                pg.Measure("L", pg.Homodyne(first, **detector), "a"),
                pg.Displace("C1", q=-pg.Outcome("a") / np.sin(first)),
                pg.Prepare("C2", r),
                pg.Entangle("C1", "C2", 1.0),
                pg.Measure("C1", pg.Homodyne(second, **detector), "b"),
                pg.Displace("C2", q=-pg.Outcome("b") / pg.sin(second)),
                pg.Output(("C2",)),
            ]
        )
        .validate()
    )


def test_adaptive_wire_matches_fock_postselection():
    pattern, outcomes = _adaptive_wire(0.3), {"a": 0.4, "b": -0.7}
    gaussian = pg.simulate(
        pattern, backend="gaussian", inputs=SOURCE, measurement_outcomes=outcomes
    )
    fock = pg.simulate(
        pattern, backend="piquasso-fock", cutoff=36, inputs=SOURCE, measurement_outcomes=outcomes
    )
    assert all(item["postselected"] for item in gaussian.measurement_statistics.values())
    assert gaussian.log_likelihood == pytest.approx(fock.log_likelihood, abs=1e-6)
    mean, cov = _fock_moments(fock.state, "C2")
    np.testing.assert_allclose(gaussian.state.mean, mean, atol=1e-6)
    np.testing.assert_allclose(gaussian.state.covariance, cov, atol=1e-5)


def test_lossy_inefficient_noisy_homodyne_matches_mixed_fock():
    pattern = (
        pg.Pattern(inputs=("L",))
        .extend(
            [
                pg.Prepare("V", 0.3),
                pg.BeamSplitter("L", "V", 0.6),
                pg.Loss("L", 0.9),
                pg.Measure("L", pg.Homodyne(0.4, efficiency=0.85, noise=0.05), "a"),
                pg.Displace("V", q=0.5 * pg.Outcome("a")),
                pg.Output(("V",)),
            ]
        )
        .validate()
    )
    kwargs = {"inputs": SOURCE, "measurement_outcomes": {"a": 0.4}}
    gaussian = pg.simulate(pattern, backend="gaussian", **kwargs)
    mixed = pg.simulate(pattern, backend="piquasso-mixed-fock", cutoff=12, **kwargs)
    assert gaussian.log_likelihood == pytest.approx(mixed.log_likelihood, abs=1e-5)
    mean, cov = _fock_moments(mixed.state, "V")
    np.testing.assert_allclose(gaussian.state.mean, mean, atol=1e-5)
    np.testing.assert_allclose(gaussian.state.covariance, cov, atol=1e-5)


def _cz_pair(r, weight, measurement):
    # q-squeezed input 0 (Var q0 = e^{-2r}), vacuum input 1, CZ, then measure mode 0.
    # Closed form as in the Kalman update of photonic-qrc-eoc (https://github.com/
    # chinmoybiswasdeep/photonic-qrc-eoc, branch CV, commit
    # 493f4c3f186ee314796964d6ddb6b2749171b7bb), MIT License, Copyright (c) 2026 Chinmoy
    # Biswas; notebook 1_unified_bosonic_eoc_dv_cv_mbqrc_colab, cell 93.
    pattern = pg.Pattern(inputs=(0, 1)).extend(
        [pg.Entangle(0, 1, weight), pg.Measure(0, measurement, "m"), pg.Output((1,))]
    )
    return pattern.validate(), {0: pg.GaussianInput.squeezed(r)}


@pytest.mark.parametrize("backend", ["gaussian", "piquasso"])
@pytest.mark.parametrize("efficiency,noise", [(1.0, 0.0), (0.7, 0.2)])
def test_cz_pair_closed_form_postselection(backend, efficiency, noise):
    r, g, m = 0.4, 0.8, 0.9
    var_q, sigma2 = np.exp(-2 * r), (1 - efficiency) / efficiency + noise
    pattern, inputs = _cz_pair(r, g, pg.Homodyne(0.0, efficiency, noise))
    result = pg.simulate(pattern, backend=backend, inputs=inputs, measurement_outcomes={"m": m})
    gain = g * var_q / (var_q + sigma2)
    np.testing.assert_allclose(result.state.mean, [0.0, m * gain], atol=1e-13)
    expected = np.diag([1.0, 1 + g**2 * var_q - g * var_q * gain])
    np.testing.assert_allclose(result.state.covariance, expected, atol=1e-13)
    assert result.outcomes["m"] == m
    stats = result.measurement_statistics["m"]
    assert stats["kind"] == "density" and stats["postselected"]
    assert stats["value"] == pytest.approx(norm.pdf(m, 0, np.sqrt(var_q + sigma2)), rel=1e-13)


@pytest.mark.parametrize("seed_squeezing", [None, 0.3])
def test_cz_pair_closed_form_phase_space_postselection(seed_squeezing):
    g, outcome = 0.8, (0.5, -1.1)
    if seed_squeezing is None:
        measurement, seed = pg.Heterodyne(), np.eye(2)
    else:
        seed = np.diag([np.exp(-2 * seed_squeezing), np.exp(2 * seed_squeezing)])
        measurement = pg.Generaldyne(seed)
    pattern, _ = _cz_pair(0.0, g, measurement)
    result = pg.simulate(pattern, backend="gaussian", measurement_outcomes={"m": outcome})
    total = np.diag([1.0, 1 + g**2]) + seed  # mode 0 before detection
    x, y = outcome
    np.testing.assert_allclose(
        result.state.mean, [g * y / total[1, 1], g * x / total[0, 0]], atol=1e-13
    )
    expected = multivariate_normal.pdf(outcome, np.zeros(2), total)
    assert np.exp(result.log_likelihood) == pytest.approx(expected, rel=1e-12)
    assert result.outcomes["m"] == outcome


@pytest.mark.parametrize("backend", ["gaussian", "piquasso"])
def test_sampled_density_is_analytic_density(backend):
    r, g, efficiency, noise = 0.4, 0.8, 0.7, 0.2
    pattern, inputs = _cz_pair(r, g, pg.Homodyne(0.0, efficiency, noise))
    scale = np.sqrt(np.exp(-2 * r) + (1 - efficiency) / efficiency + noise)
    for seed in range(5):
        result = pg.simulate(pattern, backend=backend, inputs=inputs, seed=seed)
        m = result.outcomes["m"]
        assert not result.measurement_statistics["m"]["postselected"]
        assert np.exp(result.log_likelihood) == pytest.approx(norm.pdf(m, 0, scale), rel=1e-12)


@pytest.mark.parametrize("frame", [False, True])
def test_postselection_replays_sampled_trajectory(frame):
    pattern = _adaptive_wire(0.6, efficiency=0.85, noise=0.05)
    for seed in range(4):
        sampled = pg.simulate(pattern, backend="gaussian", inputs=SOURCE, seed=seed, frame=frame)
        replay = {
            backend: pg.simulate(
                pattern,
                backend=backend,
                inputs=SOURCE,
                frame=frame,
                measurement_outcomes=sampled.outcomes,
            )
            for backend in ("gaussian", "piquasso")
        }
        for result in replay.values():
            assert result.outcomes == sampled.outcomes
            np.testing.assert_allclose(result.state.mean, sampled.state.mean, atol=1e-12)
            np.testing.assert_allclose(
                result.state.covariance, sampled.state.covariance, atol=1e-12
            )
            assert result.log_likelihood == pytest.approx(sampled.log_likelihood, abs=1e-12)


def test_frame_and_physical_displacements_postselect_identically():
    pattern, outcomes = _adaptive_wire(0.6, efficiency=0.85), {"a": 0.4, "b": -0.7}
    physical, virtual = (
        pg.simulate(
            pattern, backend="gaussian", inputs=SOURCE, frame=f, measurement_outcomes=outcomes
        )
        for f in (False, True)
    )
    assert virtual.physical_displacements < physical.physical_displacements
    assert virtual.outcomes == physical.outcomes == outcomes
    np.testing.assert_allclose(virtual.state.mean, physical.state.mean, atol=1e-12)
    np.testing.assert_allclose(virtual.state.covariance, physical.state.covariance, atol=1e-12)
    assert virtual.log_likelihood == pytest.approx(physical.log_likelihood, abs=1e-12)


def test_gaussian_postselection_guards():
    pattern, inputs = _cz_pair(0.4, 0.8, pg.Homodyne(0.0))
    with pytest.raises(ValueError, match="key"):
        pg.simulate(pattern, backend="gaussian", measurement_outcomes={"missing": 0.0})
    hetero, _ = _cz_pair(0.0, 0.8, pg.Heterodyne())
    for frame in (False, True):
        for bad in (np.nan, (0.1, 0.2), "x"):
            with pytest.raises((ValueError, TypeError)):
                pg.simulate(
                    pattern, backend="gaussian", frame=frame, measurement_outcomes={"m": bad}
                )
        for bad in (0.1, (0.1, np.inf), (0.1, 0.2, 0.3)):
            with pytest.raises(ValueError):
                pg.simulate(
                    hetero, backend="gaussian", frame=frame, measurement_outcomes={"m": bad}
                )
    counting = pg.Pattern().append(pg.Prepare(0, 0)).measure(0, pg.PhotonNumber(), key="n")
    with pytest.raises(NotImplementedError):
        pg.simulate(counting, backend="gaussian", measurement_outcomes={"n": 0})


def test_far_tail_postselection_raises_instead_of_minus_infinite_likelihood():
    # Review finding: a ~150-sigma record underflowed to density 0 and log_likelihood -inf.
    pattern = pg.Pattern().extend([pg.Prepare(0, state=pg.GaussianInput.squeezed(8.0))])
    pattern.measure(0, pg.Homodyne.q())
    with pytest.raises(ValueError, match="zero numerical density"):
        pg.simulate(pattern, backend="gaussian", measurement_outcomes={0: 0.05})
