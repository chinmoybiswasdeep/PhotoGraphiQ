import numpy as np
import pytest

import photographiq as pg
from tests.v03_reference import physical


@pytest.mark.parametrize("outcome", [0, 1, 2])
def test_diagonal_count_conditioning(outcome):
    weights = np.array([0.2, 0.3, 0.5])
    source = pg.FockDensityMatrix(np.diag(weights), ((0, 2), (1, 1), (2, 0)))
    p = pg.Pattern(inputs=("a", "b")).measure("a", pg.PhotonNumber())
    result = pg.simulate(
        p,
        initial_state=source,
        backend="piquasso-mixed-fock",
        cutoff=5,
        measurement_outcomes={"a": outcome},
    )
    physical(result.state.density_matrix)
    assert np.exp(result.log_likelihood) == pytest.approx(weights[outcome], abs=1e-13)
    assert result.state.probabilities[(2 - outcome,)] == pytest.approx(1)


@pytest.mark.parametrize("angle,eta,noise", [(0.0, 1.0, 0.0), (0.6, 0.7, 0.0), (1.2, 0.8, 0.15)])
@pytest.mark.parametrize("outcome", [-0.5, 0.4])
def test_correlated_gaussian_homodyne_schur_complement(angle, eta, noise, outcome):
    r, weight, cutoff = 0.08, 0.15, 14
    p = (
        pg.Pattern(inputs=("a", "b"))
        .extend([pg.Squeeze("a", r), pg.Entangle("a", "b", weight)])
        .measure("a", pg.Homodyne(angle, efficiency=eta, noise=noise))
    )
    result = pg.simulate(
        p, backend="piquasso-mixed-fock", cutoff=cutoff, measurement_outcomes={"a": outcome}
    )
    # Build covariance directly from Heisenberg equations, vacuum covariance I.
    s = np.diag([np.exp(-r), np.exp(r), 1.0, 1.0])
    cz = np.eye(4)
    cz[1, 2] = cz[3, 0] = weight
    cov = cz @ s @ s.T @ cz.T
    h = np.array([np.cos(angle), np.sin(angle)])
    variance = h @ cov[:2, :2] @ h + (1 - eta) / eta + noise
    cross = cov[2:, :2] @ h
    mean = cross * outcome / variance
    conditional = cov[2:, 2:] - np.outer(cross, cross) / variance
    physical(result.state.density_matrix)
    density = np.exp(-(outcome**2) / (2 * variance)) / np.sqrt(2 * np.pi * variance)
    assert np.exp(result.log_likelihood) == pytest.approx(density, abs=2e-8)
    for i, a in enumerate([0, np.pi / 2]):
        np.testing.assert_allclose(
            result.state.quadrature("b", a), [mean[i], conditional[i, i]], atol=2e-7, rtol=0
        )
