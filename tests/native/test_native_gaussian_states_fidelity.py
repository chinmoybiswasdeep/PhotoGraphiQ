"""GaussianState.fidelity (NumPy BBP, mixed branch) needs neither Piquasso nor the extension."""

import sys

import numpy as np
import pytest
import scipy.linalg

import photographiq as pg
from photographiq.gaussian import omega


def mixed_state(rng, n):
    s = np.eye(2 * n)
    for _ in range(3):
        h = rng.normal(size=(2 * n, 2 * n))
        s = scipy.linalg.expm(omega(n) @ (h + h.T) * 0.3) @ s
    cov = s @ np.diag(np.repeat(rng.uniform(1.1, 2.5, n), 2)) @ s.T
    return pg.GaussianState(rng.normal(size=2 * n), (cov + cov.T) / 2, tuple(range(n)))


def test_mixed_fidelity_runs_without_piquasso(monkeypatch):
    monkeypatch.setitem(sys.modules, "piquasso", None)  # any import of piquasso now fails
    n1, n2 = 0.4, 1.7
    t1 = pg.GaussianState(np.zeros(2), (2 * n1 + 1) * np.eye(2), ("a",))
    t2 = pg.GaussianState([0.3, -0.2], (2 * n2 + 1) * np.eye(2), ("a",))
    # Displaced thermal pair: F = F_0 exp(-|d|^2 / (2 (V1 + V2))) for isotropic V.
    f0 = (np.sqrt((n1 + 1) * (n2 + 1)) - np.sqrt(n1 * n2)) ** -2
    expected = f0 * np.exp(-0.5 * 0.13 / (2 * n1 + 1 + 2 * n2 + 1))
    assert np.isclose(t1.fidelity(t2), expected, rtol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_numpy_fidelity_matches_piquasso(n):
    pq = pytest.importorskip("piquasso")

    def to_pq(state):
        result = pq.GaussianState(d=n, connector=pq.NumpyConnector(), config=pq.Config(hbar=2.0))
        result.xpxp_mean_vector = state.mean.copy()
        result.xpxp_covariance_matrix = 2 * state.covariance
        return result

    rng = np.random.default_rng(n)
    for _ in range(5):
        a, b = mixed_state(rng, n), mixed_state(rng, n)
        assert np.isclose(a.fidelity(b), to_pq(a).fidelity(to_pq(b)), rtol=1e-9)
