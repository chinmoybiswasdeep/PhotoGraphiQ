import numpy as np
import pytest

import photographiq as pg
from photographiq.compiler import decompose_symplectic
from photographiq.gaussian import rotation, squeezing, teleportation_matrix, wire_channel


@pytest.mark.parametrize(
    "s",
    [
        np.eye(2),
        rotation(0.7),
        rotation(np.pi),
        squeezing(0.4),
        rotation(0.3) @ squeezing(0.6) @ rotation(0.8),
    ],
)
def test_exact_compiler_matrix(s):
    actual = np.eye(2)
    for k in decompose_symplectic(s):
        actual = teleportation_matrix(k) @ actual
    np.testing.assert_allclose(actual, s, atol=1e-12)


@pytest.mark.parametrize("shears", [[0.0], [0.0] * 4, [0.3, -0.2, 0.7]])
def test_cluster_ensemble_channel(shears):
    r = 0.7
    p = pg.protocols.wire(shears, squeezing=r)
    source = pg.GaussianInput.coherent(0.3 + 0.15j)
    ensemble = pg.run_shots(
        p, 1800, inputs={0: source}, backend="gaussian", seed=892
    ).ensemble_state()
    s, noise = wire_channel(shears, r)
    np.testing.assert_allclose(ensemble.mean, s @ source.mean, atol=0.09)
    np.testing.assert_allclose(ensemble.covariance, s @ s.T + noise, atol=0.15)


def test_braunstein_kimble_ensemble():
    r = 0.8
    p = pg.protocols.teleportation(squeezing=r)
    ensemble = pg.run_shots(
        p, 1800, inputs={0: pg.GaussianInput.coherent(0.2 + 0.1j)}, backend="gaussian", seed=41
    ).ensemble_state()
    np.testing.assert_allclose(ensemble.mean, [0.4, 0.2], atol=0.08)
    np.testing.assert_allclose(ensemble.covariance, np.eye(2) * (1 + 2 * np.exp(-2 * r)), atol=0.12)


def test_adaptive_backend_agreement():
    p = pg.protocols.adaptive(squeezing=0.6)
    a = pg.simulate(p, parameters={"k": 0.4}, backend="gaussian", seed=97)
    b = pg.simulate(p, parameters={"k": 0.4}, backend="piquasso", seed=97)
    np.testing.assert_allclose(list(a.outcomes.values()), list(b.outcomes.values()), atol=1e-11)
    np.testing.assert_allclose(a.state.mean, b.state.mean, atol=1e-11)
    np.testing.assert_allclose(a.state.covariance, b.state.covariance, atol=1e-11)
