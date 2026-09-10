import numpy as np
import pytest

import photographiq as pg
from photographiq.backends import GaussianBackend


def test_exact_homodyne_schur_complement():
    b = GaussianBackend()
    b.reset(18)
    b.prepare(0, 0.4)
    b.prepare(1, 0.3)
    b.entangle(0, 1, 0.7)
    b.displace(0, 0.2, 0.6)
    initial = b.get_state()
    theta = 0.63
    h = np.array([np.cos(theta), np.sin(theta)])
    v = h @ initial.covariance[:2, :2] @ h
    cross = initial.covariance[2:, :2] @ h
    m = b.measure(0, pg.Homodyne(theta), theta)
    np.testing.assert_allclose(
        b.get_state().mean, initial.mean[2:] + cross * (m - h @ initial.mean[:2]) / v
    )
    np.testing.assert_allclose(
        b.get_state().covariance, initial.covariance[2:, 2:] - np.outer(cross, cross) / v
    )


@pytest.mark.parametrize("measurement,variance", [(pg.Homodyne.q(), 1.0), (pg.Heterodyne(), 2.0)])
def test_vacuum_measurement_statistics(measurement, variance):
    p = pg.Pattern().append(pg.Prepare(0, 0)).measure(0, measurement)
    values = pg.run_shots(p, 2500, backend="gaussian", seed=921).values(0)
    assert np.max(np.abs(values.mean(axis=0))) < 0.10
    assert np.max(np.abs(values.var(axis=0) - variance)) < 0.15


def test_seed_zero_and_independent_shots():
    p = pg.Pattern().append(pg.Prepare(0, 0)).measure(0)
    a = pg.run_shots(p, 10, backend="gaussian", seed=0).values(0)
    b = pg.run_shots(p, 10, backend="gaussian", seed=0).values(0)
    np.testing.assert_array_equal(a, b)
    assert len(set(a)) == 10


def test_detector_efficiency_and_generaldyne():
    p = pg.Pattern().append(pg.Prepare(0, 0)).measure(0, pg.Homodyne(efficiency=0.5))
    values = pg.run_shots(p, 1500, backend="gaussian", seed=23).values(0)
    assert abs(values.var() - 2) < 0.16
    pg.simulate(pg.Pattern(pg.CVGraph.line(2)).measure(0, pg.Generaldyne(np.diag([0.5, 2.0]))))
    with pytest.raises(ValueError):
        pg.Generaldyne(np.eye(2) * 0.2)
