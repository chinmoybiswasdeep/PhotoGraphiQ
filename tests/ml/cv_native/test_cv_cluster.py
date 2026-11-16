"""Directly trainable weighted CV clusters (CVCluster): flow, channels, measurement semantics."""

import numpy as np
import pytest

import photographiq as pg
from photographiq import gaussian as gm
from photographiq.analysis import gaussian_channel
from photographiqml.cv import CVCluster


@pytest.mark.parametrize(
    "wires,columns,rungs", [(1, 5, None), (2, 3, None), (2, 4, (1, 3)), (3, 4, None)]
)
def test_channel_matches_flow_pattern_and_ideal_map_is_symplectic(wires, columns, rungs):
    model = CVCluster(wires, columns, squeezing=0.7, rungs=rungs, rung_weight=0.6)
    rng = np.random.default_rng(wires * 10 + columns)
    params = {n: rng.normal() for n in model.parameters}
    X, t, Y, measurements = model.channel(params, mode="physical")
    channel = gaussian_channel(model.pattern(params))
    np.testing.assert_allclose(X, channel.matrix, atol=1e-11)
    np.testing.assert_allclose(Y, channel.noise, atol=1e-11)
    np.testing.assert_allclose(X @ gm.omega(wires) @ X.T, gm.omega(wires), atol=1e-10)
    assert measurements == wires * (columns - 1)
    assert np.linalg.eigvalsh(Y).min() > 0  # finite squeezing always adds noise
    assert not model.channel(params)[2].any()  # ideal limit


def test_single_wire_is_a_product_of_teleportation_shears():
    model = CVCluster(1, 4, angles="direct")
    thetas = [0.7, 1.9, 2.4]
    X = model.channel(dict(zip(model.parameters, thetas)))[0]
    expected = np.eye(2)
    for theta in thetas:
        expected = gm.teleportation_matrix(np.cos(theta) / np.sin(theta)) @ expected
    np.testing.assert_allclose(X, expected, atol=1e-12)


def test_angles_are_pi_periodic_and_singular_angles_are_rejected():
    model = CVCluster(2, 3, angles="direct", squeezing=0.5)
    base = dict(zip(model.parameters, [0.4, 1.3, 2.0, 2.7]))
    shifted = {k: v + np.pi for k, v in base.items()}
    for a, b in zip(model.channel(base, mode="physical"), model.channel(shifted, mode="physical")):
        np.testing.assert_allclose(a, b, atol=1e-10)
    with pytest.raises(ValueError, match="singular"):
        model.channel(base | {model.parameters[0]: np.pi})
    bounded = CVCluster(2, 3)
    phis = bounded.angle_parameters([0.4, 1.3, 2.0, 2.7])
    np.testing.assert_allclose(bounded.angles(phis), [0.4, 1.3, 2.0, 2.7])
    np.testing.assert_allclose(bounded.channel(phis)[0], model.channel(base)[0], atol=1e-12)
    with pytest.raises(ValueError):
        bounded.angle_parameters([0.01, 1, 1, 1])


def test_trainable_weights_recertify_the_flow():
    model = CVCluster(2, 3, trainable_weights=True)
    assert model.weight_names[0] in model.parameters
    params = dict.fromkeys(model.parameters, 0.2)
    first = model.certificate(params)
    second = model.certificate(params | {model.weight_names[0]: 1.7})
    assert first is not second and len(model._flow_cache) == 3  # + construction weights
    # the recertified pattern implements the channel of the new weights
    new = params | {model.weight_names[0]: 1.7}
    np.testing.assert_allclose(
        model.channel(new)[0], gaussian_channel(model.pattern(new)).matrix, atol=1e-11
    )
    assert not np.allclose(model.channel(new)[0], model.channel(params)[0])


def test_invalid_layouts_are_rejected():
    with pytest.raises(ValueError):
        CVCluster(2, 1)
    with pytest.raises(ValueError):
        CVCluster(2, 3, rungs=(5,))
    with pytest.raises(ValueError):
        CVCluster(2, 3, rung_weight=0.0)
    with pytest.raises(ValueError):
        CVCluster(2, 3, angles="free")


def test_homodyne_conditioning_matches_an_independent_schur_complement():
    """One homodyne at angle theta on a correlated two-mode Gaussian state."""
    rng = np.random.default_rng(5)
    a = rng.normal(size=(4, 4))
    cov = a @ a.T + 4 * np.eye(4)
    mean = rng.normal(size=4)
    theta, outcome = 0.83, 0.4
    engine = pg.registry.create_backend("gaussian", None)
    engine.reset(0)
    engine.set_state(pg.GaussianState(mean, cov, ("m", "k")))
    engine.measure("m", pg.Homodyne(theta), theta, outcome=outcome)
    h = np.array([np.cos(theta), np.sin(theta), 0, 0])
    var = h @ cov @ h
    gain = cov[2:, :] @ h / var
    np.testing.assert_allclose(engine.state.mean, mean[2:] + gain * (outcome - h @ mean))
    np.testing.assert_allclose(engine.state.covariance, cov[2:, 2:] - np.outer(gain, gain) * var)


def test_cluster_trajectories_conditional_versus_ensemble():
    model = CVCluster(2, 3, squeezing=0.6)
    params = {n: 0.3 * (i - 1.5) for i, n in enumerate(model.parameters)}
    pattern = model.pattern(params)
    X, t, Y, _ = model.channel(params, mode="physical")
    d = np.array([0.4, 0.1, -0.3, 0.2])
    inputs = {n: pg.GaussianInput(tuple(d[2 * i : 2 * i + 2])) for i, n in enumerate(model.inputs)}
    shots = 400
    runs = pg.run_shots(pattern, shots, seed=2, inputs=inputs).trajectories
    states = [r.state.reduced(model.outputs) for r in runs]
    means = np.array([s.mean for s in states])
    for s in states[:5]:  # Gaussian conditioning: covariance independent of outcomes
        np.testing.assert_allclose(s.covariance, states[0].covariance, atol=1e-9)
    ensemble = states[0].covariance + np.cov(means.T)
    predicted = X @ X.T + Y
    se = np.sqrt(2 / shots) * np.sqrt(np.outer(np.diag(predicted), np.diag(predicted)))
    assert np.all(np.abs(ensemble - predicted) < 5 * se)
    assert np.all(np.abs(means.mean(0) - X @ d) < 5 * np.sqrt(np.diag(predicted) / shots))
    # at finite squeezing the conditional mean still depends on the outcomes
    assert np.cov(means.T).trace() > 1e-3
