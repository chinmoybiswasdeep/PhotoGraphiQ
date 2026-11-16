"""CV feature maps and trainable Gaussian programs: analytic, compiled and executed references."""

import numpy as np
import pytest

import photographiq as pg
from photographiq import gaussian as gm
from photographiq.analysis import gaussian_channel
from photographiqml.cv import CVFeatureMap, Feature, GaussianProgram, layered_program


def random_program(rng):
    prog = layered_program(2, 1)
    prog.cz(0, 1, "g").shear(1, "sh").displace(0, Feature(1, 0.5), "dp").rotate(1, Feature(0))
    params = {n: rng.normal(scale=0.6) for n in prog.parameters}
    return prog, params


# ---------------------------------------------------------------- feature maps
def test_displacement_encoding_moments_and_energy():
    fmap = CVFeatureMap(2, scale=0.7, quadrature="p")
    mean, cov = fmap.moments(np.array([0.3, -1.2]))
    np.testing.assert_allclose(mean, [0, 0.21, 0, -0.84])
    np.testing.assert_allclose(cov, np.eye(4))
    # coherent amplitude alpha = mean / 2 (hbar = 2): <n> = |alpha|^2 summed over modes
    assert fmap.mean_photons(np.array([0.3, -1.2])) == pytest.approx((0.21**2 + 0.84**2) / 4)


def test_squeezing_and_rotation_encodings():
    sq = CVFeatureMap(1, "squeezing", scale=0.5)
    _, cov = sq.moments(np.array([0.8]))
    np.testing.assert_allclose(cov, np.diag([np.exp(-0.8), np.exp(0.8)]))
    assert sq.mean_photons(np.array([0.8])) == pytest.approx(np.sinh(0.4) ** 2)
    rot = CVFeatureMap(1, "rotation", scale=1.0, offset=2.0)
    mean, cov = rot.moments(np.array([np.pi / 2]))
    np.testing.assert_allclose(mean, [0, 2], atol=1e-15)
    np.testing.assert_allclose(cov, np.eye(2), atol=1e-15)


def test_feature_map_rejects_invalid_inputs():
    fmap = CVFeatureMap(2, "squeezing", scale=1.0, max_squeezing=1.0)
    for bad in ([[0.1, 0.2, 0.3]], [[np.nan, 0.0]], [[2.0, 0.0]], [0.1, 0.2]):
        with pytest.raises(ValueError):
            fmap.check(bad)
    for kwargs in (
        {"encoding": "amplitude"},
        {"quadrature": "x"},
        {"scale": np.inf},
        {"input_squeezing": 2.0, "max_squeezing": 1.0},
    ):
        with pytest.raises(ValueError):
            CVFeatureMap(1, **kwargs)
    with pytest.raises(ValueError):
        CVFeatureMap(0)


# ---------------------------------------------------------------- ideal programs
def test_ideal_channel_matches_independent_gate_products():
    prog = GaussianProgram(2).rotate(0, "a").squeeze(1, "r").beamsplitter(0, 1, "t").cz(1, 0, "g")
    prog.shear(0, "s").displace(1, "q", 0.3)
    params = {"a": 0.4, "r": 0.3, "t": 1.1, "g": -0.7, "s": 0.25, "q": -0.5}
    X, t, Y, _ = prog.channel(params)
    expected = (
        gm.embed(np.array([[1, 0], [0.25, 1]]), [0, 1], 4)
        @ gm.embed(gm.cz(-0.7), [2, 3, 0, 1], 4)
        @ gm.beamsplitter(1.1)
        @ gm.embed(gm.squeezing(0.3), [2, 3], 4)
        @ gm.embed(gm.rotation(0.4), [0, 1], 4)
    )
    np.testing.assert_allclose(X, expected, atol=1e-14)
    np.testing.assert_allclose(t, [0, 0, -0.5, 0.3], atol=1e-14)
    np.testing.assert_allclose(X @ gm.omega(2) @ X.T, gm.omega(2), atol=1e-13)
    assert not Y.any()


def test_ideal_moments_match_gate_level_pattern_execution():
    rng = np.random.default_rng(3)
    prog, params = random_program(rng)
    fmap = CVFeatureMap(2, scale=0.8)
    x = np.array([0.4, -0.9])
    X, t, _, _ = prog.channel(params, x)
    d, V = fmap.moments(x)
    inputs = {
        j: pg.GaussianInput(
            tuple(d[2 * j : 2 * j + 2]), tuple(map(tuple, V[2 * j : 2 * j + 2, 2 * j : 2 * j + 2]))
        )
        for j in range(2)
    }
    for backend in ("gaussian", "native-gaussian"):
        state = pg.simulate(prog.gate_pattern(params, x), backend=backend, inputs=inputs).state
        np.testing.assert_allclose(state.mean, X @ d + t, atol=1e-12)
        np.testing.assert_allclose(state.covariance, X @ V @ X.T, atol=1e-12)


# ---------------------------------------------------------------- compiled CV-MBQC channel
@pytest.mark.parametrize("r", [0.4, 1.2])
def test_compiled_channel_matches_the_executable_pattern(r):
    rng = np.random.default_rng(7)
    prog, params = random_program(rng)
    x = np.array([0.3, 0.8])
    X, t, Y, steps = prog.channel(params, x, mode="physical", squeezing=r)
    pattern = prog.circuit(params, x).compile(squeezing=r)
    channel = gaussian_channel(pattern)
    np.testing.assert_allclose(X, channel.matrix, atol=1e-11)
    np.testing.assert_allclose(Y, channel.noise, atol=1e-10)
    np.testing.assert_allclose(t, channel.displacement, atol=1e-11)
    np.testing.assert_allclose(X, prog.channel(params, x)[0], atol=1e-11)  # logical map exact
    assert steps == sum(isinstance(c, pg.Measure) for c in pattern.commands)
    assert np.linalg.eigvalsh(Y).min() > -1e-10


def test_finite_squeezing_noise_scales_as_exp_minus_two_r():
    prog = layered_program(2, 1)
    params = dict.fromkeys(prog.parameters, 0.3)
    noise = [
        prog.channel(params, mode="physical", squeezing=r)[2] * np.exp(2 * r)
        for r in (0.5, 1.0, 2.0)
    ]
    np.testing.assert_allclose(noise[0], noise[1], atol=1e-10)
    np.testing.assert_allclose(noise[0], noise[2], atol=1e-10)


def test_compiled_pattern_trajectories_reproduce_the_ensemble_channel():
    prog = GaussianProgram(1).rotate(0, 0.6).squeeze(0, 0.3)
    r, shots = 0.8, 400
    pattern = prog.circuit({}).compile(squeezing=r)
    X, t, Y, _ = prog.channel({}, mode="physical", squeezing=r)
    d = np.array([0.5, -0.2])
    result = pg.run_shots(
        pattern, shots, seed=11, inputs={0: pg.GaussianInput(tuple(d))}
    ).trajectories
    out = pattern.outputs[0]
    cond_means = np.array([s.state.mean for s in result])
    cond_cov = result[0].state.covariance
    # conditional covariance is outcome independent; ensemble = conditional + spread of means
    for s in result[:5]:
        np.testing.assert_allclose(s.state.covariance, cond_cov, atol=1e-10)
    ensemble = cond_cov + np.cov(cond_means.T)
    predicted = X @ X.T + Y
    se = np.sqrt(2 / shots) * np.sqrt(np.outer(np.diag(predicted), np.diag(predicted)))
    assert np.all(np.abs(ensemble - predicted) < 5 * se)
    assert np.all(
        np.abs(cond_means.mean(0) - (X @ d + t)) < 5 * np.sqrt(np.diag(predicted) / shots)
    )
    assert out in result[0].state.nodes


def test_bounded_squeezing_and_serialisation_roundtrip():
    prog = GaussianProgram(1).squeeze(0, "s", bound=0.5).rotate(0, Feature(0, 2.0))
    assert prog.squeezing_values({"s": 50.0}, [0.1])[0] == pytest.approx(0.5)
    again = GaussianProgram.from_config(prog.config())
    params, x = {"s": 0.3}, np.array([0.2])
    for a, b in zip(prog.channel(params, x), again.channel(params, x)):
        np.testing.assert_array_equal(a, b)
    assert prog.data_dependent and prog.parameters == ("s",)


def test_program_validation_errors():
    with pytest.raises(ValueError):
        GaussianProgram(2).beamsplitter(0, 0, 0.1)
    with pytest.raises(ValueError):
        GaussianProgram(1).rotate(0, np.nan)
    with pytest.raises(ValueError):
        GaussianProgram(1).squeeze(0, "s", bound=-1)
    with pytest.raises(ValueError):
        GaussianProgram(1).rotate(0, 0.1).channel({}, mode="physical")
    with pytest.raises(ValueError):
        GaussianProgram(1).rotate(0, Feature(3)).channel({}, [0.1])
