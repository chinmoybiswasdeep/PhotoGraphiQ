"""CV regressors/classifiers: gradients, training, shots, penalties, resources, compatibility."""

import json

import numpy as np
import pytest

import photographiqml as pgml
from photographiqml.cv import (
    CVClassifier,
    CVCluster,
    CVFeatureMap,
    CVRegressor,
    GaussianProgram,
    central_difference,
    complex_step_gradient,
    layered_program,
)


def data(n=30, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2))
    return X, 0.7 * X[:, 0] - 0.4 * X[:, 1] + 0.2


# ---------------------------------------------------------------- gradients
def test_complex_step_matches_closed_form_derivative():
    """<q> after R(theta) on a q-displaced input d is d cos(theta): d/dtheta = -d sin(theta)."""
    model = CVRegressor(GaussianProgram(1).rotate(0, "t"), readout="q")
    X = np.array([[0.8]])

    def mean_q(v):
        return model.features({"t": v[0]}, X)[0, 0]

    for theta in (0.3, 1.2, 2.9):
        g = complex_step_gradient(mean_q, [theta])[0]
        assert g == pytest.approx(-0.8 * np.sin(theta), abs=1e-14)
        assert central_difference(mean_q, [theta])[0] == pytest.approx(g, abs=1e-8)


@pytest.mark.parametrize("program", ["compiled", "cluster"])
@pytest.mark.parametrize("r", [0.3, 1.5])
def test_complex_step_matches_central_differences_on_physical_losses(program, r):
    X, y = data(12)
    prog = layered_program(2, 1) if program == "compiled" else CVCluster(2, 3, squeezing=r)
    model = CVClassifier(
        prog,
        mode="physical",
        squeezing=r if program == "compiled" else None,
        head="homodyne",
        readout="q",
    )
    rng = np.random.default_rng(1)
    names = model.quantum_parameters
    labels = (y > 0.2).astype(int)

    def loss(v):
        params = dict(zip(names, v[:-1]))
        return model._loss(model.features(params, X), v[-1:], labels)

    v = rng.normal(scale=0.4, size=len(names) + 1)
    cs, fd = complex_step_gradient(loss, v), central_difference(loss, v, 1e-6)
    np.testing.assert_allclose(cs, fd, atol=1e-6, rtol=1e-6)


def test_gradient_near_the_homodyne_margin_is_finite_and_consistent():
    model = CVCluster(1, 3, margin=0.05)
    reg = CVRegressor(model, mode="physical")
    X = np.array([[0.5]])

    def f(v):
        return reg.features(dict(zip(model.parameters, v)), X, shots=None)[0, 0]

    v = np.array([4.0, -4.0])  # theta within ~1e-5 rad of the excluded margin
    cs = complex_step_gradient(f, v)
    assert np.all(np.isfinite(cs))
    np.testing.assert_allclose(cs, central_difference(f, v, 1e-5), rtol=1e-4, atol=1e-8)


# ---------------------------------------------------------------- training
def test_regressor_learns_an_affine_target_reproducibly():
    X, y = data()
    fit = lambda: CVRegressor(layered_program(2, 1), epochs=60, seed=3).fit(X, y, X[:10], y[:10])  # noqa: E731
    a, b = fit(), fit()
    assert a.score(X, y)["mse"] < 1e-3
    assert a.history.losses == b.history.losses and a.history.parameters == b.history.parameters
    h = a.history
    assert len(h.losses) == len(h.validation_losses) == len(h.gradient_norms) == 61
    assert h.evaluations[-1] > h.evaluations[0] and h.settings["gradient"] == "complex-step"
    json.dumps(h.as_dict())


def test_gaussian_mean_readout_is_affine_in_displacement_features():
    """Fixed Gaussian program + displacement encoding + linear mean readout = affine map."""
    X, y = data(10)
    model = CVRegressor(layered_program(2, 2), epochs=3).fit(X, y)
    a, b = np.array([[0.3, -0.2]]), np.array([[-0.6, 0.9]])
    f = lambda z: model.predict(z)[0]  # noqa: E731
    for lam in (0.25, 0.5, 2.0):
        assert f(lam * a + (1 - lam) * b) == pytest.approx(lam * f(a) + (1 - lam) * f(b), abs=1e-10)


@pytest.mark.parametrize("optimizer", ["sgd", "lbfgs"])
def test_other_optimisers_and_user_gradients(optimizer):
    X, y = data(20)
    model = CVRegressor(CVCluster(2, 3), optimizer=optimizer, epochs=15, learning_rate=0.02)
    losses = model.fit(X, y).history.losses
    assert losses[-1] < losses[0]
    calls = []

    def gradient(v):
        calls.append(1)
        return central_difference(lambda w: float(np.real(model_obj(w))), v)

    user = CVRegressor(GaussianProgram(2).rotate(0, "a"), epochs=2, gradient=gradient)
    model_obj = lambda w: np.mean((user._predict(user.features({"a": w[0]}, X), w[1:]) - y) ** 2)  # noqa: E731
    user.fit(X, y)
    assert calls and user.history.settings["gradient"] == "user"


def test_classifiers_binary_homodyne_and_multiclass():
    rng = np.random.default_rng(4)
    X = rng.uniform(-1, 1, (40, 2))
    yb = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    logistic = CVClassifier(CVCluster(2, 3), epochs=40, learning_rate=0.1).fit(X, yb)
    assert logistic.score(X, yb)["accuracy"] >= 0.95
    proba = logistic.predict_proba(X)
    np.testing.assert_allclose(proba.sum(1), 1)
    homodyne = CVClassifier(
        layered_program(2, 1), head="homodyne", epochs=40, learning_rate=0.1
    ).fit(X, yb)
    assert homodyne.score(X, yb)["accuracy"] >= 0.85
    sampled = homodyne.predict_proba(X, shots=50, seed=1)
    np.testing.assert_allclose(sampled * 50, np.round(sampled * 50), atol=1e-9)
    y3 = np.digitize(X[:, 0], [-0.33, 0.33])
    soft = CVClassifier(
        layered_program(2, 1), head="softmax", classes=3, readout="qp", epochs=60, learning_rate=0.1
    ).fit(X, y3)
    assert soft.score(X, y3)["accuracy"] >= 0.8
    with pytest.raises(ValueError):
        CVClassifier(CVCluster(2, 3), head="softmax", classes=2)
    with pytest.raises(ValueError):
        CVClassifier(CVCluster(2, 3), head="homodyne", readout="p")


def test_shot_estimator_is_exact_in_distribution_and_reproducible():
    X, y = data(8)
    model = CVRegressor(
        CVCluster(2, 3, squeezing=0.4), mode="physical", shots=64, epochs=5, seed=2
    ).fit(X, y)
    a, b = model.predict(X, shots=64, seed=9), model.predict(X, shots=64, seed=9)
    np.testing.assert_array_equal(a, b)
    exact = model.predict(X)
    means, covs = model.moments(model.params, X)
    var = np.einsum("j,njk,k->n", model.head[:-1], covs[:, 0::2, 0::2], model.head[:-1]) / 64
    draws = np.array([model.predict(X, shots=64, seed=s) for s in range(300)])
    z = (draws.mean(0) - exact) / np.sqrt(var / 300)
    assert np.all(np.abs(z) < 5)
    np.testing.assert_allclose(draws.var(0), var, rtol=0.35)


def test_penalties_and_finite_squeezing_semantics():
    X, y = data(15)
    prog = layered_program(2, 1)
    noisy = CVRegressor(
        prog, mode="physical", squeezing=0.5, penalty="noise", penalty_weight=0.1, epochs=10
    )
    noisy.fit(X, y)
    assert noisy.history.settings["penalty"] == "noise"
    with pytest.raises(ValueError):
        CVRegressor(prog, penalty="noise")  # needs physical execution
    with pytest.raises(ValueError):
        CVRegressor(prog, mode="physical")  # compiled execution needs the resource squeezing
    # flow-corrected unconditional means are those of the ideal circuit at any squeezing
    params = dict.fromkeys(prog.parameters, 0.2)
    m_ideal, c_ideal = noisy.moments(params, X, mode="ideal")
    m_phys, c_phys = noisy.moments(params, X, mode="physical")
    np.testing.assert_allclose(m_ideal, m_phys, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(c_phys - c_ideal) > 0)


def test_resource_report_and_configuration_errors():
    X, y = data(6)
    model = CVRegressor(CVCluster(2, 4, squeezing=1.0), mode="physical", epochs=2).fit(X, y)
    report = model.resource_report(X)
    assert report["measurements"] == 6 and report["resource_modes"] == 6
    assert report["resource_squeezing_db"] == pytest.approx(8.686, abs=1e-3)
    assert report["measurement_depth"] >= 3 and report["added_noise_per_quadrature"] > 0
    json.dumps({k: float(v) if isinstance(v, np.floating) else v for k, v in report.items()})
    with pytest.raises(TypeError):
        CVRegressor(object())
    with pytest.raises(ValueError):
        CVRegressor(CVCluster(2, 3), CVFeatureMap(3))
    with pytest.raises(ValueError):
        CVRegressor(CVCluster(2, 3), readout="moments", shots=10)
    with pytest.raises(RuntimeError):
        CVRegressor(CVCluster(2, 3)).predict(X)


def test_existing_model_families_are_untouched():
    for name in (
        "MuTA",
        "MuTAKernel",
        "MuTAClassifier",
        "MuTARegressor",
        "PhysicalMuTA",
        "GKPBridge",
        "Trainer",
    ):
        assert hasattr(pgml, name)
    assert pgml.MuTA(1, 1).representation == "logical"
    with pytest.raises(AttributeError):
        CVCluster(1, 2).representation  # noqa: B018  (the CV family is not a MuTA representation)


def test_trainable_rung_weights_use_real_derivatives_and_start_at_their_value():
    X, y = data(12)
    cluster = CVCluster(2, 3, trainable_weights=True, rung_weight=0.8)
    model = CVRegressor(cluster, mode="physical", epochs=5, init_scale=0.0)
    assert model.gradient == "finite-difference"  # complex step cannot see the weights
    with pytest.raises(ValueError):
        CVRegressor(CVCluster(2, 3, trainable_weights=True), gradient="complex-step")
    model.fit(X, y)
    first = dict(zip(model.history.settings["parameter_names"], model.history.parameters[0]))
    assert first[cluster.weight_names[0]] == pytest.approx(0.8)
    assert model.params[cluster.weight_names[0]] != pytest.approx(0.8)  # the weight trains


def test_rejected_configurations_after_review():
    with pytest.raises(ValueError):
        CVRegressor(CVCluster(2, 3), penalty="squeezing", penalty_weight=0.1)
    X, y = data(6)
    ideal = CVRegressor(layered_program(2, 1), epochs=1).fit(X, y)
    report = ideal.resource_report(X)
    assert report["resource_squeezing_r"] is None and report["structure_compiled_at_r"] == 1.0
