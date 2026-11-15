import numpy as np
import pytest
from scipy.linalg import expm

from photographiqml import (
    MuTA,
    MuTAClassifier,
    MuTAKernel,
    MuTARegressor,
    QuantumInstrumentModel,
    Trainer,
)
from photographiqml.diagnostics import pure_qfi
from photographiqml.logical import X, Z
from photographiqml.models import haar_states, infidelity
from photographiqml.training import finite_difference


@pytest.mark.parametrize("optimizer", ["adam", "sgd", "lbfgs"])
def test_optimizer(optimizer):
    point, history = Trainer(optimizer, epochs=180).fit(lambda x: np.sum((x - 0.3) ** 2), [1, -1])
    assert np.linalg.norm(point - 0.3) < 1e-3
    assert history.losses[-1] < history.losses[0]


def test_gradient_and_gate_learning():
    model = MuTA(1, one_column=True)
    states = haar_states(1, 7, 42)
    target = expm(0.5j * 0.8 * X)
    targets = states @ target.T

    def objective(p):
        return infidelity(model.run_batch(states, p), targets)

    initial = np.array(list(model.initialize(0).values()))
    # Each angle occurs in a single +/-1/2-generator gate, so expectation
    # infidelity obeys the standard two-term shift rule for this objective.
    shifted = []
    for i in range(4):
        delta = np.eye(4)[i] * np.pi / 2
        shifted.append((objective(initial + delta) - objective(initial - delta)) / 2)
    assert np.allclose(finite_difference(objective, initial), shifted, atol=1e-8)
    point, history = Trainer(epochs=100).fit(objective, initial)
    assert history.losses[-1] < 1e-4
    assert objective(point) < 1e-4


def test_kernel_equation_and_psd():
    kernel = MuTAKernel()
    data = np.random.default_rng(42).normal(size=(12, 2))
    states = kernel.features(data)
    for row, (x0, x1) in zip(states, data):
        rz = np.kron(expm(0.5j * x0 * Z), expm(0.5j * x1 * Z))
        expected = (
            rz @ expm(0.5j * np.cos(x0) * np.cos(x1) * np.kron(X, X)) @ rz @ np.array([1, 0, 0, 0])
        )
        assert abs(np.vdot(row, expected)) ** 2 == pytest.approx(1)
    d = kernel.diagnostics(data)
    assert d["minimum_eigenvalue"] > -1e-10
    assert d["diagonal_error"] < 1e-10
    assert np.allclose(kernel(data, data), kernel.gram_matrix(data))


def test_classifier_and_regressor():
    data = np.array([[0], [0.1], [3.0], [3.14]])
    classifier = MuTAClassifier(MuTA(1), trainer=Trainer(epochs=60)).fit(data, [1, 1, 0, 0])
    assert classifier.score(data, [1, 1, 0, 0]) == 1
    assert np.allclose(classifier.predict_proba(data).sum(axis=1), 1)
    regressor = MuTARegressor(MuTA(1), trainer=Trainer(epochs=40)).fit(data, np.cos(data[:, 0]))
    assert np.max(abs(regressor.predict(data) - np.cos(data[:, 0]))) < 0.05


def test_instrument_and_qfi():
    instrument = QuantumInstrumentModel(MuTA(2, one_column=True))
    branches = instrument.run([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
    assert sum(b.probability for b in branches) == pytest.approx(1)
    assert all(b.probability == pytest.approx(0.5) for b in branches)
    assert abs(branches[0].state[0]) == pytest.approx(1)
    assert abs(branches[1].state[1]) == pytest.approx(1)
    assert pure_qfi([1 / np.sqrt(2), 1 / np.sqrt(2)], Z / 2) == pytest.approx(1)
