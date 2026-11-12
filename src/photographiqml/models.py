"""Small logical supervised models and a quantum-output instrument."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .logical import statevector
from .training import Trainer


def angle_encode(features):
    """Product Ry encoding of shape (n_wires,), with no data normalization."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 1 or not len(features) or not np.isfinite(features).all():
        raise ValueError("Features must be a nonempty finite vector")
    state = np.array([1.0])
    for angle in features:
        state = np.kron(state, [np.cos(angle / 2), np.sin(angle / 2)])
    return state


class MuTARegressor:
    """Scalar affine head on first-wire Z expectation, for classical inputs."""

    def __init__(self, model, *, trainer=None, seed=0, physical_options=None):
        if model.representation not in ("logical", "gkp-physical"):
            raise ValueError("This supervised wrapper currently requires logical MuTA")
        if model.representation == "gkp-physical":
            from .physical_training import DiscreteSearch

            if trainer is not None and not isinstance(trainer, DiscreteSearch):
                raise ValueError(
                    "Physical angles require DiscreteSearch; Adam cannot optimize categorical measurements"
                )
            trainer = trainer or DiscreteSearch(seed=seed)
            if physical_options is None or physical_options.get("mode") != "physical-shots":
                raise ValueError(
                    "Physical supervised models require explicit physical_options with mode='physical-shots', shots and seed"
                )
            if "shots" not in physical_options or "seed" not in physical_options:
                raise ValueError("Declare physical shots and seed explicitly")
        self.physical_options = dict(physical_options or {})
        self.trainer: Any = trainer or Trainer()
        self.model, self.seed = model, seed
        self.weights = None

    def _inputs(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.model.n_wires or not np.isfinite(X).all():
            raise ValueError(f"Inputs must have shape (N,{self.model.n_wires})")
        return np.array([angle_encode(x) for x in X]).reshape(-1, 2**self.model.n_wires)

    def _predict(self, states, weights):
        if self.model.representation == "gkp-physical":
            z, _ = self._physical_features(states, weights[:-2])
            return weights[-2] * z + weights[-1]
        output = self.model.run_batch(states, weights[:-2])
        half = 2 ** (self.model.n_wires - 1)
        z = np.sum(abs(output[:, :half]) ** 2, axis=1) - np.sum(abs(output[:, half:]) ** 2, axis=1)
        return weights[-2] * z + weights[-1]

    def _physical_features(self, states, angles):
        results = [self.model.run(state, angles, **self.physical_options) for state in states]
        probabilities = [r.decoded_marginals[self.model.output_nodes[0]] for r in results]
        self.physical_diagnostics = [r.diagnostics for r in results]
        self.convergence = [r.convergence for r in results]
        return np.array([p[0] - p[1] for p in probabilities]), results

    def _loss(self, prediction, y):
        return float(np.mean((prediction - y) ** 2))

    def fit(self, X, y):
        states = self._inputs(X)
        y = np.asarray(y, dtype=float)
        if y.shape != (len(states),) or not len(y) or not np.isfinite(y).all():
            raise ValueError("Targets must have shape (N,) and be finite and nonempty")
        if self.model.representation == "gkp-physical":
            from scipy.optimize import minimize

            heads = {}
            diagnostics = {}

            def objective(angles):
                z, _ = self._physical_features(states, angles)
                # This continuous optimization is classical readout-head fitting only.
                fitted = minimize(
                    lambda head: self._loss(head[0] * z + head[1], y) + 1e-4 * np.sum(head**2),
                    [1.0, 0.0],
                    method="BFGS",
                )
                heads[tuple(angles)] = fitted.x
                diagnostics[tuple(angles)] = (self.physical_diagnostics, self.convergence)
                return float(fitted.fun)

            self.history = self.trainer.fit(
                self.model, objective, initial=np.zeros(self.model.n_parameters)
            )
            self.weights = np.r_[self.history.parameters, heads[tuple(self.history.parameters)]]
            self.physical_diagnostics, self.convergence = diagnostics[
                tuple(self.history.parameters)
            ]
            return self
        initial = np.r_[list(self.model.initialize(self.seed).values()), 1.0, 0.0]
        self.weights, self.history = self.trainer.fit(
            lambda p: self._loss(self._predict(states, p), y), initial
        )
        return self

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Call fit before predict")
        return self._predict(self._inputs(X), self.weights)


class MuTAClassifier(MuTARegressor):
    """Binary logistic head; output columns correspond to labels 0 and 1."""

    def fit(self, X, y):
        if not np.isin(np.asarray(y), [0, 1]).all():
            raise ValueError("Classifier labels must be 0 or 1")
        return super().fit(X, y)

    def _loss(self, prediction, y):
        return float(np.mean(np.logaddexp(0, prediction) - y * prediction))

    def predict_proba(self, X):
        from scipy.special import expit

        probability = expit(super().predict(X))
        return np.column_stack([1 - probability, probability])

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        prediction = self.predict(X)
        y = np.asarray(y)
        if y.shape != prediction.shape or not len(y):
            raise ValueError("Score labels must match a nonempty batch")
        return float(np.mean(prediction == y))


@dataclass(frozen=True)
class InstrumentBranch:
    outcome: int
    probability: float
    state: np.ndarray | None


class QuantumInstrumentModel:
    """MuTA followed by destructive computational measurement of one wire.

    This valid instrument is not a reproduction of the paper's learned
    teleportation instrument with variational intermediate controls.
    """

    def __init__(self, model, measured_wire=0):
        if model.representation != "logical":
            raise NotImplementedError(
                "QuantumInstrumentModel requires logical MuTA; a physical quantum-output instrument needs a separately validated post-measurement map"
            )
        if (
            isinstance(measured_wire, bool)
            or not isinstance(measured_wire, int)
            or not 0 <= measured_wire < model.n_wires
        ):
            raise ValueError("measured_wire must index a model wire")
        self.model, self.measured_wire = model, measured_wire

    def run(self, input_state, parameters=None):
        state = self.model.run(input_state, parameters).state
        tensor = np.moveaxis(
            state.reshape([2] * self.model.n_wires), self.measured_wire, 0
        ).reshape(2, -1)
        branches = []
        for bit, vector in enumerate(tensor):
            probability = float(np.vdot(vector, vector).real)
            branches.append(
                InstrumentBranch(
                    bit, probability, vector / np.sqrt(probability) if probability > 1e-15 else None
                )
            )
        return tuple(branches)


def haar_states(n_wires, n_samples, seed=0):
    from .ansatz.triangle import positive_integer

    positive_integer(n_wires, "n_wires")
    positive_integer(n_samples, "n_samples")
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n_samples, 2**n_wires)) + 1j * rng.normal(
        size=(n_samples, 2**n_wires)
    )
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def infidelity(predictions, targets):
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    if predictions.ndim != 2 or predictions.shape != targets.shape or not len(predictions):
        raise ValueError("Infidelity expects matching nonempty batches (N,2**n)")
    n_wires = int(np.log2(predictions.shape[1]))
    for p, t in zip(predictions, targets):
        statevector(p, n_wires)
        statevector(t, n_wires)
    return float(1 - np.mean(abs(np.sum(predictions.conj() * targets, axis=1)) ** 2))
