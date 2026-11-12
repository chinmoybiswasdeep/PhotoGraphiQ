"""Deterministic objective training; no stochastic-trajectory autodiff claims."""

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy.optimize import minimize


def finite_difference(objective, x, step=1e-6):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(step) or step <= 0:
        raise ValueError("Use a finite parameter vector and positive finite difference step")
    gradient = np.empty_like(x)
    for i in range(len(x)):
        delta = np.zeros_like(x)
        delta[i] = step
        gradient[i] = (objective(x + delta) - objective(x - delta)) / (2 * step)
    return gradient


@dataclass
class History:
    losses: list = field(default_factory=list)
    validation_losses: list = field(default_factory=list)
    gradient_norms: list = field(default_factory=list)
    parameter_norms: list = field(default_factory=list)
    seconds: list = field(default_factory=list)


class Trainer:
    """Optimize a deterministic scalar objective of a vector.

    Gradients default to central differences; supply a validated derivative
    callable for another strategy. Histories include the initial point.
    """

    def __init__(self, optimizer="adam", epochs=100, learning_rate=0.05):
        if optimizer not in ("adam", "sgd", "lbfgs"):
            raise ValueError("optimizer must be adam, sgd or lbfgs")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 0:
            raise ValueError("epochs must be a nonnegative integer")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        self.optimizer, self.epochs, self.learning_rate = optimizer, epochs, learning_rate

    def fit(self, objective, initial, *, gradient=None, validation=None, callback=None):
        x = np.asarray(initial, dtype=float).copy()
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError("initial must be a finite vector")
        grad = gradient or (lambda p: finite_difference(objective, p))
        history = History()
        start = perf_counter()

        def record(point):
            value = float(objective(point))
            derivative = np.asarray(grad(point), dtype=float)
            if (
                not np.isfinite(value)
                or derivative.shape != point.shape
                or not np.isfinite(derivative).all()
            ):
                raise ValueError("Objective and gradient must be finite and have compatible shapes")
            history.losses.append(value)
            history.gradient_norms.append(float(np.linalg.norm(derivative)))
            history.parameter_norms.append(float(np.linalg.norm(point)))
            history.seconds.append(perf_counter() - start)
            if validation is not None:
                history.validation_losses.append(float(validation(point)))
            if callback is not None:
                callback(point.copy(), history)
            return derivative

        g = record(x)
        if self.optimizer == "lbfgs" and self.epochs and len(x):
            result = minimize(
                objective,
                x,
                jac=grad,
                method="L-BFGS-B",
                callback=record,
                options={"maxiter": self.epochs},
            )
            return result.x, history
        m, v = np.zeros_like(x), np.zeros_like(x)
        for t in range(1, self.epochs + 1):
            if self.optimizer == "adam":
                m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g**2
                update = (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
            else:
                update = g
            x -= self.learning_rate * update
            g = record(x)
        return x, history
