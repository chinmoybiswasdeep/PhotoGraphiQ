"""Tutorial 1: a first CV regressor (Gaussian program, displacement encoding, affine head).

Quantum: coherent-state encoding D(x_j)|0> per mode and a trainable two-mode Gaussian
program; readout is the mean q quadrature of each output mode (exact moments).
Classical: the affine head w . <q> + b and the Adam optimiser (complex-step gradients).

Key claim checked: on an affine target the model reaches the ordinary-least-squares test
error, and compiled CV-MBQC execution at finite squeezing gives the same predictions
(flow-corrected unconditional means do not depend on the resource squeezing).
"""

from time import perf_counter

import numpy as np

from photographiqml.cv import CVFeatureMap, CVRegressor, layered_program

start = perf_counter()
rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, (60, 2))
y = 0.7 * X[:, 0] - 0.4 * X[:, 1] + 0.2 + 0.02 * rng.standard_normal(60)
X_train, y_train, X_test, y_test = X[:40], y[:40], X[40:], y[40:]

program = layered_program(2, 1)  # R, S (bounded |r| <= 1), beam splitter, R, D per mode
model = CVRegressor(program, CVFeatureMap(2, scale=1.0), readout="q", epochs=80, seed=0)
model.fit(X_train, y_train)

test_mse = model.score(X_test, y_test)["mse"]
A = np.c_[X_train, np.ones(len(X_train))]
coef = np.linalg.lstsq(A, y_train, rcond=None)[0]
ols_mse = float(np.mean((np.c_[X_test, np.ones(len(X_test))] @ coef - y_test) ** 2))

# The same trained gates, compiled to a finite-squeezing CV-MBQC pattern (r = 1).
physical = CVRegressor(program, mode="physical", squeezing=1.0).set_params(*model.get_params())
gap = np.abs(physical.predict(X_test) - model.predict(X_test)).max()

print(f"quantum parameters: {len(model.quantum_parameters)}, head parameters: {len(model.head)}")
print(f"train loss: {model.history.losses[0]:.4f} -> {model.history.losses[-1]:.6f}")
print(f"test MSE: CV model {test_mse:.6f}, OLS {ols_mse:.6f}")
print(f"max |ideal - compiled (r=1)| prediction difference: {gap:.2e}")
print(f"runtime: {perf_counter() - start:.1f} s")

assert test_mse <= 2 * ols_mse + 1e-4
assert gap < 1e-10
