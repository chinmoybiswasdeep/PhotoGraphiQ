"""Tutorial 8: physical resources of CV models and accuracy-resource trade-offs.

``model.resource_report(X)`` reports, separately from the learning objective, the resources
of the executable pattern behind a trained model: resource modes, entangling edges, homodyne
measurements, measurement depth, feedforward corrections, peak live modes, resource and gate
squeezing (r and dB), added noise, input photons, runtime and memory.

Part 1 compares a compiled two-mode layered program with a 2 x 4 CVCluster on one task.
Part 2 trades cluster size and resource squeezing against the shot-limited (S = 100)
expected test MSE. Part 3 shows the squeezing penalty (sum of sinh^2 r over squeeze gates,
the photons those gates add to vacuum) lowering the gate squeezing of a layered program.
"""

from time import perf_counter

import numpy as np

from photographiqml.cv import CVCluster, CVRegressor, layered_program

start = perf_counter()
rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, (80, 2))
y = 0.7 * X[:, 0] - 0.4 * X[:, 1] + 0.2 + 0.02 * rng.standard_normal(80)
X_train, y_train, X_test, y_test = X[:50], y[:50], X[50:], y[50:]

# Part 1: resource reports of two trained models (both executed physically at r = 1).
compiled = CVRegressor(layered_program(2, 1), mode="physical", squeezing=1.0, epochs=40)
cluster = CVRegressor(CVCluster(2, 4, squeezing=1.0), mode="physical", epochs=40)
reports = {
    name: m.fit(X_train, y_train).resource_report(X_test)
    for name, m in (("compiled layer", compiled), ("2x4 cluster", cluster))
}
keys = (
    "trainable_quantum_parameters",
    "resource_modes",
    "entangling_edges",
    "measurements",
    "measurement_depth",
    "feedforward_corrections",
    "peak_live_modes",
    "resource_squeezing_db",
    "added_noise_per_quadrature",
    "mean_input_photons",
)
print(f"{'':30} {'compiled layer':>15} {'2x4 cluster':>12}")
for k in keys:
    a, b = reports["compiled layer"][k], reports["2x4 cluster"][k]
    print(f"{k:30} {a:>15.4g} {b:>12.4g}")

# Part 2: cluster size x resource squeezing vs expected MSE of S = 100-shot predictions.
S = 100
print(f"\nexpected test MSE with S = {S} shots (ideal-trained clusters):")
print(f"{'columns':>8} {'measurements':>13} " + " ".join(f"r={r:<6}" for r in (0.5, 1.0, 1.5)))
table = {}
for c in (3, 4, 5):
    model = CVRegressor(CVCluster(2, c), epochs=60).fit(X_train, y_train)
    row = []
    for r in (0.5, 1.0, 1.5):
        physical = CVRegressor(CVCluster(2, c, squeezing=r), mode="physical")
        physical.set_params(*model.get_params())
        mu, cov = physical.moments(physical.params, X_test)
        w = physical.head[:-1]
        bias = (mu[:, 0::2] @ w + physical.head[-1] - y_test) ** 2
        var = np.einsum("j,njk,k->n", w, cov[:, 0::2, 0::2], w) / S
        row.append(float(np.mean(bias + var)))
    table[c] = row
    print(f"{c:>8} {2 * (c - 1):>13} " + " ".join(f"{v:<8.5f}" for v in row))

# Part 3: the squeezing penalty.
photons = {}
for weight in (0.0, 0.5):
    kw = {"penalty": "squeezing", "penalty_weight": weight} if weight else {}
    m = CVRegressor(layered_program(2, 1), epochs=60, **kw).fit(X_train, y_train)
    r_gates = np.real(m.program.squeezing_values(m.params))
    photons[weight] = float(np.sum(np.sinh(r_gates) ** 2))
    print(
        f"\npenalty weight {weight}: gate r = {np.round(r_gates, 3)}, "
        f"sum sinh^2 r = {photons[weight]:.4f}, test MSE = {m.score(X_test, y_test)['mse']:.6f}"
    )
print(f"runtime: {perf_counter() - start:.1f} s")

assert reports["2x4 cluster"]["measurements"] == 2 * (4 - 1)
assert reports["compiled layer"]["resource_modes"] == reports["compiled layer"]["measurements"]
assert all(r["measurement_depth"] <= r["measurements"] for r in reports.values())
assert all(row[0] > row[1] > row[2] for row in table.values())  # more squeezing, lower error
assert photons[0.5] <= photons[0.0]
