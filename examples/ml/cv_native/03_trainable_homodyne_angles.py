"""Tutorial 3: training homodyne measurement angles on a weighted CV cluster (CVCluster).

CVCluster is a directly trainable weighted grid cluster; it is not the CVMuTA proposal on the
roadmap, which remains planned. Here a 2 x 4 grid: column 0 holds the encoded inputs,
column 3 the outputs; the four nodes in between and the two outputs are p-squeezed
resource modes (r = 1). Each measured node i is read out by
the homodyne q cos(theta_i) + p sin(theta_i); CV-flow feedforward (Booth-Markham) makes the
unconditional channel deterministic. The angles theta_i are the trainable parameters; the
CZ rung weights are trainable too here, and the flow certificate is recomputed per weight
tuple. Teleportation shear k_i = cot(theta_i): theta is pi-periodic and 0 mod pi is singular.
"""

from time import perf_counter

import numpy as np

from photographiq.analysis import gaussian_channel
from photographiqml.cv import CVCluster, CVRegressor

start = perf_counter()
cluster = CVCluster(2, 4, squeezing=1.0, trainable_weights=True)
print("parameters:", cluster.parameters)

# Bounded parameterisation: theta = pi/2 + (pi/2 - margin) tanh(phi).
thetas = [0.4, 1.3, 2.0, 2.7, 1.1, 2.2]
params = cluster.angle_parameters(thetas) | dict.fromkeys(cluster.weight_names, 0.8)
assert np.allclose(cluster.angles(params), thetas)

# Exact channel by forward substitution vs the executable flow pattern.
X, t, Y, measurements = cluster.channel(params, mode="physical")
reference = gaussian_channel(cluster.pattern(params))
err_X = np.abs(X - reference.matrix).max()
err_Y = np.abs(Y - reference.noise).max()
print(f"measurements: {measurements}; |X - X_pattern| = {err_X:.1e}, |Y - Y_pattern| = {err_Y:.1e}")

# Direct angles: pi-periodicity and rejection of the singular q measurement.
direct = CVCluster(2, 4, squeezing=1.0, angles="direct")
base = dict(zip(direct.parameters, thetas))
shifted = {k: v + np.pi for k, v in base.items()}
period_err = np.abs(direct.channel(base)[0] - direct.channel(shifted)[0]).max()
try:
    direct.channel(base | {direct.parameters[0]: np.pi})
    raise AssertionError("singular angle accepted")
except ValueError as error:
    print("theta = pi rejected:", error)
print(f"pi-periodicity error: {period_err:.1e}")

# Train angles and rung weights on an affine target. Rung weights enter the channel through
# their real part only, so complex-step derivatives with respect to them are zero: use
# central differences whenever trainable_weights=True.
rng = np.random.default_rng(1)
Xd = rng.uniform(-1, 1, (40, 2))
yd = 0.7 * Xd[:, 0] - 0.4 * Xd[:, 1] + 0.2
model = CVRegressor(
    cluster, mode="physical", optimizer="lbfgs", epochs=30, seed=0, gradient="finite-difference"
).fit(Xd, yd)
names = model.history.settings["parameter_names"]
initial = [model.history.parameters[0][names.index(n)] for n in cluster.weight_names]
final = [float(model.params[n]) for n in cluster.weight_names]
print("trained angles (rad, mod pi):", np.round(np.mod(cluster.angles(model.params), np.pi), 3))
print("rung weights: initial", np.round(initial, 3), "-> trained", np.round(final, 3))
print(f"certificates cached: {len(cluster._flow_cache)}")
print(f"train MSE: {model.score(Xd, yd)['mse']:.2e}")
print(f"runtime: {perf_counter() - start:.1f} s")

assert err_X < 1e-10 and err_Y < 1e-10 and measurements == 6
assert period_err < 1e-10
assert model.score(Xd, yd)["mse"] < 1e-4
assert not np.allclose(initial, final)
