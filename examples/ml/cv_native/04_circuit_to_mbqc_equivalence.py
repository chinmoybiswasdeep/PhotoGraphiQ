"""Tutorial 4: a trainable Gaussian circuit and its compiled CV-MBQC channel.

``GaussianProgram.channel(mode="ideal")`` is the direct optical circuit: d -> S d + t.
``mode="physical"`` is the unconditional channel d -> X d + t, V -> X V X^T + Y of the
finite-squeezing pattern that ``photographiq.compile_circuit`` builds for the same gates.
Checks: X equals S, (X, t, Y) equal photographiq.analysis.gaussian_channel of the compiled
Pattern, Y scales as exp(-2 r), the gate-level Pattern reproduces the ideal moments on the
Gaussian backend, and trajectory execution of the compiled Pattern averages to the channel.
"""

from time import perf_counter

import numpy as np

import photographiq as pg
from photographiq.analysis import gaussian_channel
from photographiqml.cv import CVFeatureMap, Feature, layered_program

start = perf_counter()
rng = np.random.default_rng(7)
program = layered_program(2, 1)
program.cz(0, 1, "g").rotate(1, Feature(0))  # one data-dependent gate
params = {n: rng.normal(scale=0.6) for n in program.parameters}
x = np.array([0.3, 0.8])

S, t_ideal, _, _ = program.channel(params, x)
r = 1.0
X, t, Y, steps = program.channel(params, x, mode="physical", squeezing=r)
pattern = program.circuit(params, x).compile(squeezing=r)
ref = gaussian_channel(pattern)
errors = {
    "|X - S|": np.abs(X - S).max(),
    "|X - X_pattern|": np.abs(X - ref.matrix).max(),
    "|t - t_pattern|": np.abs(t - ref.displacement).max(),
    "|Y - Y_pattern|": np.abs(Y - ref.noise).max(),
}
measurements = sum(isinstance(c, pg.Measure) for c in pattern.commands)
print(f"teleportation steps: {steps}, pattern measurements: {measurements}")
for name, value in errors.items():
    print(f"{name:>16}: {value:.1e}")

scaled = [
    program.channel(params, x, mode="physical", squeezing=s)[2] * np.exp(2 * s)
    for s in (0.5, 1.0, 2.0)
]
scaling_error = max(np.abs(scaled[0] - s).max() for s in scaled[1:])
print(f"Y exp(2r) constant over r in (0.5, 1, 2): max deviation {scaling_error:.1e}")
print(f"added noise per quadrature tr(Y)/4 at r = 1: {np.trace(Y) / 4:.3f} (vacuum units)")

fmap = CVFeatureMap(2, scale=0.8)
d, V = fmap.moments(x)
inputs = {
    j: pg.GaussianInput(
        tuple(d[2 * j : 2 * j + 2]), tuple(map(tuple, V[2 * j : 2 * j + 2, 2 * j : 2 * j + 2]))
    )
    for j in range(2)
}
state = pg.simulate(program.gate_pattern(params, x), backend="gaussian", inputs=inputs).state
gate_err = np.abs(state.mean - (S @ d + t_ideal)).max()
print(f"gate-level Pattern vs ideal mean: {gate_err:.1e}")

# Trajectories of a small compiled program: ensemble mean -> channel mean.
small = layered_program(1, 1)
p1 = {n: 0.3 for n in small.parameters}
X1, t1, Y1, _ = small.channel(p1, mode="physical", squeezing=0.8)
compiled = small.circuit(p1).compile(squeezing=0.8)
d1, shots = np.array([0.5, -0.2]), 300
runs = pg.run_shots(compiled, shots, seed=11, inputs={0: pg.GaussianInput(tuple(d1))}).trajectories
means = np.array([run.state.mean for run in runs])
cov = X1 @ X1.T + Y1
z = np.abs(means.mean(0) - (X1 @ d1 + t1)) / np.sqrt(np.diag(cov) / shots)
print(f"trajectory ensemble mean vs channel ({shots} shots): max |z| = {z.max():.2f}")
print(f"runtime: {perf_counter() - start:.1f} s")

assert all(v < 1e-9 for v in errors.values()) and steps == measurements
assert scaling_error < 1e-9 and gate_err < 1e-12 and z.max() < 4
