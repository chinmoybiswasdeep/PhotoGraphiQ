"""Logical X from p homodyne. See the matching documentation for physical limitations."""

import numpy as np

import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

p = pg.Pattern(inputs=(0,)).measure(0, code.logical_measurement("X"), key="x")
r = pg.simulate(
    p,
    inputs={0: code.plus()},
    backend="piquasso-fock",
    cutoff=code.cutoff,
    measurement_outcomes={"x": -0.2},
)
print(r.outcomes["x"])
assert r.outcomes["x"].bit == 0
for state in (code.plus(), code.minus()):
    v = np.array(state.amplitudes)
    print(np.einsum("i,kij,j->k", v.conj(), pg.modular_effects(code.cutoff, "X"), v).real)
