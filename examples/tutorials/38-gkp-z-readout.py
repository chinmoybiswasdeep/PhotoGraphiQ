"""Logical Z from q homodyne. See the matching documentation for physical limitations."""

import numpy as np

import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

p = pg.Pattern(inputs=(0,)).measure(0, code.logical_measurement("Z"), key="z")
r = pg.simulate(
    p,
    inputs={0: code.zero()},
    backend="piquasso-fock",
    cutoff=code.cutoff,
    measurement_outcomes={"z": 0.25},
)
print(r.outcomes["z"])
print(r.measurement_statistics)
assert r.outcomes["z"].bit == 0
assert r.outcomes["z"].raw_outcome == 0.25
v = np.array(code.zero().amplitudes)
print(
    "integrated Z probabilities:",
    np.einsum("i,kij,j->k", v.conj(), pg.modular_effects(code.cutoff), v).real,
)
