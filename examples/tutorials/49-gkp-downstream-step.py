"""Preparing a downstream physical measurement step. See the matching documentation for physical limitations."""

import numpy as np

import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

measurement = code.logical_measurement("XY", alpha=np.pi)
p = pg.Pattern(inputs=(0,)).measure(0, measurement, key="logical")
r = pg.simulate(
    p,
    inputs={0: code.plus()},
    backend="piquasso-fock",
    cutoff=code.cutoff,
    measurement_outcomes={"logical": 0.1},
)
print(r.outcomes["logical"])
assert r.outcomes["logical"].bit == 1
print("One signed-X step only. Full physical MuTA remains unsupported.")
