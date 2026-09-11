"""The pi/4 measurement boundary. See the matching documentation for physical limitations."""

import numpy as np

import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

alpha = np.pi / 4
print("ideal qubit probabilities:", pg.IdealLogicalXYMeasurement(alpha).probabilities([1, 0]))
try:
    code.logical_measurement("XY", alpha=alpha)
except NotImplementedError as error:
    print("physical request refused:", error)
else:
    raise AssertionError("Unvalidated physical pi/4 readout must be refused")
