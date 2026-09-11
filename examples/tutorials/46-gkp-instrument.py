"""Encoded finite measurement instrument. See the matching documentation for physical limitations."""

import numpy as np

import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

instrument = code.discrimination_instrument()
probabilities = instrument.probabilities(code.zero().amplitudes)
print(probabilities)
assert abs(probabilities[1]) < 1e-10
assert probabilities["inconclusive"] > 0
assert abs(probabilities["outside-code"]) < 1e-10
p, rho = instrument.conditional(code.zero().amplitudes, 0)
print("branch probability:", p, "trace:", np.trace(rho))
assert abs(np.trace(rho) - 1) < 1e-10
