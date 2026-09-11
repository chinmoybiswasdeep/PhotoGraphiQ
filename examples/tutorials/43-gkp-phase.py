"""GKP phase gate. See the matching documentation for physical limitations."""

import numpy as np

import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

p = pg.Pattern(inputs=(0,)).append(code.logical_gate(0, "S"))
r = pg.simulate(p, inputs={0: code.plus()}, backend="piquasso-fock", cutoff=code.cutoff)
target = code.encode(1, 1j)
fidelity = abs(np.vdot(target.amplitudes, r.state.state_vector)) ** 2
print("S target fidelity:", fidelity)
print(code.diagnostics(r.state))
assert 0 < fidelity < 1
