"""GKP Hadamard. See the matching documentation for physical limitations."""

import numpy as np

import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

p = pg.Pattern(inputs=(0,)).append(code.logical_gate(0, "H"))
r = pg.simulate(p, inputs={0: code.zero()}, backend="piquasso-fock", cutoff=code.cutoff)
fidelity = abs(np.vdot(code.plus().amplitudes, r.state.state_vector)) ** 2
print("H target fidelity:", fidelity)
print(code.diagnostics(r.state))
assert 0 < fidelity <= 1 + 1e-12
