"""Logical X and Z displacements. See the matching documentation for physical limitations."""

import numpy as np

import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

for gate in ("X", "Z"):
    source = code.zero() if gate == "X" else code.plus()
    target = code.one() if gate == "X" else code.minus()
    p = pg.Pattern(inputs=(0,)).append(code.logical_gate(0, gate))
    r = pg.simulate(p, inputs={0: source}, backend="piquasso-fock", cutoff=code.cutoff)
    fidelity = abs(np.vdot(target.amplitudes, r.state.state_vector)) ** 2
    print(gate, "target fidelity:", fidelity, "leakage:", code.leakage(r.state.state_vector))
    assert 0 < fidelity < 1
