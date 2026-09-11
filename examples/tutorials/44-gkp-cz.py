"""Two-mode logical CZ. See the matching documentation for physical limitations."""

import numpy as np

import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

v = np.array(code.plus().amplitudes)
data = {(i, j): v[i] * v[j] for i in range(code.cutoff) for j in range(code.cutoff - i)}
mass = sum(abs(a) ** 2 for a in data.values())
source = pg.FockSuperposition.from_mapping({k: a / np.sqrt(mass) for k, a in data.items()})
p = pg.Pattern(inputs=(0, 1)).append(code.logical_cz(0, 1))
r = pg.simulate(p, initial_state=source, backend="piquasso-fock", cutoff=code.cutoff)
print("total preparation mass:", mass)
probabilities = pg.multimode_readout(r.state, {0: "X", 1: "Z"})
print(probabilities)
assert abs(sum(probabilities["joint_probabilities"].values()) - 1) < 1e-9
