"""Finite-energy GKP resources. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import photographiq as pg

resource = pg.GKPResource(0, peak_width=0.5, envelope=0.5, peaks=4, grid_points=2049)
initial, mass = resource.project(48)
result = pg.simulate(
    pg.Pattern(inputs=(0,)), inputs={0: initial}, backend="piquasso-fock", cutoff=48
)
print("Captured Fock weight:", mass)
print("Stabilizer expectations:", pg.gkp.stabilizers(result.state))
print("Decoded shifted cell:", pg.gkp.decode_shift(pg.gkp.SPACING + 0.1))
assert mass > 0.99
q, psi = resource.wavefunction()
plt.plot(q, abs(psi) ** 2)
plt.xlim(-10, 10)
plt.xlabel("q")
plt.ylabel("Probability density")
plt.close("all")
