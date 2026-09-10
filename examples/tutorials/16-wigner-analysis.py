"""Wigner-function analysis. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import photographiq as pg

result = pg.simulate(
    pg.Pattern(inputs=(0,)), inputs={0: pg.FockInput.number(1)}, backend="piquasso-fock", cutoff=8
)
grid = result.state.wigner(np.linspace(-6, 6, 101), np.linspace(-6, 6, 101))
print("Captured mass:", grid.captured_mass)
print("Negative volume:", grid.negative_volume)
assert abs(grid.captured_mass - 1) < 1e-5
assert grid.negative_volume > 0.1
grid.plot()
plt.close("all")
