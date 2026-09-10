"""Cubic-phase gate. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import photographiq as pg

pattern = pg.Pattern(inputs=(0,)).append(pg.CubicPhase(0, 0.03))
result = pg.simulate(pattern, backend="piquasso-fock", cutoff=24)
mean, variance = result.state.quadrature(0, np.pi / 2)
print("Momentum mean and variance:", mean, variance)
print("Minimum retained norm:", min(result.state.retained_norms))
assert abs(mean - 0.03) < 1e-5
result.state.wigner(np.linspace(-4, 4, 51), np.linspace(-4, 4, 51)).plot()
plt.close("all")
