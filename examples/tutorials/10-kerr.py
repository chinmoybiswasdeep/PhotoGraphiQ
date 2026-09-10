"""Kerr nonlinearity. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import photographiq as pg

initial = pg.FockInput((2**-0.5, 2**-0.5))
pattern = pg.Pattern(inputs=(0,)).append(pg.Kerr(0, 0.3))
result = pg.simulate(pattern, inputs={0: initial}, backend="piquasso-fock", cutoff=8)
print("Photon number:", result.state.photon_number(0))
print("q moments:", result.state.quadrature(0))
assert np.isclose(result.state.photon_number(0), 0.5)
assert np.isclose(result.state.quadrature(0)[0], np.cos(0.3))
result.state.wigner(np.linspace(-4, 4, 51), np.linspace(-4, 4, 51)).plot()
plt.close("all")
