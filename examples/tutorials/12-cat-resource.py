"""Cat-state resource. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import photographiq as pg

pattern = pg.Pattern(inputs=(0,))
result = pg.simulate(
    pattern, inputs={0: pg.CatResource(0.8, 1)}, backend="piquasso-fock", cutoff=20
)
print("Parity:", result.state.parity())
print("Photon number:", result.state.photon_number(0))
assert np.isclose(result.state.parity(), 1.0)
result.state.wigner(np.linspace(-4, 4, 61), np.linspace(-4, 4, 61)).plot()
plt.close("all")
