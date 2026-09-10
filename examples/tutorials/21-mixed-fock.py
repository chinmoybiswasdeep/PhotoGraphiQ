"""Mixed Fock evolution. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import photographiq as pg

pattern = pg.Pattern(inputs=(0,)).append(pg.Kerr(0, 0.2)).append(pg.Loss(0, 0.6))
result = pg.simulate(
    pattern, inputs={0: pg.FockInput.number(2)}, backend="piquasso-mixed-fock", cutoff=6
)
rho = result.state.density_matrix
print("Photon probabilities:", result.state.probabilities)
print("Purity:", np.trace(rho @ rho).real)
assert np.isclose(result.state.photon_number(0), 1.2)
assert np.linalg.eigvalsh(rho).min() > -1e-12
result.state.wigner(np.linspace(-5, 5, 61), np.linspace(-5, 5, 61)).plot()
plt.close("all")
