"""Cubic resource injection. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import photographiq as pg

pattern = pg.non_gaussian.cubic_injection(gamma=0.02, squeezing=0.2)
result = pg.simulate(pattern, backend="piquasso-fock", cutoff=24, measurement_outcomes={"m": 0.1})
print("Homodyne density:", np.exp(result.log_likelihood))
print("Conditional p moments:", result.state.quadrature("in", np.pi / 2))
assert result.measurement_statistics["m"]["kind"] == "density"
pattern.draw()
plt.close("all")
