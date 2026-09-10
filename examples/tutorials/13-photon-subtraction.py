"""Heralded photon subtraction. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import photographiq as pg

theta = 0.2
pattern = pg.non_gaussian.photon_subtraction(theta=theta)
result = pg.simulate(
    pattern,
    inputs={"in": pg.FockInput.number(2)},
    backend="piquasso-fock",
    cutoff=6,
    measurement_outcomes={"count": 1},
)
probability = np.exp(result.log_likelihood)
print("Herald probability:", probability)
print("Conditional photon number:", result.state.photon_number("in"))
assert np.isclose(probability, 2 * np.sin(theta) ** 2 * np.cos(theta) ** 2)
pattern.draw()
plt.close("all")
