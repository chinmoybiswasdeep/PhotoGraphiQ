"""CV quantum teleportation. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import photographiq as pg

pattern = pg.protocols.teleportation(squeezing=0.8)
channel = pg.gaussian_channel(pattern)
initial = pg.GaussianInput.coherent(0.3 + 0.1j).state(0)
output = channel.apply(initial)
print("Unconditional output mean:", np.round(output.mean, 6))
print("Added noise:", np.round(channel.noise, 6))
np.testing.assert_allclose(channel.noise, 2 * np.exp(-1.6) * np.eye(2), atol=1e-12)
print(
    "Conditional outcomes:",
    pg.simulate(pattern, inputs={0: pg.GaussianInput.coherent(0.3 + 0.1j)}, seed=7).outcomes,
)
pattern.draw()
plt.close("all")
