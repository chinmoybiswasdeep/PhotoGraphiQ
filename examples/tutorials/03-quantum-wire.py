"""One-dimensional quantum wire. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import photographiq as pg

one = pg.protocols.wire([0.0], squeezing=0.8)
four = pg.protocols.identity(squeezing=0.8)
print("One-step map:", np.round(pg.gaussian_channel(one).matrix, 6))
channel = pg.gaussian_channel(four)
print("Four-step map:", np.round(channel.matrix, 6))
print("Accumulated noise:", np.round(channel.noise, 6))
np.testing.assert_allclose(channel.matrix, np.eye(2), atol=1e-12)
four.draw()
plt.close("all")
