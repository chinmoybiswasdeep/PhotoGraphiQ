"""Finite squeezing and noise. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import photographiq as pg

values = []
for r in (0.4, 0.8, 1.2):
    channel = pg.gaussian_channel(pg.protocols.identity(squeezing=r))
    values.append(float(np.trace(channel.noise)))
    print(f"r={r:.1f}, noise trace={values[-1]:.6f}")
assert values[0] > values[1] > values[2]
plt.plot([0.4, 0.8, 1.2], values, marker="o")
plt.xlabel("Resource squeezing r")
plt.ylabel("Added covariance trace")
plt.close("all")
