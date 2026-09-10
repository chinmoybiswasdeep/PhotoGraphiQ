"""Two-mode entangling computation. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import photographiq as pg

circuit = pg.Circuit(2).cz(0, 1, 0.4)
pattern = circuit.compile(squeezing=0.8)
channel = pg.gaussian_channel(pattern)
print("Logical CZ map:", np.round(channel.matrix, 6))
state = channel.apply(pg.GaussianState(np.zeros(4), np.eye(4), (0, 1)))
print("Output labels:", state.nodes)
print("Cross covariance:", np.round(state.covariance[:2, 2:], 6))
assert abs(state.covariance[0, 3]) > 0
pg.visualize_compilation(circuit, pattern)
plt.close("all")
