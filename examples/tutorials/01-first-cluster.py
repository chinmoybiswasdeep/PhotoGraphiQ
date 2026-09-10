"""Your first CV cluster state. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import photographiq as pg

graph = pg.CVGraph.line(3, squeezing=0.7)
pattern = pg.Pattern(graph)
result = pg.simulate(pattern, backend="gaussian", seed=7)
nullifiers = graph.nullifiers()
noise = nullifiers @ result.state.covariance @ nullifiers.T
print("Output modes:", result.state.nodes)
print("Nullifier covariance:", np.round(noise, 6))
np.testing.assert_allclose(noise, np.eye(3) * np.exp(-1.4), atol=1e-12)
pattern.draw()
plt.close("all")
