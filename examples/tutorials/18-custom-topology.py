"""Custom CV graph topology. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

import photographiq as pg

network = nx.Graph()
network.add_edge("source", "upper", weight=0.4)
network.add_edge("source", "lower", weight=-0.3)
graph = pg.CVGraph(network, squeezing=0.6, inputs=("source",))
pattern = pg.Pattern(graph).measure("source", pg.Homodyne.p())
result = pg.simulate(pattern, seed=7, backend="gaussian")
print("Surviving labels:", result.state.nodes)
print("Covariance shape:", result.state.covariance.shape)
assert set(result.state.nodes) == {"upper", "lower"}
pattern.draw(positions={"source": (0, 0), "upper": (1, 1), "lower": (1, -1)})
plt.close("all")
