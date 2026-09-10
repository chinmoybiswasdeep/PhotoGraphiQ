"""Graph topology comparison. Writes data.csv and figure.svg into --output."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import photographiq as pg

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("outputs"))
output = parser.parse_args().output
output.mkdir(parents=True, exist_ok=True)
rows = []
for index, graph in enumerate(
    (
        pg.CVGraph.line(4, squeezing=0.6),
        pg.CVGraph.ring(4, squeezing=0.6),
        pg.CVGraph.star(3, squeezing=0.6),
    )
):
    result = pg.simulate(pg.Pattern(graph), backend="gaussian")
    rows.append((index, sum(result.state.photon_number(n) for n in result.state.nodes)))
data = np.asarray(rows)
np.savetxt(
    output / "data.csv",
    data,
    delimiter=",",
    header="topology_index,total_mean_photons",
    comments="",
)
fig, ax = plt.subplots()
ax.plot(data[:, 0], data[:, 1], marker="o")
ax.set(
    xlabel="Topology index: line, ring, star",
    ylabel="Total mean photons",
    title="Graph topology comparison",
)
fig.savefig(output / "figure.svg", bbox_inches="tight")
plt.close("all")
print(data)
print("Wrote", output / "data.csv", "and", output / "figure.svg")
