"""Teleportation fidelity versus squeezing. Writes data.csv and figure.svg into --output."""

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
for r in np.linspace(0.2, 1.4, 9):
    channel = pg.gaussian_channel(pg.protocols.teleportation(squeezing=float(r)))
    initial = pg.GaussianInput.coherent(0.3).state(0)
    state = channel.apply(initial)
    target = pg.GaussianInput.coherent(0.3).state(state.nodes[0])
    rows.append((r, state.overlap(target)))
data = np.asarray(rows)
np.savetxt(
    output / "data.csv",
    data,
    delimiter=",",
    header="Resource squeezing,Coherent target fidelity",
    comments="",
)
fig, ax = plt.subplots()
ax.plot(data[:, 0], data[:, 1], marker="o")
ax.set(
    xlabel="Resource squeezing",
    ylabel="Coherent target fidelity",
    title="Teleportation fidelity versus squeezing",
)
fig.savefig(output / "figure.svg", bbox_inches="tight")
plt.close("all")
print(data)
print("Wrote", output / "data.csv", "and", output / "figure.svg")
