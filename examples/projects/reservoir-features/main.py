"""Introductory reservoir-style feature extraction. Writes data.csv and figure.svg into --output."""

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
pattern = pg.protocols.wire([0.0, 0.2], squeezing=0.7)
channel = pg.gaussian_channel(pattern)
for amplitude in np.linspace(-0.5, 0.5, 9):
    state = channel.apply(pg.GaussianInput.coherent(amplitude).state(0))
    rows.append((amplitude, state.photon_number(state.nodes[0])))
data = np.asarray(rows)
np.savetxt(
    output / "data.csv",
    data,
    delimiter=",",
    header="Input coherent amplitude,Output photon-number feature",
    comments="",
)
fig, ax = plt.subplots()
ax.plot(data[:, 0], data[:, 1], marker="o")
ax.set(
    xlabel="Input coherent amplitude",
    ylabel="Output photon-number feature",
    title="Introductory reservoir-style feature extraction",
)
fig.savefig(output / "figure.svg", bbox_inches="tight")
plt.close("all")
print(data)
print("Wrote", output / "data.csv", "and", output / "figure.svg")
