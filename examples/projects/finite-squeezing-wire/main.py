"""Finite-squeezing quantum wire. Writes data.csv and figure.svg into --output."""

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
for length in (4, 8, 12, 16):
    channel = pg.gaussian_channel(pg.protocols.wire([0.0] * length, squeezing=0.7))
    rows.append((length, float(np.trace(channel.noise))))
data = np.asarray(rows)
np.savetxt(
    output / "data.csv",
    data,
    delimiter=",",
    header="Wire steps,Added covariance trace",
    comments="",
)
fig, ax = plt.subplots()
ax.plot(data[:, 0], data[:, 1], marker="o")
ax.set(xlabel="Wire steps", ylabel="Added covariance trace", title="Finite-squeezing quantum wire")
fig.savefig(output / "figure.svg", bbox_inches="tight")
plt.close("all")
print(data)
print("Wrote", output / "data.csv", "and", output / "figure.svg")
