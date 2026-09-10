"""Gaussian circuit compiler. Writes data.csv and figure.svg into --output."""

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
for angle in (0.1, 0.2, 0.3, 0.4):
    circuit = pg.Circuit(1).rotate(0, angle).squeeze(0, 0.1)
    pattern = circuit.compile(squeezing=0.8)
    rows.append((angle, float(np.trace(pg.gaussian_channel(pattern).noise))))
pg.visualize_compilation(circuit, pattern, output=output / "compilation.svg")
data = np.asarray(rows)
np.savetxt(
    output / "data.csv",
    data,
    delimiter=",",
    header="Rotation angle,Added covariance trace",
    comments="",
)
fig, ax = plt.subplots()
ax.plot(data[:, 0], data[:, 1], marker="o")
ax.set(xlabel="Rotation angle", ylabel="Added covariance trace", title="Gaussian circuit compiler")
fig.savefig(output / "figure.svg", bbox_inches="tight")
plt.close("all")
print(data)
print("Wrote", output / "data.csv", "and", output / "figure.svg")
