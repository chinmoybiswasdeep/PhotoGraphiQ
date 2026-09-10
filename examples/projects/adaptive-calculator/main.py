"""Adaptive MBQC calculator. Writes data.csv and figure.svg into --output."""

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
pattern = pg.protocols.adaptive(squeezing=0.7)
for k in (-0.3, 0.0, 0.3):
    shots = pg.run_shots(pattern, 32, seed=7, backend="gaussian", parameters={"k": k})
    values = [r.state.quadrature(pattern.outputs[0])[0] for r in shots.trajectories]
    rows.append((k, float(np.mean(values))))
data = np.asarray(rows)
np.savetxt(
    output / "data.csv",
    data,
    delimiter=",",
    header="Bound parameter k,Sample mean output q",
    comments="",
)
fig, ax = plt.subplots()
ax.plot(data[:, 0], data[:, 1], marker="o")
ax.set(xlabel="Bound parameter k", ylabel="Sample mean output q", title="Adaptive MBQC calculator")
fig.savefig(output / "figure.svg", bbox_inches="tight")
plt.close("all")
print(data)
print("Wrote", output / "data.csv", "and", output / "figure.svg")
