"""Heralded photon subtraction. Writes data.csv and figure.svg into --output."""

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
for theta in np.linspace(0.05, 0.3, 6):
    pattern = pg.non_gaussian.photon_subtraction(theta=float(theta))
    result = pg.simulate(
        pattern,
        inputs={"in": pg.FockInput.number(2)},
        backend="piquasso-fock",
        cutoff=8,
        measurement_outcomes={"count": 1},
    )
    rows.append((theta, float(np.exp(result.log_likelihood))))
data = np.asarray(rows)
np.savetxt(
    output / "data.csv", data, delimiter=",", header="Tap angle,Herald probability", comments=""
)
fig, ax = plt.subplots()
ax.plot(data[:, 0], data[:, 1], marker="o")
ax.set(xlabel="Tap angle", ylabel="Herald probability", title="Heralded photon subtraction")
fig.savefig(output / "figure.svg", bbox_inches="tight")
plt.close("all")
print(data)
print("Wrote", output / "data.csv", "and", output / "figure.svg")
