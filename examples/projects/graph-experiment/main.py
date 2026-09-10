"""Small graph-state experiment. Writes data.csv and figure.svg into --output."""

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
for weight in (0.2, 0.4, 0.6, 0.8):
    pattern = pg.Pattern().extend(
        [pg.Prepare("a", 0.6), pg.Prepare("b", 0.6), pg.Entangle("a", "b", weight)]
    )
    result = pg.simulate(pattern, backend="gaussian")
    rows.append((weight, float(result.state.covariance[0, 3])))
pattern.draw(output=output / "resource.svg")
data = np.asarray(rows)
np.savetxt(
    output / "data.csv", data, delimiter=",", header="CZ weight,q_a,p_b covariance", comments=""
)
fig, ax = plt.subplots()
ax.plot(data[:, 0], data[:, 1], marker="o")
ax.set(xlabel="CZ weight", ylabel="q_a,p_b covariance", title="Small graph-state experiment")
fig.savefig(output / "figure.svg", bbox_inches="tight")
plt.close("all")
print(data)
print("Wrote", output / "data.csv", "and", output / "figure.svg")
