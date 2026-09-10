"""Cubic-phase convergence study. Writes data.csv and figure.svg into --output."""

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
pattern = pg.Pattern(inputs=(0,)).append(pg.CubicPhase(0, 0.05))
study = pg.cutoff_convergence(pattern, [12, 18, 24, 32], high_order_moments=True)
for row in study.rows:
    rows.append((row["cutoff"], row["high_order_moments"][0]["p4"]))
data = np.asarray(rows)
np.savetxt(
    output / "data.csv", data, delimiter=",", header="Cutoff,Fourth momentum moment", comments=""
)
fig, ax = plt.subplots()
ax.plot(data[:, 0], data[:, 1], marker="o")
ax.set(xlabel="Cutoff", ylabel="Fourth momentum moment", title="Cubic-phase convergence study")
fig.savefig(output / "figure.svg", bbox_inches="tight")
plt.close("all")
print(data)
print("Wrote", output / "data.csv", "and", output / "figure.svg")
