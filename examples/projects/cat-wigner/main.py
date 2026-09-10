"""Cat states and Wigner negativity. Writes data.csv and figure.svg into --output."""

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
for alpha in (0.3, 0.6, 0.9, 1.2):
    result = pg.simulate(
        pg.Pattern(inputs=(0,)),
        inputs={0: pg.CatResource(alpha)},
        backend="piquasso-fock",
        cutoff=24,
    )
    grid = result.state.wigner(np.linspace(-6, 6, 81), np.linspace(-6, 6, 81))
    rows.append((alpha, grid.negative_volume))
grid.plot()[0].figure.savefig(output / "wigner.svg", bbox_inches="tight")
data = np.asarray(rows)
np.savetxt(
    output / "data.csv",
    data,
    delimiter=",",
    header="Cat amplitude,Finite-grid negative volume",
    comments="",
)
fig, ax = plt.subplots()
ax.plot(data[:, 0], data[:, 1], marker="o")
ax.set(
    xlabel="Cat amplitude",
    ylabel="Finite-grid negative volume",
    title="Cat states and Wigner negativity",
)
fig.savefig(output / "figure.svg", bbox_inches="tight")
plt.close("all")
print(data)
print("Wrote", output / "data.csv", "and", output / "figure.svg")
