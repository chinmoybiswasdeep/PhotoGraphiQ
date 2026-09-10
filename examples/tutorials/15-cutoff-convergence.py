"""Cutoff convergence. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import photographiq as pg

pattern = pg.Pattern(inputs=(0,)).append(pg.CubicPhase(0, 0.04))
study = pg.cutoff_convergence(pattern, [12, 18, 24], high_order_moments=True)
for row in study.rows:
    print(row["cutoff"], row["fidelity_to_previous"], row["high_order_moments"][0]["p4"])
assert study.rows[-1]["fidelity_to_previous"] > 0.9999
plt.plot(
    [r["cutoff"] for r in study.rows],
    [r["high_order_moments"][0]["p4"] for r in study.rows],
    marker="o",
)
plt.xlabel("Total-photon cutoff")
plt.ylabel("Raw fourth momentum moment")
plt.close("all")
