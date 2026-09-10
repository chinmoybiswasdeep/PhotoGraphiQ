"""Classical feed-forward. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import photographiq as pg

pattern = pg.Pattern(pg.CVGraph.line(2, squeezing=0.7, inputs=(0,)))
pattern.measure(0, pg.Homodyne.p(), key="m")
pattern.append(pg.Signal("correction", -pg.Outcome("m")))
pattern.displace(1, q=pg.Outcome("correction"))
result = pg.simulate(pattern, seed=7, backend="gaussian")
print("Measurements:", result.outcomes)
print("All records:", result.records)
print("Executed displacements:", result.physical_displacements)
assert result.records["correction"] == -result.outcomes["m"]
pattern.draw()
plt.close("all")
