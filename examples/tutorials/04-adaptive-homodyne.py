"""Adaptive homodyne measurement. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import photographiq as pg

pattern = pg.protocols.adaptive(squeezing=0.7)
result = pg.simulate(pattern, parameters={"k": 0.2}, seed=7, backend="gaussian")
print("Outcomes:", result.outcomes)
print("Records:", result.records)
print("Output q moments:", result.state.quadrature(pattern.outputs[0]))
assert len(result.outcomes) == 2
from photographiq.visualization import draw_dependencies  # noqa: E402

draw_dependencies(pattern)
plt.close("all")
