"""Building a reusable Pattern. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import photographiq as pg

pattern = pg.protocols.wire([pg.Parameter("k")], squeezing=0.7)
restored = pg.Pattern.from_json(pattern.to_json())
for k in (-0.2, 0.0, 0.2):
    result = pg.simulate(restored, parameters={"k": k}, seed=7, backend="gaussian")
    print("k=", k, "output q=", result.state.quadrature(restored.outputs[0]))
assert restored.to_json() == pattern.to_json()
restored.draw()
plt.close("all")
