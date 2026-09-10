"""Automatic differentiation. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import jax
import matplotlib.pyplot as plt
import numpy as np

import photographiq as pg

jax.config.update("jax_enable_x64", True)
from photographiq import autodiff  # noqa: E402

pattern = pg.Pattern(inputs=(0,)).append(pg.Rotate(0, pg.Parameter("theta")))
initial = pg.FockInput((2**-0.5, 2**-0.5))


def objective(theta):
    return autodiff.expectation(
        pattern, {"theta": theta}, cutoff=6, inputs={0: initial}, observable="q"
    )


gradient = float(jax.grad(objective)(0.3))
print("Gradient:", gradient)
print("Analytical derivative:", -np.sin(0.3))
assert abs(gradient + np.sin(0.3)) < 1e-10
plt.plot([0.0, 0.3, 0.6], [float(objective(t)) for t in [0.0, 0.3, 0.6]], marker="o")
plt.xlabel("Rotation angle")
plt.ylabel("Conditional q mean")
plt.close("all")
