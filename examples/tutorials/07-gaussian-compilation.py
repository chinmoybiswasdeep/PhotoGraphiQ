"""Gaussian gate compilation. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import photographiq as pg

circuit = pg.Circuit(1).rotate(0, 0.3).squeeze(0, 0.2)
pattern, trace = circuit.compile(squeezing=0.8, return_trace=True)
channel = pg.gaussian_channel(pattern)
print("Linear map:", np.round(channel.matrix, 6))
print("Added noise:", np.round(channel.noise, 6))
print("Generated measurements:", [step.measurements for step in trace.steps])
assert len(trace.steps) == 2
pg.visualize_compilation(circuit, pattern, trace=trace)
plt.close("all")
