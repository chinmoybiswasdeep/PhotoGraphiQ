"""Compiling a nonlinear resource injection. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import photographiq as pg

circuit = pg.Circuit(1).cubic_phase(0, 0.02)
pattern, trace = circuit.compile(squeezing=0.2, return_trace=True)
result = pg.simulate(
    pattern, backend="piquasso-fock", cutoff=24, measurement_outcomes={("cubic", 1): 0.1}
)
print("Generated commands:", len(pattern.commands))
print("Branch density:", np.exp(result.log_likelihood))
assert len(trace.steps[0].measurements) == 1
pg.visualize_compilation(circuit, pattern, trace=trace)
plt.close("all")
