"""Circuit to MBQC compilation and visualization. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import photographiq as pg

circuit = pg.Circuit(1).rotate(0, 0.3).squeeze(0, 0.15)
pattern, trace = circuit.compile(squeezing=0.8, return_trace=True)
for step in trace.steps:
    print(
        "Gate",
        step.gate_index,
        step.source_gate,
        "resources",
        step.resource_nodes,
        "measurements",
        step.measurements,
        "corrections",
        step.corrections,
    )
assert sum(len(step.measurements) for step in trace.steps) == sum(
    isinstance(c, pg.Measure) for c in pattern.commands
)
pg.visualize_compilation(circuit, pattern, trace=trace)
plt.close("all")
