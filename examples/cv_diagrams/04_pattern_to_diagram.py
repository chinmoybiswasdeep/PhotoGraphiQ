"""Tutorial 4: executable Patterns and Gaussian graphs as typed diagrams.

``from_pattern`` reads an executable CV-MBQC Pattern into a diagram (feed-forward must be
exactly affine in outcomes; adaptive angles are rejected, never approximated). ``to_pattern``
goes back. ``from_gaussian_graph`` / ``to_gaussian_graph`` connect to the pure-state
Gaussian graphical calculus (MFvL 2011) without merging the two.

Key claims checked: the diagram of a compiled circuit has the same averaged channel as
``analysis.gaussian_channel`` of the pattern and its round trip; an adaptive pattern is
rejected; a cluster built from generators has the GaussianGraph.cluster adjacency matrix;
conditioning the diagram on a q-homodyne record reproduces the MFvL q-measurement rule.
"""

from time import perf_counter

import numpy as np

import photographiq as pg
from photographiq.analysis import gaussian_channel
from photographiq.calculus import GaussianGraph
from photographiq.calculus.cv_diagrams import (
    CZ,
    CVDiagram,
    Homodyne,
    Prepare,
    channel_semantics,
    from_circuit,
    from_gaussian_graph,
    from_pattern,
    resource_cost,
    teleport_steps,
    to_gaussian_graph,
    to_pattern,
    unitary_semantics,
)

start = perf_counter()

# 1. Circuit -> compiled Pattern -> diagram -> Pattern.
circuit = pg.Circuit(2).rotate(0, 0.3).squeeze(1, 0.2).cz(0, 1, 0.5).displace(0, 0.3, 0.1)
pattern = circuit.compile(squeezing=1.0)
diagram = from_pattern(pattern)
cost = resource_cost(diagram)
print(diagram)
print(
    {
        k: cost[k]
        for k in (
            "resource_modes",
            "measurement_count",
            "measurement_depth",
            "feedforward_commands",
            "entangling_edges",
        )
    }
)
print("teleportation steps found:", len(teleport_steps(diagram)))
X, Y, d = channel_semantics(diagram)
ref, back = gaussian_channel(pattern), gaussian_channel(to_pattern(diagram))
pattern_gap = max(
    np.abs(X - ref.matrix).max(),
    np.abs(Y - ref.noise).max(),
    np.abs(d - ref.displacement).max(),
    np.abs(back.matrix - ref.matrix).max(),
    np.abs(back.noise - ref.noise).max(),
)
print(f"max |diagram - gaussian_channel| incl. round trip: {pattern_gap:.1e}")
S, _ = unitary_semantics(from_circuit(circuit))
print(f"ideal circuit vs compiled X: {np.abs(S - X).max():.1e} (flow-corrected: X is ideal)")

# 2. Unsupported input: an adaptive homodyne angle.
adaptive = pg.Pattern(inputs=(0,)).extend([pg.Prepare(1, 0.5), pg.Entangle(0, 1)])
adaptive.measure(0, pg.Homodyne(0.4))
adaptive.measure(1, pg.Homodyne(pg.Outcome(0)))
adaptive.append(pg.Output(()))
try:
    from_pattern(adaptive.validate())
    rejected = False
except NotImplementedError as exc:
    rejected = True
    print("adaptive pattern rejected:", exc)

# 3. GaussianGraph interop: a 3-node weighted cluster built from generators.
r = 0.7
graph = GaussianGraph.cluster([[0, 1.0, 0], [1.0, 0, 0.5], [0, 0.5, 0]], r)
built = CVDiagram(
    (),
    tuple(Prepare.squeezed(j, r) for j in range(3)) + (CZ(0, 1, 1.0), CZ(1, 2, 0.5)),
    (0, 1, 2),
)
graph_gap = np.abs(to_gaussian_graph(built).Z - graph.Z).max()
round_gap = np.abs(to_gaussian_graph(from_gaussian_graph(graph)).Z - graph.Z).max()
print("Z of built cluster:", np.round(to_gaussian_graph(built).Z, 4).tolist())
print(f"|Z(diagram) - GaussianGraph.cluster| = {graph_gap:.1e}; round trip {round_gap:.1e}")

# q-homodyne of node 1 with record 0.37: diagram conditioning vs the MFvL rule.
measured = CVDiagram((), built.boxes + (Homodyne(1, 0.0, "m"),), (0, 2))
got, expected = to_gaussian_graph(measured, {"m": 0.37}), graph.measure_q(1, 0.37)
rule_gap = max(np.abs(got.Z - expected.Z).max(), np.abs(got.mean - expected.mean).max())
print(f"q-measurement: |diagram conditioning - GaussianGraph.measure_q| = {rule_gap:.1e}")
try:  # the outcome-averaged state is mixed: no graph exists
    to_gaussian_graph(measured)
except ValueError as exc:
    print("averaged state:", exc)
print(f"runtime: {perf_counter() - start:.1f} s")

assert pattern_gap < 1e-11 and np.abs(S - X).max() < 1e-10 and rejected
assert cost["resource_modes"] == cost["measurement_count"] == len(teleport_steps(diagram))
assert graph_gap < 1e-12 and round_gap < 1e-12 and rule_gap < 1e-10
