"""Run with: python examples/first_cluster.py.

This hand-written resource illustrates measurement and feed-forward. Outcomes
are sampled quadratures; mean/covariance describe the surviving conditional mode.
Use a certified protocol builder when a particular gate transformation is needed.
"""

import photographiq as pg

graph = pg.CVGraph.line(3, squeezing=1.0, inputs=(0,))
pattern = pg.Pattern(graph)
pattern.measure(0, pg.Homodyne.p())
pattern.displace(1, q=-pg.Outcome(0))
# This hand-written pattern is causal; use protocols.wire for a gate-certified wire.
pattern.measure(1, pg.Homodyne(angle=pg.Parameter("theta")))
result = pg.simulate(
    pattern, parameters={"theta": 0.7}, inputs={0: pg.GaussianInput.coherent(0.2)}, seed=42
)
print("Outcomes:", result.outcomes)
print("Output nodes:", result.state.nodes)
print("Mean:", result.state.mean)
print("Covariance:\n", result.state.covariance)
