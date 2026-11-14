**Flow finding and measurement layering.** (a) A six-node qubit open graph (input 0, outputs
4 and 5, XY-plane measurements at angles 0.1 n). (b) `flows.analyze` finds a
`causal-flow (Danos-Kashefi 2006; Mhalla-Perdrix 2008 Alg. 1)` with status `certified`; the layered layout places each measured node
in its layer column (`Lk`, measurement order `#k`): layers
[[0], [2], [1, 3]]. Arrows are exactly the dependencies of the
reported correction map (corrections act on later layers and outputs only). (c) A 2 x 4 CV
ladder (`CVGraph.rectangular`, homodyne shears 0.3 ... 1.8) analysed with the real-linear CV
flow: status `certified`, layers [['(0, 0)', '(1, 0)'], ['(0, 1)', '(1, 1)'], ['(0, 2)', '(1, 2)']].
Qubit flow certificates are not CV certificates and vice versa; failing to certify is not a
proof of nondeterminism.

Source: `scripts/generate_demo_figures.py` (`06_flow_finding_and_layering`).
