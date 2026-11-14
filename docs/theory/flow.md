# Graph flow and determinism

A measurement pattern is **deterministic** when its output does not depend on the measurement
outcomes once the corrections are applied. A *flow* is a combinatorial or linear-algebraic
certificate that a graph, with a given measurement order, admits such corrections. This page
states which notions PhotoGraphiQ implements, what each certificate means and, as important,
what a failure does **not** mean. Code: `src/photographiq/flows/` (search, verification,
`FlowAnalysisResult`) and `src/photographiq/cvflow.py` (supplied-order CV certificate).

## Scheduling is not determinism

`pattern.dependencies()` (built by `photographiq.flow.dependency_graph`) combines classical producer→consumer edges
(which outcome feeds which command) with consecutive operations on the same mode. A
topological order of that DAG is a valid *schedule*. It proves nothing about determinism.

## Qubit flows

For qubit open graphs $(G,I,O)$ with measurement planes (`Model.QUBIT`):

| Notion | Condition (informal) | Reference |
|---|---|---|
| Causal flow | a map $f:O^c\to I^c$ with $u\sim f(u)$, $u\prec f(u)$, and every other neighbour of $f(u)$ after $u$ | Danos and Kashefi 2006 |
| Generalised flow (gflow) | a set $g(u)\subseteq I^c$ whose odd neighbourhood contains $u$ (plane-dependent), with consistent order | Browne, Kashefi, Mhalla, Perdrix 2007 |
| Pauli flow | gflow relaxed for Pauli-measured nodes | Browne et al. 2007; Simmons 2021 |

`flows.analyze` runs the requested notion (by default causal flow, then gflow) and
**independently verifies** any certificate it finds before reporting `certified`. Results
agree with Graphix 0.4 on 360/360 random instances
([validation status](../validation/MASTER_VALIDATION_STATUS.md)).

## CV flow

For weighted CV graphs (`Model.CV_GAUSSIAN`) PhotoGraphiQ uses the real-linear correction
condition of Booth and Markham (Quantum 7, 1146, 2023). For a total order of the measured
nodes and each measured node $u$, find a real vector $c_u$ supported on later, non-input
nodes such that

$$(A\,c_u)_v=\delta_{uv}\quad\text{for every node } v \text{ measured up to and including } u,$$

with $A$ the **weighted** adjacency. GF(2) qubit rules are never used on CV graphs. The
correction for the rescaled homodyne record $m=y/\sin\theta$ is then a $q$-displacement
$-c_u m$ on the support of $c_u$ and the induced $p$-displacement $-(Ac_u)m$ on the surviving
nodes.

**Order search.** Layers are built backwards: a node joins the next-earlier layer when
$A[W,C]\,c=e_u$ is solvable ($W$ unsolved nodes, $C$ solved non-input nodes). Solvability is
monotone, so in exact arithmetic the search finds an admissible order if and only if one
exists. Each decision is a float64 rank test with reported margins; decisions inside the
uncertain band are re-decided in exact rational arithmetic or reported as uncertain.

| `status` | `diagnostics["decision"]` | Meaning |
|---|---|---|
| `certified` | `certified` | order and corrections found and verified; `certificate["regimes"]` says whether it covers the ideal limit only or also finite squeezing |
| `not_certified` | `impossible` | no flow exists; `diagnostics["proof"]` gives a combinatorial obstruction, exact rational witnesses, or a float64 backward-error statement. The pattern can only be postselected |
| `unknown` | `numerically_uncertain` | the float64 decision is inside the uncertainty band and was not resolved exactly |
| `unsupported` | — | model/algorithm combination outside the implemented scope (e.g. gflow on a CV graph) |

```python
import networkx as nx
from photographiq import flows
from photographiq.ir import HomodyneSpec, Model, OpenGraph, QubitMeasurement

# Qubit: a triangle with two inputs and one output has no causal flow or gflow.
tri = OpenGraph(Model.QUBIT, nx.complete_graph(3), (0, 1), (2,),
                {0: QubitMeasurement(), 1: QubitMeasurement()})
res = flows.analyze(tri)
print(res.status, "|", res.diagnostics["reason"])
# not_certified | no certificate from causal, gflow; this does not prove nondeterminism

# CV: a weighted bridge whose input-output block is singular has no CV flow.
g = nx.Graph()
g.add_weighted_edges_from([(0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (1, 3, 1.0)])
bridge = OpenGraph(Model.CV_GAUSSIAN, g, (0, 1), (2, 3),
                   {0: HomodyneSpec(0.0), 1: HomodyneSpec(0.0)}, {n: 1.0 for n in g})
res = flows.analyze(bridge)
print(res.status, res.diagnostics["decision"])      # not_certified impossible
```

## What a certificate does and does not say

- A CV flow certificate is a statement about the **ideal infinite-squeezing** protocol:
  outcome-independent output up to the corrections. At finite squeezing the corrected
  pattern is still a noisy channel ([cluster states](cluster-states.md)). The
  `finite_squeezing` regime label is a float64 claim about that specific pattern.
- A qubit flow certificate is **not** a CV certificate, and vice versa.
- `not_certified` from a search that is not complete for the notion is not a proof of
  nondeterminism. Only CV results with `decision == "impossible"` carry an impossibility
  proof.
- `cvflow.certify_cv_flow(graph, order)` checks one supplied order; failure for that order
  says nothing about other orders. `flow_pattern(graph, order)` builds the corrected pattern.

## Measurement calculus

Turning a flow into a runnable qubit pattern, standardization and signal shifting are covered
in [measurement calculus, local Cliffords and ZX](calculus.md).

## References

V. Danos and E. Kashefi, PRA 74, 052310 (2006); D. E. Browne, E. Kashefi, M. Mhalla and
S. Perdrix, New J. Phys. 9, 250 (2007); R. I. Booth and D. Markham, Quantum 7, 1146 (2023).
Robustness study: [CV flow robustness](../validation/cv-flow-robustness.md).
