# MBQC calculus and flow analysis

This page covers the graph-level IR (`photographiq.ir`), flow and determinism
analysis (`photographiq.flows`) and the measurement calculus
(`photographiq.calculus`): DKP qubit patterns, local Cliffords and a ZX subset.
All of it is pure Python/NumPy and runs on the CPU. The lower-level
supplied-order CV-flow certificate (`photographiq.cvflow`) and CV command
scheduling are described in [Causality and supplied-order CV-flow](flow.md) and
are not repeated here.

Every example on this page was run in the project environment. Each printed
output is shown as a comment.

## Open graphs and computational models

`OpenGraph` is the typed graph-level IR shared by flow analysis, compilation,
measurement calculus and visualization. Its `Model` is part of the data, and
the models are never interchanged:

| Model | Measurement spec | Edges | Corrections |
|---|---|---|---|
| `Model.QUBIT` | `QubitMeasurement(plane, angle, pauli=None)` | unweighted | Pauli X/Z |
| `Model.CV_GAUSSIAN` | `HomodyneSpec(shear)`, angle `atan2(1, shear)` | real, nonzero weights | displacements |
| `Model.CV_NONGAUSSIAN` | `HomodyneSpec` | real weights | not analysed |
| `Model.GKP_LOGICAL` | `QubitMeasurement` | unweighted | not analysed |

The qubit measurement planes measure XY `cos(a) X + sin(a) Y`,
YZ `cos(a) Z + sin(a) Y` and XZ `cos(a) Z + sin(a) X`. Outcome 0 is the +1
eigenvalue. These are the same planes as Graphix 0.4 (Graphix uses units of pi).

```python
import networkx as nx
from photographiq.ir import Model, OpenGraph, QubitMeasurement

og = OpenGraph(Model.QUBIT, nx.path_graph(4), (0,), (3,),
               {n: QubitMeasurement("XY", 0.3) for n in range(3)})
print(og.measured, OpenGraph.from_dict(og.to_dict()).to_dict() == og.to_dict())
# (0, 1, 2) True
try:
    OpenGraph(Model.QUBIT, nx.Graph([(0, 1, {"weight": 2.0})]), (0,), (1,),
              {0: QubitMeasurement()})
except ValueError as e:
    print(e)
# qubit graphs are unweighted; edge (0, 1)
```

`OpenGraph.from_cvgraph(graph, shears)` and `OpenGraph.to_cvgraph()` convert
between CV models and an executable `CVGraph`. `to_dict()` and `from_dict()` give
a JSON-compatible description. Layout coordinates are not part of the IR.

## Flow analysis

`photographiq.flows.analyze(og, algorithm="auto")` applies only the flow notion
of the graph's model:

- `Model.QUBIT`: causal flow, then gflow, then Pauli flow (Pauli flow only when
  some measurement has a `pauli` tag). All three use GF(2) on the unweighted
  graph and return maximally delayed layers. Each finder is complete for its
  notion, and an independent verifier checks every certificate before it is
  returned.
- `Model.CV_GAUSSIAN`: real-linear CV flow on the weighted adjacency, plus a
  numerical determinism check.
- `Model.CV_NONGAUSSIAN` and `Model.GKP_LOGICAL`: status `unsupported`.

You can also call a finder directly: `find_causal_flow`, `find_gflow`,
`find_pauli_flow` or `find_cv_flow`. The verifiers `verify_causal_flow`,
`verify_gflow` and `verify_pauli_flow` return a list of violated conditions;
an empty list means the certificate is valid.

### `FlowAnalysisResult`

Every analysis returns a frozen `FlowAnalysisResult` with these fields:
`status`, `model`, `algorithm`, `measurement_order` (total order, earliest
first), `layers` (partial order; nodes in one layer are mutually unordered, and
outputs are not listed), `correction_map`, `certificate`, `assumptions` and
`diagnostics`. The statuses mean:

| Status | Meaning |
|---|---|
| `certified` | A certificate was found and independently verified. |
| `not_certified` | The finder found no certificate for this flow notion. `diagnostics['complete']` says whether the finder is complete; when it is, no such flow exists (for CV flow: in exact arithmetic, and up to `tolerance` numerically). This is never a proof of nondeterminism. |
| `unknown` | The analysis could not decide, e.g. the numerical CV check failed or could not run. |
| `unsupported` | The model and algorithm combination is outside the implemented scope. |

Qubit correction maps have the form `{node: {'x': frozenset, 'z': frozenset}}`,
restricted to nodes measured later and outputs. CV correction maps have the
form `{node: {target: {'q': cq, 'p': cp}}}`, meaning `q += cq*m` and
`p += cp*m` with `m = record / sin(theta)`.

### Causal flow and gflow

```python
import networkx as nx
from photographiq.ir import Model, OpenGraph, QubitMeasurement
from photographiq import flows

og = OpenGraph(Model.QUBIT, nx.path_graph(4), (0,), (3,),
               {n: QubitMeasurement("XY", 0.3) for n in range(3)})
r = flows.find_causal_flow(og)
print(r.status, r.certificate["flow"], r.layers, r.correction_map[1])
# certified {2: 3, 1: 2, 0: 1} ((0,), (1,), (2,)) {'x': frozenset({2}), 'z': frozenset({3})}

g = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 3), (2, 4), (3, 4)])
og2 = OpenGraph(Model.QUBIT, g, (3, 2), (0, 4),
                {n: QubitMeasurement("XY", 0.5) for n in (1, 2, 3)})
print(flows.find_causal_flow(og2).status, flows.find_gflow(og2).status)
# not_certified certified
r2 = flows.analyze(og2)
print(r2.algorithm.split()[0], r2.layers,
      flows.verify_gflow(og2, r2.certificate["correction_sets"], r2.layers))
# gflow ((2, 3), (1,)) []
```

Causal flow is defined for XY measurements only. Graphs with YZ or XZ
measurements need gflow.

When no flow exists, the result says so without claiming that the pattern is
nondeterministic:

```python
import networkx as nx
from photographiq.ir import Model, OpenGraph, QubitMeasurement
from photographiq import flows

tri = OpenGraph(Model.QUBIT, nx.complete_graph(3), (0, 1), (2,),
                {0: QubitMeasurement(), 1: QubitMeasurement()})
res = flows.analyze(tri)
print(res.status, res.diagnostics["reason"])
# not_certified no certificate from causal, gflow; this does not prove nondeterminism
```

### Pauli flow

A node is treated as a Pauli measurement only when `QubitMeasurement.pauli` is
set, and its plane and angle must describe that Pauli observable; otherwise
`pauli_label` raises `ValueError`. Pauli-measured X/Y nodes may correct each
other inside one layer.

```python
import math
import networkx as nx
from photographiq.ir import Model, OpenGraph, QubitMeasurement
from photographiq import flows

xm = QubitMeasurement("XZ", math.pi / 2, "X")  # X Pauli written in the XZ plane
og = OpenGraph(Model.QUBIT, nx.path_graph(3), (0,), (2,), {0: xm, 1: xm})
print(flows.pauli_label(xm), flows.find_gflow(og).status)
# X not_certified
r = flows.analyze(og)
print(r.algorithm.split()[0], r.certificate["correction_sets"], r.layers)
# pauli-flow {0: frozenset({1}), 1: frozenset({2})} ((0, 1),)
```

### CV determinism

For `Model.CV_GAUSSIAN`, `find_cv_flow` (also reached through `analyze`) does
three things:

1. It searches an admissible measurement order by backward real-linear
   layering over the weighted adjacency (Booth and Markham, Quantum 7, 1146
   (2023)).
2. It re-certifies that order with `photographiq.cvflow.certify_cv_flow`.
3. Unless `verify=False`, it builds the corrected finite-squeezing pattern from
   exactly the reported correction map and runs `verify_cv_determinism`.

The edge weights decide whether a CV flow exists: CV flow is a different notion
from qubit gflow on the same skeleton.

```python
import networkx as nx
from photographiq.ir import Model, OpenGraph, HomodyneSpec
from photographiq import flows

g = nx.Graph()
g.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
og = OpenGraph(Model.CV_GAUSSIAN, g, (0,), (3,),
               {0: HomodyneSpec(0.2), 1: HomodyneSpec(-0.4), 2: HomodyneSpec(0.7)},
               {n: 0.6 for n in g})
r = flows.analyze(og)
print(r.status, r.layers, r.correction_map[0])
# certified ((0,), (1,), (2,)) {1: {'q': -1.0, 'p': -0.0}, 2: {'q': -0.0, 'p': -1.0}}
rep = r.certificate["determinism"]
print(rep["ideal_limit_deterministic"], rep["finite_squeezing_gain"] > 0.1)
# True True
print(flows.analyze(og, "gflow").status)
# unsupported
```

CV determinism holds only in the infinite-squeezing limit. To check it,
`verify_cv_determinism` raises every resource squeezing by 0, `shift` and
2 `shift`, computes the outcome-averaged channel `(S, N)`, and obtains the
conditional output moments by exact Gaussian postselection (`simulate` with
fixed `measurement_outcomes` on the `gaussian` backend). These postselected
runs are a numerical certificate, not a deterministic execution. The report
keeps the following quantities separate:

- the ideal map `matrix`
- the finite-squeezing noise (`finite_squeezing_noise`)
- the remaining dependence of the conditional output on the records at finite
  squeezing (`finite_squeezing_gain`), which must decay like `exp(-2 kappa x)`

If the numerical check fails or cannot run, the status is `unknown`, never
`certified`. The check is reliable only at moderate resource squeezing. For
the four-node wire above (with `HomodyneSpec()` on every node and the default
`shift`), every r from 2.5 to 3.2 is certified. At r = 3.3 the check fails, and
from r = 3.5 to 4.0 it cannot run because the raised covariances fail float64
Gaussian validation. Some values in this range (r = 3.4, and r just above 4.0)
are still certified. Treat `unknown` at high squeezing as undecided.

`search_cv_order(og)` runs the order search alone. `resource_graph(og)` returns
the `CVGraph` whose output order matches the corrected pattern.

## Qubit measurement calculus

`photographiq.calculus.qubit` implements the DKP measurement calculus. It has
the commands `N`, `E`, `M`, `X`, `Z` and `S`, `QubitPattern`, exact branch
simulation, equivalence and determinism checks, standardization, signal
shifting and flow-to-pattern construction. Commands are listed in execution
order. Branch maps are compared per branch up to a phase of that branch, which
cannot be observed. Dense simulation is limited to `MAX_QUBITS = 20` qubits
that are live at the same time, plus the inputs.

```python
from photographiq.calculus.qubit import N, E, M, X, QubitPattern
from photographiq.calculus import standardize, equivalent, is_deterministic

p = QubitPattern((0,), [N(1), E(0, 1), M(0, "XY", 0.3), X(1, {0}),
                        N(2), E(1, 2), M(1, "XY", 0.5), X(2, {1})])
trace = []
s = standardize(p, trace=trace)
print(p.is_standard(), s.is_standard(), sorted({rule for rule, _ in trace}))
# False True ['EX', 'MX', 'commute']
print(equivalent(p, s).equivalent, is_deterministic(s).deterministic)
# True True
```

`pattern_from_flow(og, result)` builds the standard pattern of a qubit open
graph from a certified flow result or from an explicit correction table. It
does not assume determinism, so check the result with `is_deterministic`.
`signal_shift` moves the outcome-flipping part of each measurement domain into
`S` commands:

```python
import networkx as nx
from photographiq.ir import Model, OpenGraph, QubitMeasurement
from photographiq.flows import find_causal_flow
from photographiq.calculus import pattern_from_flow, signal_shift, equivalent, is_deterministic
from photographiq.calculus.qubit import S

og = OpenGraph(Model.QUBIT, nx.path_graph(4), (0,), (3,),
               {n: QubitMeasurement("XY", 0.2 * (n + 1)) for n in range(3)})
pat = pattern_from_flow(og, find_causal_flow(og))
sh = signal_shift(pat)
print(is_deterministic(pat).deterministic,
      [(c.node, set(c.domain)) for c in sh.commands if isinstance(c, S)],
      equivalent(pat, sh).equivalent)
# True [(2, {0})] True
```

The test suite cross-checks standardization, signal shifting and branch
statevectors against Graphix 0.4 (`tests/calculus/test_calculus_graphix.py`).

On the CV side, `photographiq.calculus.cv` provides two tools. `command_view`
returns command roles, outcome variables and displacement corrections as exact
affine forms. `ideal_limit_check` compares the unconditional Gaussian channel
with an ideal symplectic target at several squeezing values and reports the
added noise separately. No CV graphical calculus is implemented.

```python
from photographiq.calculus.cv import ideal_limit_check, wire_step_pattern
from photographiq.gaussian import teleportation_matrix

rep = ideal_limit_check(lambda r: wire_step_pattern(0.3, r), teleportation_matrix(0.3),
                        [0.5, 1.0, 2.0])
print(rep.max_matrix_error < 1e-14, [round(p.noise_norm, 4) for p in rep.points])
# True [0.3679, 0.1353, 0.0183]
```

## Local Clifford transformations

`photographiq.calculus.clifford` provides the following:

- the 24-element single-qubit Clifford group (`Clifford`)
- qubit graph states (`graph_state`)
- local complementation (`local_complement`), with the local Clifford that
  implements it (`lc_cliffords`)
- an exhaustive LC-equivalence search for small labelled graphs
  (`lc_equivalence`; at most 10 nodes by default, without vertex relabelling)

`lc_equivalence` returns a witness that has been checked on statevectors, or
`None` after the whole orbit has been searched.

```python
import networkx as nx
from photographiq.calculus import Clifford, local_complement, lc_equivalence

print(len(Clifford.all()), Clifford.named("HS"))
# 24 Clifford(4, 'HS')
print(sorted(local_complement(nx.star_graph(3), 0).edges))
# [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
w = lc_equivalence(nx.star_graph(3), nx.complete_graph(4))
print(w.sequence, w.residual < 1e-12, lc_equivalence(nx.star_graph(3), nx.path_graph(4)))
# (0,) True None
```

Qubit local complementation does not transfer to weighted CV graph states.
`cv_local_complementation()` always raises `NotImplementedError`.

## ZX subset

`photographiq.calculus.zx.Diagram` is a qubit ZX diagram whose `matrix()`
method returns the exact linear map, including the scalar. The implemented
rewrites are listed below. Each one checks its precondition, raises
`RewriteError` when the precondition fails, multiplies `scalar` by the exact
factor and records a `RewriteStep` in `trace`.

- `fuse`
- `remove_self_loops`
- `remove_identity`
- `color_change`
- `copy_state` (Pauli states only)
- `bialgebra` (2x2 only)
- `hopf`
- `cancel_hadamard_pair`
- `simplify_spiders` (applies fusion, self-loop removal and identity removal
  until nothing changes)

The following are not implemented: pi-copy, ZX local complementation and
pivoting, bialgebra of other arities than 2x2, circuit extraction, full
simplification strategies, and CV-ZX identities.

```python
import numpy as np
from photographiq.calculus import zx
from photographiq.calculus.zx import Diagram

d = Diagram()
a, b = d.add_vertex("Z", 0.4), d.add_vertex("Z", 0.7)
d.add_edge(d.add_input(), a); d.add_edge(a, b); d.add_edge(b, d.add_output())
f = zx.fuse(d, a, b)
print(np.allclose(f.matrix(), d.matrix()), round(f.phase(a), 6), f.trace)
# True 1.1 [RewriteStep(rule='fuse', vertices=(0, 1), factor=(1+0j))]
```

`graph_state_diagram(G)` builds a diagram equal to the graph state `|G>`.
`pattern_branch_diagram(pattern, outcomes)` builds a diagram whose matrix
equals one branch Kraus operator of a `QubitPattern`:

```python
import numpy as np
from photographiq.calculus import zx
from photographiq.calculus.qubit import N, E, M, X, QubitPattern, run_branch

p = QubitPattern((1,), [N(2), E(1, 2), M(1, "XY", -0.6), X(2, {1})])
bd = zx.pattern_branch_diagram(p, {1: 1})
s = zx.simplify_spiders(bd)
print(np.allclose(s.matrix(), run_branch(p, {1: 1}).operator), len(bd.spiders()), len(s.spiders()))
# True 5 2
```

## See also

- [Causality and supplied-order CV-flow](flow.md): `certify_cv_flow`,
  `flow_pattern` and the CV command DAG.
- [Architecture figures](../architecture/README.md), Figure E (graph-theoretic
  compilation: flow analysis, `pattern_from_flow` and the calculus).
- Tests: `tests/flow/`, `tests/calculus/`, `tests/graphix_reference/`.
  Graphix benchmark: `benchmarks/graphix/bench_graphix_mbqc.py`.
