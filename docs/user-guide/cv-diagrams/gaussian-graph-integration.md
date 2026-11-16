# GaussianGraph integration

[`GaussianGraph`](../gaussian-graphical-calculus.md) represents a pure Gaussian **state** as a
complex symmetric matrix Z = V + iU (MFvL, PRA 83, 042335 (2011)). A `CVDiagram` represents a
**process**. The two stay separate modules. Two converters connect them:

| Function | Direction | Notes |
|---|---|---|
| `from_gaussian_graph(graph)` | graph → diagram | An input-free diagram with one `PrepareState(graph.nodes, graph.mean, graph.to_covariance())` box. Its outputs are the graph nodes. |
| `to_gaussian_graph(diagram, outcomes=None, *, rtol=1e-9)` | diagram → graph | The diagram must have no inputs. If it has records, `outcomes` must fix every one of them, and the **conditional** output state is used. The state must be pure; `GaussianGraph.from_covariance` raises otherwise. |

Without `outcomes`, a diagram with records raises `ValueError` ("averaged states are mixed").
An outcome-averaged state, or a state after loss, is mixed and has no graph.

## A cluster built from generators

```python
from photographiq.calculus import GaussianGraph
from photographiq.calculus.cv_diagrams import CZ, CVDiagram, Homodyne, Prepare, to_gaussian_graph

r = 0.7
graph = GaussianGraph.cluster([[0, 1.0, 0], [1.0, 0, 0.5], [0, 0.5, 0]], r)
built = CVDiagram(
    (),
    tuple(Prepare.squeezed(j, r) for j in range(3)) + (CZ(0, 1, 1.0), CZ(1, 2, 0.5)),
    (0, 1, 2),
)
print(abs(to_gaussian_graph(built).Z - graph.Z).max())
# 0.0
```

The diagram gives Z = A + i e^{−2r} I, which is MFvL Eq. 2.29.

## Measurement rules as conditioning

A q-homodyne of node 1 with record 0.37, computed by conditioning the diagram semantics, equals
the MFvL q-measurement rule (vertex deletion):

```python
measured = CVDiagram((), built.boxes + (Homodyne(1, 0.0, "m"),), (0, 2))
got, expected = to_gaussian_graph(measured, {"m": 0.37}), graph.measure_q(1, 0.37)
print(abs(got.Z - expected.Z).max(), abs(got.mean - expected.mean).max())
# 0.0 0.0
```

The test suite checks the same thing for p-homodyne (`measure_p`). The process-level form of
vertex deletion is the rewrite rule `cz-q-measurement-to-feedforward`
([rule table](rewrite-rules.md#rule-table)).

## Wire shortening is a limit, not an identity (R62)

Experiment R62 (`paper_experiments/22_cv_diagrams/62_wire_shortening.py`, status `pass`)
measured interior nodes of a 5-node linear cluster at finite squeezing. It computed the
post-measurement graph twice: once with `GaussianGraph.measure_p` and once by conditioning the
diagram. The two agreed at every r in [0, 3].

| r | p-homodyne of one interior node: new coupling (imaginary) | p-homodyne of two adjacent nodes: new real edge |
|---|---|---|
| 0 | 1.0 i | −0.5 |
| 1 | 7.389 i | −0.98201 |
| 2 | 54.598 i | −0.99966 |
| 3 | 403.43 i | −0.999994 |

Measuring one node couples its neighbours by −1/Z_kk = i e^{2r}. This is a diverging
squeezing-like term, not a CZ edge. Measuring two adjacent nodes gives a real edge that tends
to −1 (a CZ of weight −1) with corrections O(e^{−2r}). The familiar two-node wire shortening
therefore holds only in the ideal limit. At finite squeezing it is a different state, which
is why diagrams never use it as an exact rule. The exact process-level replacement is
teleport-chain resynthesis, which is **logical** only; see
[Resource optimisation](resource-optimization.md).

[Tutorial 4](examples.md#4-pattern-to-diagram) runs the cluster and q-measurement checks.
