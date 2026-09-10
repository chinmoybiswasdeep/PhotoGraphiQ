# Causality and supplied-order CV-flow


```python
from photographiq.cvflow import certify_cv_flow, flow_pattern

graph = pg.CVGraph.line(4, squeezing=0.8, inputs=(0,), outputs=(3,))
certificate = certify_cv_flow(graph, [0, 1, 2])
pattern = flow_pattern(graph, [0, 1, 2], shears={1: 0.3})
```

The certificate solves a real-linear correction equation at each cut of this
specific order. It does not optimize or search orders, and graph inputs cannot
be used as resource stabilizer correction columns. In contrast,
`pattern.dependencies()` is an ordinary causal command DAG and `pattern.schedule()`
returns command indices. Neither should be labeled qubit gflow.


See the [API reference](../api/index.md) and [tutorials](../tutorials/index.md) for executable examples.
