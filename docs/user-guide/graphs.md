# Graphs and labelled resources


```python
import photographiq as pg
import networkx as nx

graph = pg.CVGraph.line(4, squeezing=1.0)
result = pg.simulate(pg.Pattern(graph), seed=1)
print(graph.nullifiers() @ result.state.covariance @ graph.nullifiers().T)

weighted = nx.Graph()
weighted.add_edge("left", "right", weight=0.7)
graph = pg.CVGraph(weighted, squeezing={"left": 0.7, "right": 1.2}, inputs=("left",))
```

Labels can be hashable Python objects at runtime; safe JSON supports primitive
labels and tuples of them. Node squeezing and edge weights may be `Parameter`
expressions. NetworkX edge attributes use `weight`; node attributes use `squeezing`.
Directed graphs, multigraphs, self-loops and nonfinite numerical resources fail.
Temporal and dual-rail constructors mean line/ladder topologies, not detailed
optical hardware models. `graph.ancillas` excludes inputs; measured nodes are
defined by a pattern's `Measure` commands, and `pattern.outputs` reports survivors.


See the [API reference](../api/index.md) and [tutorials](../tutorials/index.md) for executable examples.
