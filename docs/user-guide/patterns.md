# Patterns and reusable computations


```python
text = pattern.to_json()
restored = pg.Pattern.from_json(text)
print(restored.inspect())

from photographiq.visualization import draw_graph, draw_dependencies

draw_graph(pg.CVGraph.square(3)).figure.savefig("cluster.pdf", bbox_inches="tight")
draw_dependencies(pg.protocols.wire([0, 0.3])).figure.savefig("dag.pdf", bbox_inches="tight")
```

`to_json(path)` optionally writes to a file; `from_json` takes JSON text rather
than an ambiguous path. Quantum state trajectories are not serialized as patterns.
Drawing returns a matplotlib Axes for customization. Inputs, outputs, labels,
weights and squeezing are visible; `draw_pattern` adds measurement order and
angle annotations, while `draw_dependencies` shows classical and quantum causal
edges. For large patterns export these views separately for legibility.

See [Experimental non-Gaussian MBQC](../non_gaussian.md) for conditional homodyne,
correlated resources, cubic/cat injection, heralding, Wigner plots and cutoff studies.


See the [API reference](../api/index.md) and [tutorials](../tutorials/index.md) for executable examples.
