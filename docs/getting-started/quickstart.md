# Your first five minutes

Prepare a coherent input, measure one end of a two-node cluster, and correct the
remaining mode. Install the visualization extra to display the graph.

```python
import photographiq as pg

graph = pg.CVGraph.line(2, squeezing=1.0, inputs=(0,))
pattern = pg.Pattern(graph).measure(0, pg.Homodyne.p())
pattern.displace(1, q=-pg.Outcome(0))
result = pg.simulate(pattern, inputs={0: pg.GaussianInput.coherent(0.3 + 0.2j)}, seed=7)
print(result.outcomes)
print(result.state.nodes)
print(result.state.quadrature(1))
pattern.draw(output="first-cluster.svg")
```

The outcome dictionary holds real sampled homodyne readings, not amplitudes.
`nodes` identifies surviving modes. `quadrature(1)` returns the mean and variance
of q on mode 1. A single wire step includes a Fourier transform; it is not identity
teleportation. Resource squeezing is finite, so interpret the conditional output
using its covariance rather than assuming an ideal unitary state.

## Start from a circuit

```python
circuit = pg.Circuit(1).rotate(0, 0.4).squeeze(0, 0.2)
compiled, trace = circuit.compile(squeezing=1.2, return_trace=True)
pg.visualize_compilation(circuit, compiled, trace=trace, output="compilation.svg")
result = pg.simulate(compiled, seed=7)
print(result.state.mean)
```

Matching colors connect each gate to its generated resource nodes. The trace also
records measurement and correction command indices. `compiled` is a reusable
Pattern. [Tutorial 20](../tutorials/20-compilation-visualization.md) walks through
the correspondence. Next read [inputs](../user-guide/inputs.md) and
[outputs](../user-guide/outputs.md), then try the executable tutorial series.
