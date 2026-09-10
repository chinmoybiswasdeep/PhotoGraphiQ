# Gaussian gate compilation


```python
pattern = pg.protocols.identity(squeezing=1.0)
result = pg.simulate(pattern, inputs={0: pg.GaussianInput.coherent(0.3 + 0.1j)}, seed=7)
print(result.state.quadrature(pattern.outputs[0]))

pattern = pg.Circuit(1).rotate(0, 0.4).squeeze(0, 0.2).compile(squeezing=1.2)
channel = pg.gaussian_channel(pattern)
print(channel.matrix, channel.noise)
```

A single `protocols.wire([0])` implements a Fourier step, not identity. Four
steps give identity. `protocols.teleportation` is the two-resource optical
Braunstein–Kimble protocol, distinct from the canonical CZ wire.
Input modes default to vacuum unless supplied. `GaussianInput` supports coherent,
squeezed and arbitrary physical single-mode covariance states. Correlated inputs
can be passed as `initial_state=GaussianState(...)` in pattern input order, without
an `inputs` mapping. A `PiquassoBackend` can import a native Gaussian state via
`import_state(native, nodes, source_hbar=2)`, then its `get_state()` can be supplied
as `initial_state`. Declaring the external hbar prevents silent unit mismatch.


See the [API reference](../api/index.md) and [tutorials](../tutorials/index.md) for executable examples.
