# User guide

> Historical v0.2 reference. For the v0.3 additions and current backend support, see the [documentation home](index.md) and [feature matrix](validation/feature-matrix.md).

Install with `pip install -e '.[dev]'` in a Python environment. `visualization`
and `validation` extras install matplotlib and GraphiX separately. The package
pins Piquasso because its covariance conventions and instruction support are
explicitly tested. Read [the mathematical reference](theory.md) before interpreting
measurement outcomes or finite-squeezing fidelity.

## First cluster and arbitrary graphs

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

## Teleportation, Gaussian gates and inputs

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

## Adaptive measurements and feed-forward

```python
pattern = pg.protocols.adaptive(squeezing=0.8)
result = pg.simulate(pattern, parameters={"k": 0.3}, seed=10, frame=True)
print(result.outcomes)

from photographiq.expressions import CallableExpression

angle = CallableExpression(lambda records: 0.2 + records[0] ** 2, dependencies={0})
```

`Outcome(key)` refers to an earlier measurement or `Signal`. Heterodyne outcomes
need `Outcome(key, component=0)` or `component=1`. Expressions support arithmetic,
sin, cos, exp and atan2. A callable only receives its declared record subset and
cannot be serialized. Parameters are externally bound real scalars; there is no
string evaluation or automatic differentiation. `Signal('name', expression)` stores
a new classical register, and `Result.records` contains both signals and outcomes.
Duplicate keys, missing producers, cycles and future dependencies are rejected.

## Fixed-order CV-flow

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

## Finite squeezing, noise and repeated shots

```python
pattern = pg.protocols.identity(squeezing=0.8)
shots = pg.run_shots(pattern, 1000, backend="gaussian", seed=123)
ensemble = shots.ensemble_state()
print(ensemble.covariance)

pattern = pg.Pattern().append(pg.Prepare(0, 0.8))
pattern.append(pg.Loss(0, transmissivity=0.9, thermal_photons=0.05))
pattern.measure(0, pg.Homodyne(angle=0.2, efficiency=0.85, noise=0.02))
```

Each shot receives a distinct child seed and executes a complete adaptive
trajectory. `shots.values(key)` extracts a record array. Seeds including zero
are reproducible. Finite squeezing, environmental photon loss and detector
inefficiency are separate mechanisms. `ensemble_state` includes covariance of
the trajectory means, not just the average conditional covariance. For nonlinear
adaptation the ensemble is generally not a Gaussian state; the container represents
only its moments, so Gaussian fidelity/parity formulas for that container need not
equal the mixture's actual observables. Compute per-trajectory observables and
average them instead when necessary.

## Non-Gaussian resources

```python
resource = pg.FockInput.cat(alpha=0.4, cutoff=12).photon_added()
pattern = pg.Pattern().append(pg.Prepare("a", state=resource))
pattern.append(pg.CubicPhase("a", gamma=0.005))
result = pg.simulate(pattern, backend="piquasso-fock", cutoff=16, seed=4)
print(result.state.photon_number("a"), result.state.retained_norms)
```

Photon-number measurement is supported via `PhotonNumber()`. Use a cutoff scan
to check observables. A low retained norm raises an exception. Gaussian MBQC uses
no Fock cutoff. Noisy Fock homodyne, mixed Fock inputs, GKP resources and universal
non-Gaussian compilation are explicitly unavailable in this release.

## Saving, inspecting and plotting

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

See [Experimental non-Gaussian MBQC](non_gaussian.md) for conditional homodyne,
correlated resources, cubic/cat injection, heralding, Wigner plots and cutoff studies.
