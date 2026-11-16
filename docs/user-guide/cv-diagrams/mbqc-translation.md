# MBQC translation

Diagrams convert to and from PhotoGraphiQ's executable objects. Unsupported input is rejected
with `NotImplementedError`. It is never approximated.

## `Pattern` → diagram: `from_pattern(pattern, parameters=None)`

The pattern is validated first. Symbolic parameters are resolved from `parameters`.

| Pattern command | Diagram box |
|---|---|
| `Prepare(node, r)` (no state) | `Prepare.squeezed(node, r)` |
| `Prepare(node, state=GaussianInput(...))` | `Prepare(node, mean, covariance)` |
| `Prepare(node, state=FockInput)` with a single number state | `FockPrepare(node, n)` |
| `Entangle(u, v, w)` | `CZ(u, v, w)` |
| `Rotate`, `Squeeze`, `QuadraticPhase`, `BeamSplitter`, `Loss` | `Rotation`, `Squeezer`, `Shear`, `BeamSplitter`, `GaussianLoss` |
| `Displace(node, q, p)` with constant q and p | `Displacement` |
| `Displace(node, q, p)` affine in scalar outcomes | `Feedforward` (gains from `calculus.cv.affine_form`) |
| `Measure(node, Homodyne(θ, η, σ²), key)` | `Homodyne(node, θ, key, η, σ²)` |
| `Measure(node, PhotonNumber(), key)` | `PhotonCount(node, key)` |
| `Kerr(node, κ)` | `Kerr(node, κ)` |
| `CubicPhase(node, γ)` | `PhasePolynomial(node, (0, 0, 0, γ/6))` |
| `Output` | (outputs of the diagram) |

These inputs are rejected:

- outcome-dependent parameters, such as adaptive homodyne angles;
- non-affine feed-forward and vector-outcome feed-forward;
- other preparations (for example Fock superpositions) and other measurements;
- every other command (for example photon addition or subtraction).

## Diagram → `Pattern`: `to_pattern(diagram)`

The result is a validated, executable `Pattern`.

| Box | Pattern commands |
|---|---|
| `Prepare` with `squeezing` set | `Prepare(node, r)` |
| other `Prepare` | `Prepare(node, state=GaussianInput(...))` |
| `FockPrepare` | `Prepare(node, state=FockInput.number(n))` |
| one-mode `Symplectic` | `Rotate`, `Squeeze`, `Rotate` (SVD), plus `Displace` if d ≠ 0 |
| `Feedforward` | `Displace` with `Outcome` expressions |
| `ClassicalSource(key, μ, v)` | an ancilla `("source", key)` prepared with q-mean μ and covariance diag(v, 1/v), measured by q-homodyne into `key` |
| `PhasePolynomial` (degree ≤ 3) | `QuadraticPhase(4c₂)`, `Displace(0, 2c₁)`, `CubicPhase(6c₃)` |
| `CZ`, `Rotation`, `Squeezer`, `Shear`, `BeamSplitter`, `Displacement`, `GaussianLoss`, `Homodyne`, `PhotonCount`, `Kerr` | the corresponding command |

`Pattern` has no command for classical randomness. A `ClassicalSource` is therefore realised
exactly by one optical mode, and the recompiled pattern uses one mode per source
([tutorial 6](examples.md#6-recompile-and-execute) checks the averaged channel).

`to_pattern` raises `NotImplementedError` for `Discard` (keep the mode as an output instead),
multi-mode `Symplectic`, `GaussianChannelBox`, `PrepareState`, `Postselect`, `GKPPrepare`,
`ModularDecode` and phase polynomials above degree 3.

## Circuits: `from_circuit` / `to_circuit`

`from_circuit(circuit)` gives the **ideal** direct-gate diagram of a `photographiq.Circuit` on
modes 0..n−1. Labelled rotations and squeezers become `Rotation` and `Squeezer`. Other
one-mode Gaussian gates become `Symplectic`. Identity wires are skipped. `cubic_phase` becomes
a `PhasePolynomial` and `kerr` becomes `Kerr`.

`to_circuit(diagram)` needs wires 0..n−1 in order on both sides. It supports rotations,
squeezers, shears, displacements, CZ, beam splitters, one-mode `Symplectic`, phase polynomials
up to degree 3 and Kerr. The circuit then compiles to MBQC with `circuit.compile(...)`.

## A compiled circuit as a diagram

```python
import photographiq as pg
from photographiq.calculus.cv_diagrams import from_pattern, resource_cost, teleport_steps

circuit = pg.Circuit(2).rotate(0, 0.3).squeeze(1, 0.2).cz(0, 1, 0.5).displace(0, 0.3, 0.1)
diagram = from_pattern(circuit.compile(squeezing=1.0))
print(diagram, len(teleport_steps(diagram)), resource_cost(diagram)["measurement_depth"])
# CVDiagram(inputs=(0, 1), boxes=86, outputs=(22, 18)) 21 13
```

[Tutorial 4](examples.md#4-pattern-to-diagram) checks that the diagram, the pattern and the
round trip `to_pattern(from_pattern(p))` all have the same `analysis.gaussian_channel` (to
1.8e-15). It also checks that an adaptive pattern is rejected.

## CV flow of the underlying cluster

```python
pattern_flow_report(diagram, *, exact=None) -> dict
```

This reads the open graph from the diagram and runs `flows.find_cv_flow`, the Booth–Markham
real-linear flow (see [Flow](../flow.md) and [MBQC calculus](../mbqc-calculus.md)). The graph
is built from:

- CZ edges, with their weights summed;
- squeezed preparations, which give the node squeezing;
- homodyne angles, which give the shear cot θ;
- the input and output wires.

`Feedforward`, `Displacement` and `ClassicalSource` boxes are frame shifts and are skipped.
The result has the keys `status`, `decision`, `layers`, `nodes`, `edges`, `seconds` and
`regimes`. A q measurement (θ = 0) or any other box returns
`{"status": "unsupported", "reason": ...}`. Other boxes include a `Rotation`, a
`BeamSplitter` or a loss.
