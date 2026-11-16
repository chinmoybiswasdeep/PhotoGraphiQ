# Gaussian generators

All generators are frozen dataclasses that subclass `Box`. Each box declares its ports:
`modes` (bosonic wires it acts on), `creates`, `consumes`, `reads` (records) and `writes`
(records). It also declares two flags: `gaussian` (Gaussian semantics exist) and `unitary`
(a Gaussian unitary on its modes). Import them from `photographiq.calculus.cv_diagrams`.

## Preparations

| Box | Signature | Semantics |
|---|---|---|
| `Prepare` | `Prepare(mode, mean=(0, 0), covariance=((1, 0), (0, 1)), squeezing=None)` | One-mode Gaussian state. The covariance must satisfy V + iΩ ≥ 0. |
| | `Prepare.squeezed(mode, r)` | p-squeezed resource, covariance diag(e^{2r}, e^{−2r}), with `squeezing = r` stored as metadata. This is `GaussianBackend.prepare(node, r)`. r must be finite and ≥ 0. |
| | `Prepare.vacuum(mode)` | Vacuum. |
| `PrepareState` | `PrepareState(labels, mean, covariance)` | Multi-mode Gaussian state, for example from a `GaussianGraph`. The shapes are checked; the physicality of the covariance is not. |

Only preparations built with `Prepare.squeezed` count as resource squeezing in
`resource_cost` and in teleportation-step detection.

## Gaussian unitaries

| Box | Signature | Heisenberg action |
|---|---|---|
| `Rotation` | `Rotation(mode, theta)` | `gaussian.rotation(θ)`: q → q cos θ − p sin θ, p → q sin θ + p cos θ |
| `Squeezer` | `Squeezer(mode, r)` | diag(e^{−r}, e^{r}) (always the q axis) |
| `Shear` | `Shear(mode, s)` | exp(i s q²/4): p → p + s q (PhotoGraphiQ `QuadraticPhase`) |
| `Displacement` | `Displacement(mode, dq=0, dp=0)` | (q, p) → (q + dq, p + dp) |
| `BeamSplitter` | `BeamSplitter(u, v, theta)` | `gaussian.beamsplitter(θ)` |
| `CZ` | `CZ(u, v, g=1.0)` | `gaussian.cz(g)`: p_u += g q_v, p_v += g q_u |
| `Symplectic` | `Symplectic(labels, S, d=None)` | x → S x + d on `labels` (interleaved). S Ω Sᵀ = Ω is checked. |

The two modes of a two-mode box must differ.

## Channels

| Box | Signature | Semantics |
|---|---|---|
| `GaussianLoss` | `GaussianLoss(mode, eta, thermal=0.0)` | x → √η x, added noise (1 − η)(2n̄ + 1) I |
| `GaussianChannelBox` | `GaussianChannelBox(labels, X, Y, d=None)` | x → X x + d with added noise Y. Complete positivity Y + iΩ − iXΩXᵀ ≥ 0 is checked on construction. |

## Measurement and classical boxes

| Box | Signature | Semantics |
|---|---|---|
| `Homodyne` | `Homodyne(mode, theta, key, efficiency=1.0, noise=0.0)` | Destructive measurement of q cos θ + p sin θ. Writes the real record `key`, which has a probability density. Detector noise (1 − η)/η + `noise` is added to the record variance. `.ideal` is true when η = 1 and `noise = 0`. |
| `Feedforward` | `Feedforward(mode, gains, constant=(0, 0))` | Displacement q += c_q + Σ_k a_k m_k, p += c_p + Σ_k b_k m_k. `gains` maps record key → (a_k, b_k), given as pairs or a dict. The form is exactly affine; nothing is approximated. |
| `ClassicalSource` | `ClassicalSource(key, mean, variance)` | A classical Gaussian record N(mean, variance), variance ≥ 0. Produced, for example, when an ancilla is eliminated. |
| `Discard` | `Discard(mode)` | Traces out a mode. |

## Diagram properties

| Property | Meaning |
|---|---|
| `d.records` | record keys in production order |
| `d.is_gaussian` | every box has `gaussian = True` |
| `d.is_unitary` | every box is a Gaussian unitary and the input and output wire sets are equal |

## Gaussian semantics of each generator (checked)

The test `test_generator_semantics_match_photographiq_matrices` checks the diagram semantics
of rotations, squeezers, shears, beam splitters, CZ and displacements against the matrices of
`photographiq.gaussian`, including `gaussian.embed` for the wire placement. Experiment R61
(`paper_experiments/22_cv_diagrams/61_gaussian_rewrites.py`) checks channels and homodyne
records against independent references (see [Rewrite verification](rewrite-verification.md)).

## Non-Gaussian generators

`PhasePolynomial`, `Kerr`, `FockPrepare`, `PhotonCount`, `Postselect`, `GKPPrepare` and
`ModularDecode` are described in [Non-Gaussian](non-gaussian.md) and [GKP](gkp.md).

!!! note "Phase polynomials of degree ≤ 2"
    `PhasePolynomial` with degree ≤ 2 is a Gaussian unitary: p → p + 2c₁ + 4c₂ q, a shear of
    4c₂ followed by a p displacement of 2c₁. `evaluate`, `compare`, `resource_cost` and the
    fusion rules accept it. Degree 3 or higher is non-Gaussian.
