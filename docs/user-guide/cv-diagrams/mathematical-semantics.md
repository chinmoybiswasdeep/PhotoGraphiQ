# Mathematical semantics

This page states what `evaluate(diagram)` computes and how to read the result. The proofs
are in [the theory note](../../theory/CV_ZX_FOUNDATIONS.md) (§3).

## Typed wires

A `CVDiagram(inputs, boxes, outputs)` is immutable. Its wires have one of two types:

| Wire type | Label | Created by | Consumed by | Copying |
|---|---|---|---|---|
| `bosonic` | any hashable mode name | an input, `Prepare`, `PrepareState`, `FockPrepare`, `GKPPrepare` | `Homodyne`, `PhotonCount`, `Discard`, or an output | never: a mode is linear |
| `real` | the record key | `Homodyne` (a real outcome with a probability **density**), `PhotonCount` (an integer with a **probability**), `ClassicalSource`, `ModularDecode` | read by `Feedforward`, `Postselect`, `ModularDecode` | allowed: a record may be read many times |

`CVDiagram.validate()` runs on construction. It raises `DiagramTypeError` (a `ValueError`)
when:

- a box uses a bosonic wire that is not alive (never created, or already measured);
- a bosonic port is given a record key;
- a box reads a record before it is produced, or a record key collides with a mode;
- a wire is created twice or a record is written twice;
- the declared `outputs` differ from the wires that are still alive.

Box constructors also check their parameters: finite values, a symmetric covariance that
obeys the uncertainty principle V + iΩ ≥ 0, a symplectic `Symplectic.S`, a completely
positive `GaussianChannelBox`, 0 ≤ η ≤ 1 for loss, and 0 < η ≤ 1 for homodyne efficiency.

## Hybrid Gaussian semantics

`evaluate(d)` returns a `HybridGaussian`. Every live coordinate is an affine function of the
input quadratures plus an independent Gaussian part:

    z = A x_in + b + ξ,        ξ ~ N(0, Y).

The coordinates `z` are the (q, p) Wigner variables of each live mode and the value of each
classical record. They are labelled in `coords` as `("q", mode)`, `("p", mode)` or
`("r", key)`.

How each box acts on (A, b, Y):

| Box | Action |
|---|---|
| Preparation | appends new coordinates with the given mean and covariance |
| Gaussian unitary (S, d) | z_modes → S z_modes + d |
| `GaussianLoss(η, n̄)` | z → √η z, Y += (1 − η)(2n̄ + 1) I |
| `GaussianChannelBox(X, Y, d)` | z → X z + d, Y += Y_box |
| `Homodyne(θ, η, σ²)` | replaces the two coordinates of the mode by one record h·(q, p), h = (cos θ, sin θ), with variance h V h + (1 − η)/η + σ² |
| `ClassicalSource(μ, v)` | appends a record N(μ, v) |
| `Feedforward` | adds Σ_k (a_k, b_k) m_k plus a constant to (q, p): a linear map on the joint coordinates |
| `Discard` | drops the mode's coordinates |

The homodyne step is exact. The joint law of the record and the remaining Wigner variables is
the Wigner marginal over the conjugate quadrature, and that marginal is Gaussian. A
non-Gaussian box makes `evaluate` raise `NotImplementedError`; use
[the Fock semantics](non-gaussian.md) instead.

## The three readings

`HybridGaussian` gives three views of the same object.

| Method | Returns | Meaning |
|---|---|---|
| `.channel()` | `(X, Y, d)` with x_out = X x_in + d + noise(Y) | The **outcome-averaged** channel on the outputs. Records are discarded. |
| `.unitary(atol=1e-10)` | `(S, d)` | Only when there are no records, no added noise and the same wire set; otherwise it raises `ValueError`. |
| `.conditional(outcomes, mean_in=None, cov_in=None)` | `(mean, cov, density)` | The **conditional instrument**: the output state given every record, and the joint record density. Gaussian conditioning (Schur complement) of the joint law. |

`joint(mean_in, cov_in)` returns the joint mean and covariance of (outputs, records) for a
Gaussian input. The defaults are zero mean and vacuum covariance I. `conditional` needs a
value for every record. Its `density` is a probability **density** in record units, not a
probability. With no records it returns `density = 1.0`.

Module-level shortcuts: `channel_semantics(d)`, `unitary_semantics(d)` and
`conditional_semantics(d, outcomes, mean_in, cov_in)`.

## Example

The teleportation step of the [introduction](introduction.md), conditioned on m = 0.7:

```python
mean, cov, density = evaluate(step).conditional(
    {"m": 0.7}, [0.4, -0.2], [[1.3, 0.2], [0.2, 0.9]]
)
print(np.round(mean, 6).tolist(), round(density, 6))
# [-0.023863, 0.449747] 0.18752
```

[Tutorial 1](examples.md#1-build-a-gaussian-diagram) checks this against a postselected
trajectory of the executable pattern on the Gaussian backend. They agree to 8e-17.

## Composition

| Operation | Meaning |
|---|---|
| `a.then(b)` | sequential: `b.inputs` must equal `a.outputs` (ordered), and the record keys must be disjoint |
| `a.tensor(b)` | parallel: wires and records must be disjoint |
| `d.permute_outputs(order)` | reorders the outputs |
| `compose_channels(first, second)` | (X₂X₁, X₂Y₁X₂ᵀ + Y₂, X₂d₁ + d₂) on the same modes |
| `is_completely_positive(X, Y, tol=1e-9)` | Y + iΩ − iXΩXᵀ ≥ 0 (ħ = 2) |

The test suite checks associativity, identity, tensor and permutation laws, and checks that
`channel()` of `a.then(b)` equals `compose_channels` of the parts.

## Serialisation

`to_dict()`, `to_json()` and `CVDiagram.from_dict()` round-trip every generator except
`GKPPrepare`, which holds a code object (`from_dict` raises `NotImplementedError`).
`fingerprint(digits=10)` is a canonical hash of the box sequence. It rounds parameters to
`digits` and is used to deduplicate search candidates.

## Independence of the semantics

`evaluate` is a separate implementation. The tests compare it with
`photographiq.analysis.gaussian_channel` (source-vector propagation of a `Pattern`) and with
postselected trajectories on the Gaussian backend (sequential Schur complements). See
[Rewrite verification](rewrite-verification.md).
