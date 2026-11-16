# Non-Gaussian diagrams

The non-Gaussian layer is deliberately small. It has a few generators, two rewrite rules,
and a numerical (truncated Fock) semantics. For the underlying engines see
[Non-Gaussian](../non-gaussian.md) and [Mixed Fock](../mixed-fock.md).

## Generators

| Box | Signature | Semantics |
|---|---|---|
| `PhasePolynomial` | `PhasePolynomial(mode, coefficients)` | exp(i f(q)) with f(q) = Σ_k c_k q^k (ħ = 2), acting as p → p + 2 f′(q). This is the 1 → 1 q-spider of Nagayoshi et al., PRR 7, 033141 (2025), Table II, restated in ħ = 2. c₁ is a p displacement 2c₁, c₂ is a shear 4c₂, and PhotoGraphiQ's `CubicPhase(γ)` has c₃ = γ/6. Trailing zero coefficients are dropped. |
| `Kerr` | `Kerr(mode, kappa)` | exp(i κ n²) (PhotoGraphiQ `Kerr`) |
| `FockPrepare` | `FockPrepare(mode, n)` | number state \|n⟩, n a nonnegative integer |
| `PhotonCount` | `PhotonCount(mode, key)` | destructive photon-number measurement. The integer record has a **probability**. |
| `Postselect` | `Postselect(key, value)` | keeps only the branch where record `key` equals `value`. This is not a deterministic map. |

`GKPPrepare` and `ModularDecode` are described on the [GKP](gkp.md) page.

**Converting from Nagayoshi et al.** They use ħ = 1 with x = q/√2. A phase function
g(x) = Σ g_k x^k there corresponds to c_k = g_k / 2^{k/2} here. Their rotation is
q → q cos θ + p sin θ, so `Rotation(θ)` here is their rotation by −θ. Their equalities hold up
to nonzero scalars. Here, scalars that are known are recorded in the rewrite trace, and
equivalence is judged at the density-operator level.

## Rules

| Rule | Statement | Source |
|---|---|---|
| `phase-polynomial-fusion` | exp(i f(q)) exp(i g(q)) = exp(i (f + g)(q)) | Nagayoshi et al., Eq. (70), 1 → 1 case |
| `displacement-through-phase-polynomial` | exp(i f(q)) X(a) = X(a) exp(i f(q + a)), with X(a) = exp(−i a p/2) (a q displacement by a) | Nagayoshi et al., Eq. (79), 1 → 1 case |

Both are exact operator identities on the Schwartz space. `identity-removal` also removes a
constant `PhasePolynomial` and records c₀ as a global phase. There are no rules for Kerr,
photon counting or post-selection, and no rule changes a heralding probability.

## Fock semantics: `run_fock`

```python
run_fock(diagram, cutoff, *, inputs=None, outcomes=None, seed=None, backend=None) -> FockRun
```

`run_fock` executes the diagram once on a native Fock engine. It uses `"native-mixed-fock"`
if the diagram contains `GaussianLoss`, and `"native-fock"` otherwise; `backend` overrides
this.

- `inputs` maps input wires to PhotoGraphiQ input states. The default is vacuum.
- `outcomes` fixes records by key (postselection). Unfixed records are sampled with `seed`.

`FockRun` has these fields:

| Field | Meaning |
|---|---|
| `state` | the engine state reduced to the diagram outputs (`None` if there are none) |
| `records` | every record by key |
| `probability` | product of the probabilities of the discrete (photon-count) records in this run. With `Postselect`, this is the **heralding probability**. |
| `density` | product of the homodyne record densities |
| `retained_norm` | smallest norm retained by any truncated operation |

The output state is normalised. The heralding probability is reported separately in
`probability`, never folded into the state. A `Postselect` box needs its outcome in
`outcomes`; otherwise `run_fock` raises `ValueError`.

`run_fock` supports:

- `Prepare`, `FockPrepare` and `GKPPrepare`;
- all one-mode and two-mode Gaussian unitaries, and one-mode `Symplectic`;
- `PhasePolynomial` up to degree 3, and `Kerr`;
- `GaussianLoss`;
- `Homodyne` (with efficiency and noise) and `PhotonCount`;
- `Postselect`, `Feedforward`, `ModularDecode`, `ClassicalSource` and `Discard`.

`PrepareState`, multi-mode `Symplectic` and `GaussianChannelBox` raise `NotImplementedError`.

!!! warning "Finite cutoff"
    The cutoff is an exclusive total-photon cutoff, and the engine renormalises after every
    operation. It refuses truncation beyond its norm tolerance. Agreement between two
    diagrams at a finite cutoff is **evidence, never a proof** of an operator identity on
    the infinite-dimensional space.

## `cutoff_study`

```python
cutoff_study(diagram, cutoffs, observable, **run_fock_kwargs) -> list[dict]
```

`cutoff_study` evaluates `observable(FockRun)` at each cutoff. Each row has `cutoff`, `value`,
`retained_norm` and `probability`. A cutoff that the engine refuses (a `ValueError`) is
reported as a row with `value = None` and the `refused` message. It is not hidden.

## Example: rules converge, a false rule does not

[Tutorial 8](examples.md#8-non-gaussian-diagram) fuses two cubic phase polynomials, moves a
q displacement through the result, and compares all three diagrams with a dense
matrix-exponential reference:

```text
infidelity vs dense reference at cutoffs (20, 30, 45, 60)
      original: 3.78e-05  1.39e-06  2.18e-08  6.10e-10
         fused: 3.72e-05  1.43e-06  2.25e-08  6.20e-10
   fused+moved: 3.21e-05  1.22e-06  1.88e-08  5.12e-10
    false rule: 9.08e-02  9.08e-02  9.08e-02  9.08e-02
```

The false rule commutes exp(i f(q)) with a p displacement, which is wrong because
[q, p] ≠ 0. Its infidelity does not decrease with the cutoff. R68 records the same behaviour
up to cutoff 70; see [Rewrite verification](rewrite-verification.md#recorded-evidence).

## Example: heralded photon subtraction

```python
import photographiq as pg
from photographiq.calculus.cv_diagrams import (
    BeamSplitter, CVDiagram, FockPrepare, PhotonCount, Postselect, run_fock,
)

alpha, theta = 0.8, 0.3
sub = CVDiagram(
    (0,),
    (FockPrepare("a", 0), BeamSplitter(0, "a", theta), PhotonCount("a", "n"), Postselect("n", 1)),
    (0,),
)
run = run_fock(sub, 24, inputs={0: pg.GaussianInput.coherent(alpha)}, outcomes={"n": 1})
print(run.probability)
# 0.052854319838209124
```

The closed form is μ e^{−μ} with μ = (α sin θ)² = 0.0528543198382. R68 also checks the
output's mean photon number against (α cos θ)².

## Gaussian semantics are refused

`evaluate`, `compare` and `channel_semantics` raise `NotImplementedError` for a diagram that
contains a non-Gaussian box. `d.is_gaussian` tells you in advance. A `PhasePolynomial` of
degree ≤ 2 is Gaussian and is accepted.
