# Rewrite verification

A rule's proof is in [the theory note](../../theory/CV_ZX_FOUNDATIONS.md). The code adds a
second, semantic check: any two Gaussian diagrams can be compared exactly, and the
optimiser compares every candidate with the original before accepting it.

## `compare`

```python
compare(a, b, *, atol=1e-9, rtol=1e-9) -> EquivalenceReport
```

Both diagrams must be Gaussian. They must have the same input wires (same order) and the same
number of outputs. Outputs are matched by label when the label sets agree, and by position
otherwise; the report says which. Equivalence is reported in three separate layers that are
never merged:

| Field | Layer | Condition |
|---|---|---|
| `logical_equivalent` | ideal map | max\|X_a − X_b\| ≤ tol and max\|d_a − d_b\| ≤ tol |
| `physical_equivalent` | averaged channel | logical, and max\|Y_a − Y_b\| ≤ tol |
| `conditional_equivalent` | full instrument | the joint affine law (A, b, Y) of outputs and records agrees, record by record. `None` when the record keys differ. |

The tolerance is `tol = atol + rtol * scale`, with scale = max(1, max\|X_a\|, max\|Y_a\|). The
report also holds the residuals `channel_X_residual`, `channel_Y_residual`,
`displacement_residual` and `instrument_residual`. It holds `symplectic_residual`
(max\|X Ω Xᵀ − Ω\|, NaN when the input and output counts differ), a
`measurement_record_mapping` string, the `tolerance` and an `assumptions` string.
`to_dict()` serialises the report.

All comparisons are exact linear algebra on the semantic matrices. Nothing is inferred from
sampled inputs.

## Independent references

`compare` uses the same `evaluate` for both sides. The tutorials and the tests also check
against code paths that do not use `evaluate`:

| Reference | Checks | Where |
|---|---|---|
| `analysis.gaussian_channel(to_pattern(d))` | averaged channel of the executable pattern | tests, R61, tutorials [1](examples.md#1-build-a-gaussian-diagram), [3](examples.md#3-verify-a-rewrite), [6](examples.md#6-recompile-and-execute) |
| `pg.simulate(..., measurement_outcomes=...)` on the Gaussian backend | conditional mean, covariance and record density (sequential Schur complements) | tests, R61, tutorials 1, 3 and 6 |
| Dense matrix exponentials in a truncated Fock space | the Weyl phase recorded by displacement fusion; phase-polynomial rules | R61, R68, tutorials [3](examples.md#3-verify-a-rewrite) and [8](examples.md#8-non-gaussian-diagram) |
| `GaussianGraph` rules (MFvL) | conditioning on q and p homodyne records | tests, R62, [tutorial 4](examples.md#4-pattern-to-diagram) |

## A false rule is caught

Swapping a rotation and a squeezer on the same wire is not a valid rewrite. `compare` reports
it:

```python
from photographiq.calculus.cv_diagrams import CVDiagram, Rotation, Squeezer, compare

right = CVDiagram((0,), (Rotation(0, 0.7), Squeezer(0, 0.4)), (0,))
wrong = CVDiagram((0,), (Squeezer(0, 0.4), Rotation(0, 0.7)), (0,))
report = compare(right, wrong)
print(report.logical_equivalent, round(report.channel_X_residual, 3))
# False 0.529
```

## Logical is not physical

A logical rewrite keeps (X, d) but changes Y and the records. `compare` says so explicitly.
From [tutorial 3](examples.md#3-verify-a-rewrite), resynthesising two compiled rotations
(r = 1):

```text
resynthesis: 8 -> 3 steps; logical True, physical False (Y residual 0.458); records differ: 8 -> 3; conditional equivalence not defined
```

## Recorded evidence

**R61** (`paper_experiments/22_cv_diagrams/61_gaussian_rewrites.py`, status `pass`) applied
each of the ten Gaussian rules (rotation, squeezing, shear and displacement fusion, identity
removal, symplectic fusion, channel fusion, feed-forward merge, CZ + q-homodyne, preparation +
homodyne) to 40 random seeded instances. It compared each result with the original's
executable pattern (`gaussian_channel`) and, for the measurement rules, with postselected
Gaussian-backend trajectories. Worst residuals: semantics 8.7e-16, record density 8.8e-16,
Weyl phase 1.9e-16 (dense Fock reference, cutoff 120).

**R68** (`68_non_gaussian_and_gkp.py`, status `pass`) ran the phase-polynomial rules on the
Fock engine against a dense matrix-exponential reference (cutoff 150). At cutoff 70 the
original, fused and fused-plus-propagated diagrams have infidelity at most 5.1e-11 for input
D(0.5)\|0⟩ and at most 2.4e-8 for D(1.5)\|0⟩, and the infidelity decreases with the cutoff.
A false rule (commuting exp(i f(q)) with a p displacement) stays at 1.8e-2 and 4.6e-2. This
finite-cutoff agreement is evidence for identities that are proved analytically. It is not a
proof.

Test files: `tests/calculus/test_cv_diagrams.py` and
`tests/calculus/test_cv_diagrams_optimize.py`.
