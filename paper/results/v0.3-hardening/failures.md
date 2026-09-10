# Preserved pre-fix failures

Baseline: c97ae9cc7f9fc6d06aff9b9de4dd5d7e6f3c10fa.

## JAX beamsplitter boundary (2026-09-10)

Command: `python -m pytest tests/autodiff/test_boundary_regressions.py -q --tb=short`

Four regressions failed before repair. At angle 0.3 and input |0,cutoff-1>,
the density traces were 1.09 (cutoff 2), 1.1881 (cutoff 3), and 1.41158161
(cutoff 5), instead of one (absolute tolerance 2e-13). The cutoff-2 photon
gradient was 0.6 instead of sin(0.6)=0.5646424733950354 (tolerance 2e-12).

The projected operators on different modes do not commute on the top total-number
sector. `a† b - a b†` was not anti-Hermitian there. `a† b - b† a` is
anti-Hermitian and preserves the complete number sector. This is an algebraic
bug, not a missing infinite-dimensional tail.

A subsequent independent full-state comparison caught a second beamsplitter
defect: the original sign was opposite to the package/Piquasso convention.
For |1,0> and theta=0.3, coherence rho[|10>,|01>] was -0.2823212367 instead
of +0.2823212367. Population-only reductions had hidden the sign. The final
generator is `b† a - a† b`; both trace and coherence regressions are retained.

## Mixed Gaussian tensor preparation (2026-09-10)

Six correlated homodyne regressions failed before measurement when preparing two
default Gaussian inputs. Piquasso/Numba rejected `Tuple(int32, int64)` occupations
on Windows. Native basis integers and locally generated Python integers had been
concatenated without canonicalization. The density adapter now supplies homogeneous
Python-int occupation tuples. The regression retains the two default preparations.

## JAX normalization contracts (2026-09-10)

Two boundary regressions failed with DID NOT RAISE: photon addition to
(|1>+|2>)/sqrt(2) at cutoff 3 silently removed the |3> component; preparing two
(|0>+|1>)/sqrt(2) inputs at cutoff 2 silently discarded 25% of the tensor mass.
A third test showed eager impossible PNR postselection returned NaNs without an
error. Eager invalid normalization now raises; traced invalid branches explicitly
return NaN states. Preparation masses and final boundary population are exposed.
The 1e-3 tensor truncation threshold matches the existing Fock backend policy.

## GKP half-cell tie (2026-09-10)

`decode_shift(3.5*SPACING)` returned residual +1.253314137315499 and odd parity,
contradicting the documented upper-cell tie rule (negative half spacing, even
parity). The test `test_decoder_cells_and_ties[-0.5-4]` failed. Rounding during
division moved the half integer below its mathematical value. The decoder now
compares the outcome to the boundary in the original quadrature coordinate.

## Missing synthesis contract (2026-09-10)

Three new metadata regressions failed: no approximate/order fields, no retained
per-gate synthesis report, and no Kerr approximation warning. These were interface
deficiencies rather than a demonstrated wrong quartic formula. The reports now
distinguish exact ideal primitives, product formulas, and physical finite-resource
injection. The complete quartic commutator has leading amplitude order 1/2 in the
slice count; second-order splitting of the auxiliary B pulse does not change that.

## GKP large-shift cutoff resolution (2026-09-10)

Both q and p correction tests at input shift L+0.15 and outcome L+0.1 missed the
2e-6 accuracy target: maximum mean/density errors were 3.55753e-5 at cutoff 32
and 7.26418e-6 at cutoff 48. Keep both cases and add cutoff 72; do not relax the
target or remove the shift. This separates physical finite-ancilla filtering
from numerical representation error; the final report records the refined result.

## Remote documentation build (baseline run 34528856594)

The baseline remote test matrix succeeded, but the documentation workflow failed
in strict MkDocs: `Snippet at path 'docs/tutorials/outputs/01-first-cluster.txt'
could not be found`. A broad `outputs/` ignore rule excluded all 24 generated
tutorial transcripts, hiding this on the existing local checkout. The exception
for `docs/tutorials/outputs/` makes the real transcripts part of clean checkouts.
The four-feature tests and executable examples had passed in that remote docs run.

## Codecov upload rejection (baseline job 103044353791)

The action step was green because upload failures were nonblocking, but its actual
log said: `Token required - not valid tokenless upload`. This was not a successful
coverage upload. A repository `CODECOV_TOKEN` Actions secret is required. The main
coverage badge is withheld until a successful main-branch upload is verified.

## Python 3.11 JAX type compatibility (run 34536433226)

Python 3.11.16 resolved JAX 0.10.2. All 426 tests passed, but mypy reported
`autodiff.py:25: "Config" has no attribute "x64_enabled"`. This public runtime
property exists (the tests exercised it), but is not declared in that version's
static Config interface. Accessing it via getattr accommodates the dynamic API
without suppressing mypy or removing the double-precision check. A regression
also explicitly disables x64 and verifies that execution is still rejected.
