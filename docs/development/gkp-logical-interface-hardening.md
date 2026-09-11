# GKP logical interface hardening

## Outcome and execution boundary

Version 0.3.1 adds finite encoded resources, physical X/Z readout, structured
decoding, logical frames and general finite-outcome instrument infrastructure.
It does not enable `GKPBridge.run()` or claim a full physical MuTA implementation.
Both repositories were audited locally. PhotoGraphiQML's existing ideal logical
execution and finite-resource diagnostics are distinct from a physical execution
path, and its existing refusal remains appropriate.

Audit baselines were PhotoGraphiQ `f15957d297f65f1e4761007e189f11119282dfdd`
and PhotoGraphiQML `b3b26078d5cc30fb1fad5a1f71bb714169d44639`. The sibling
working tree remains unchanged. The new work is local to PhotoGraphiQ.

| Downstream request | Current status |
| --- | --- |
| A finite GKP X/Z measurement step, including after CZ | Implemented and tested |
| Pauli-basis MuTA | Not enabled; Y and complete circuit/output policy remain missing |
| Clifford-angle MuTA | Not enabled; finite Y instrument remains unvalidated |
| π/4 MuTA | Unsupported; injection protocol missing |
| Arbitrary-angle MuTA | Unsupported; general logical rotation protocol missing |

The exact next blocker for the restricted X-only path is downstream adaptive
lowering and a validated multi-mode output/decoding policy with complete-circuit
convergence. For the general path, independently validated logical Rz/injection
is additionally required. A successful single measurement is not that validation.

## Implementation and physics

`GKPCode` delegates comb construction to the existing primitives, retaining
nonorthogonality. Its Gram-aware projector defines subspace leakage independently
of discrimination ambiguity. Dual-basis effects have explicit inconclusive and
outside-code outcomes. `MeasurementInstrument` distinguishes effects, Kraus maps,
retained updates and destructive Pattern updates. Mixed-Fock partial measurement
retains correlations; pure Fock rejects effects that may leave mixed survivors.

Physical Z and X are q and p homodyne with modular decoding at L=√(2π).
Analog likelihoods stay densities, and missing leakage/confidence remains unknown.
The soft decoder is Bayesian only for its declared finite preparation ensemble.
Frames track ideal binary byproducts, not analog noise or finite-envelope repair.
`multimode_readout` computes joint probabilities without multiplying marginals.

The [measurement derivation](../theory/gkp-measurement.md) derives H rotation,
S shear, CZ lattice phases, modular effects and the dual-basis construction.
It distinguishes exact ideal identities from finite physical approximations.
The [decoder derivation](../theory/gkp-decoding.md) specifies ties, posterior
assumptions and distinct leakage meanings. The
[downstream contract](photographiqml-contract.md) records stable exports,
parameters, result semantics and unsupported serialization/autodiff extensions.

## Validation artifacts

`validation/gkp-logical/evidence.json` records gate and measurement studies.
The [validation guide](../validation/gkp-logical-validation.md) describes the
independent oracles and tolerance assertions. Numerical release-gate status is
recorded below; historical v0.3 logs are not evidence for this change.

At width 0.55, envelope 0.5 and cutoff 80, the four tested inputs (zero, one,
plus, minus) give these ranges. Target fidelity compares with the normalized
finite encoded ideal-gate target; leakage uses the actual finite code span.

| Gate | Target fidelity range | Code-subspace leakage range |
| --- | --- | --- |
| X | 0.666053–0.691543 | 0.308421–0.333912 |
| Z | 0.621699–0.631256 | 0.368732–0.368757 |
| H | 0.958902–0.977806 | 0.017374–0.038459 |
| S | 0.619350–0.895240 | 0.102048–0.371014 |

For CZ on finite plus/plus at width 0.6 and envelope 0.55:

| Total cutoff | Independent-reference infidelity | Finite target fidelity | Subspace leakage |
| --- | --- | --- | --- |
| 24 | 5.3932e-5 | 0.570940 | 0.427975 |
| 40 | 1.9080e-6 | 0.569381 | 0.429534 |
| 64 | 5.1677e-9 | 0.569329 | 0.429586 |

The X/Z joint probabilities at cutoff 64 are approximately
(0.439038, 0.060987, 0.060992, 0.438983). Marginals are almost balanced,
so their product would incorrectly predict four probabilities near 0.25.
The last cutoff step changes all tested joint/marginal/fidelity/leakage metrics
by less than 1e-4. Large finite target error persists after numerical convergence.

For finite minus X readout at width 0.55 and envelope 0.5, the error probabilities
at cutoffs 24,48,80 are 0.01209651, 0.01220686, 0.01220681. The final
residual second moment is 0.24695474. Grid and peak refinements agree to numerical
precision for this resource; that does not establish a universal resource regime.

The optional `python experiments/check_photographiqml_gkp.py` check passed against
the sibling checkout: bridge resource amplitudes agree with `GKPCode`, the signed-X
primitive executes, and the full bridge still refuses execution. No MentPy is
imported by this check or by the PhotoGraphiQ package/tests.

## Final local release gates

All runs below used Windows and the same numerical source/test SHA-256:
`b7c9b56f402e047ece69109fd0ca56e79a8bfdf88ce7a0e3c1d0c6e9602d38f1`.
The baseline Git HEAD is recorded separately because these changes are uncommitted.
These are executed local matrix results, not claims of new remote GitHub CI runs.

| Python | Tests | Line coverage | Ruff / format / mypy | Wheel + sdist | Scientific evidence |
| --- | --- | --- | --- | --- | --- |
| 3.11.16 | 533 passed | 92.34% (3060/3314) | Passed | Passed | Passed |
| 3.12.14 | 533 passed | 92.34% (3060/3314) | Passed | Passed | Passed |
| 3.13.15 | 533 passed | 92.34% (3060/3314) | Passed | Passed | Passed |
| 3.14.4 | 533 passed | 92.21% (3007/3261) | Passed | Passed | Passed |

There are 99 new tests. The coverage denominator follows each interpreter's
measured executable statements. The explicit Python 3.12 `--fail-under=90`
coverage gate passed. Per-version environments, exit codes, test XML and logs are
under `validation/gkp-logical/python-3.*`; `summary.json` records the consolidated
counts, coverage and operation boundary. Existing release checks were not weakened.

The strict MkDocs build and public API checker passed. All 37 tutorial scripts
(including 13 new encoded tutorials), ten projects and five notebooks completed:
52 successful executions are recorded in `validation/gkp-logical/examples.log`.
The new tutorials include goal, theory, executable code, captured output,
validation, convergence and limitations. Their index is
[Encoded GKP tutorials](../tutorials/gkp-logical-index.md).

Reproduce the matrix by invoking
`python experiments/validate_release.py --output PATH` with each interpreter.
Run `python scripts/check_docs_examples.py --projects --notebooks`,
`python -m mkdocs build --strict`, and
`python -m experiments.gkp_logical_evidence` for the additional documentation
and encoded evidence artifacts. Builds produced PhotoGraphiQ 0.3.1.

## Deferred capabilities

Physical Y is deliberately unsupported despite a derived S†/X candidate.
`LogicalMeasurementSynthesis` refuses injection: no optical magic-state protocol
has been invented or silently substituted. No compiled executor was added because
the audit did not establish an expensive reusable compilation stage; resource
codeword caching and existing trajectory batches remain available. Ordered JAX
parameter batching covers only previously supported fixed-outcome operations.
No stochastic encoded MBQC differentiation is claimed.

The additive interfaces preserve the existing dependency range and root `Parameter`
export. Existing release gates and the 90% coverage floor are unchanged.
Pages setup remains **Settings → Pages → Build and deployment → Source: GitHub
Actions**. No software change attempts to repair that repository setting.
