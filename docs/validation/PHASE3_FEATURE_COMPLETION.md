# Phase 3: Feature completion

Audit date 2026-11-12, local working tree of branch `upgraded`. Each feature lists its public
API, an executable example or script, tests, failure behaviour and documentation.

## 3.1 CUDA-Q Dynamics adapter: implemented, CUDA-Q execution unverified

- **What changed:** the former `cudaq-dynamics` pattern backend, which was registered but always raised, has been removed from the registry. The registry no longer contains any GPU-device entry. `select_backend` cannot pick CUDA-Q, and asking for `"cudaq-dynamics"` explains that CUDA-Q is an adapter.
- **New module** `photographiq.cudaq_dynamics`, an optional Hamiltonian-level adapter:
  - `BosonicHamiltonian`, a normal-ordered ladder-operator algebra;
  - generators for nine gates (rotation, displacement, squeezing, beam splitter, CZ, quadratic phase, Kerr, cross-Kerr, cubic phase) in PhotoGraphiQ conventions;
  - `to_matrix`;
  - `per_mode_to_total_photon` / `total_photon_to_per_mode`, an explicit mapping between CUDA-Q per-mode truncation and PhotoGraphiQ total-photon truncation;
  - `reference_evolve` (SciPy, unitary and Lindblad);
  - `to_cudaq` / `evolve`, which translate to and call the documented CUDA-Q 0.16 API, with a lazy import, a version check, an operator-ordering check before running, an execution record, and no fallback.
- **Validated:** each generator against native-fock to at most 1.4e-13, with the global phase aligned and sign-flipped generators failing. Lindblad loss matches the native loss channel to 6e-17. Tests in `tests/cudaq_adapter/` (28 passed).
- **Not verified:** any actual CUDA-Q call. CUDA-Q ships Linux wheels only and this machine has no WSL or Docker. The real-CUDA-Q test, `test_real_cudaq_evolve_matches_reference`, is skipped here. A fake-module test checks only the call wiring.
- **Not provided:** homodyne conditioning and MBQC pattern execution on CUDA-Q.
- **Docs:** [CUDA-Q](../cudaq.md).

## 3.2 GPU Fock acceleration: implemented for batched workloads

- **Profile:**
  - pure-state patterns are dominated by Python overhead and gate-matrix construction;
  - mixed-state patterns by `apply_mixed` (1.07 of 1.16 s for a 4-mode wire at cutoff 16);
  - the C++ pure kernel costs 3–5 ms at dimension 42 504.
- **Implemented** in `photographiq.gpu.fock`:
  - `apply_operator` gives P U P for batched pure states (B, D) or density matrices (B, D, D), as one cuSPARSE SpMM with a cached CSR structure built from the native basis and ranks;
  - `apply_phase` handles diagonal gates;
  - device `"gpu"` or `"cpu"`, with no fallback, complex128 or complex64, and an `.execution` record.
- **Validation:** against the native `apply_pure`, `apply_mixed` and `phase_*`. The maximum deviation is 7.2e-16 (GPU, complex128) and 2.4e-7 (complex64). Norm and trace agree to 3.3e-16, and density matrices are Hermitian to 1e-14. 29 tests in `tests/gpu/test_gpu_fock.py`.
- **Measured** (RTX 4050, `benchmarks/gpu/results/gpu_fock_20261111-232613.json`, shared machine, so factors below about 2× are noise):

  | Case | Kernel only vs native | End to end vs native |
  |---|---|---|
  | Pure, B ≥ 32 | 1.7–150× | 1–13× |
  | Pure, B = 1, D = 465 | 0.05–0.25× | 0.04–0.2× (GPU slower) |
  | Pure, B = 1, D = 42 504 | 3.6–8.9× | 2.6–3.2× |
  | Mixed, B = 1–8 | 1.3–14× | 0.9–6.9× |

  Transfers dominate end-to-end time for large batches. Phase gates gain nothing end to end over NumPy. Single small states are slower on the GPU.
- **Docs:** [GPU](../gpu.md).

## 3.3 GPU sampling acceleration: measured, not implemented

Batched CuPy permanents, hafnians and torontonians were compared with the native kernels.

| Case | Native time ÷ GPU time |
|---|---|
| Permanent, n = 4, batch 10 000 | 45× |
| Permanent, n = 16, batch 20 | 1.3× |
| Permanent, n = 20 / 24, single | 0.33× / 0.20× (GPU slower) |
| Torontonian, n = 16, single | 0.85× |

The batched gains mostly reflect native per-call overhead, not arithmetic, and no current
workload evaluates large batches of same-size matrices. Nothing ships; a native batched kernel
should come first (see [GPU](../gpu.md)).

## 3.4 Restricted CV graphical calculus: implemented and validated

- **What it is:** the Gaussian-pure-state graphical calculus of Menicucci, Flammia and van Loock (PRA 83, 042335, 2011), in `photographiq.calculus.GaussianGraph`.
- **Representation:** a complex adjacency matrix Z = V + iU (U > 0) plus a tracked mean.
- **Correspondence:** in hbar = 2 conventions the covariance is [[U⁻¹, U⁻¹V], [VU⁻¹, U + VU⁻¹V]], which is 2× the paper's Eq. 2.19.
- **Rules:**
  - unitaries by the linear-fractional update Z' = (C + DZ)(A + BZ)⁻¹ (Eq. 2.24): CZ, phase rotation, squeezing, beam splitter, shear, and any symplectic;
  - q-measurement: delete the vertex;
  - p-measurement: Schur-complement update with tracked conditional mean;
  - tensor product and relabelling.
- **Validated:** against direct Gaussian numerics (`GaussianBackend`), 213 tests, maximum error 2e-14. Also checked: the wire-shortening ideal limit; mixed states, non-symmetric Z and non-symplectic gates are rejected.
- **Not implemented:**
  - general CV-ZX;
  - non-Gaussian elements;
  - mixed states;
  - heterodyne and noisy measurements;
  - any qubit-ZX translation.
- **Docs:** [Gaussian graphical calculus](../user-guide/gaussian-graphical-calculus.md).

## 3.5 CV-flow robustness: implemented and validated

- **Classification:** analysis results are certified, provably impossible (exact rational witnesses via `fractions.Fraction`, or a combinatorial obstruction) or numerically uncertain (a rank decision inside the margin between max(dims)·eps and √eps). Certificates name their regime: `ideal_limit`, `finite_squeezing` or both.
- **New options** on `analyze` and `find_cv_flow`: `shift=` and `exact=`. With `exact=None` (the default), an exact re-check runs automatically on graphs of up to 40 nodes.
- **Statuses unchanged,** with one exception: an uncertain float64 rank decision now gives `unknown` instead of `not_certified`.
- **Tests:** 25 new tests; flow/calculus/Graphix/analytical suites at 383 passed. Graphix agreement on qubit flows is covered by the existing tests.
- **Docs:** [CV-flow robustness](cv-flow-robustness.md).

## 3.6 Interactive visualization: verified in a real browser

Checked with Playwright 1.65.1, driving Edge 157 with real mouse, wheel and keyboard input:
- dragging, panning and zooming;
- edits round-tripping through `apply_edits`;
- stage separation (construct, flow, corrections, execution);
- correction arrows equal to `pattern_corrections` and the `Pattern` commands;
- execution steps equal to a real `simulate` trace;
- small and 6×6 layouts;
- static SVG and PDF labels.

All 9 checks pass after four fixes:
- correction labels were struck through by longer arcs;
- outcome text was faded;
- the selection panel went stale after edits;
- the SVG font fell back to Times.

Remaining limits: straight edges can pass under other nodes, outcome labels can overflow
nodes, and the 6×6 grid has 8.8 px labels. Evidence:
[interactive-visualization-browser](interactive-visualization-browser.md), with screenshots in
`assets/interactive/`, and `scripts/check_interactive_browser.py`.

## Other completions during Phases 1–3

- **Native Fock one-mode Gaussian-unitary stability fix** (Phase 2.3; see the Phase 2 audit).
- **`precision="auto"`** for threshold probabilities, with `ThresholdInfo` reporting which path ran.
- **Measurement-angle API defect** fixed in every backend.
- **Native backend options** for `cutoff_convergence` and `measurement_convergence`.

## Still unsupported

- CUDA-Q execution: unverified.
- Pattern execution on CUDA-Q.
- GPU sampling kernels.
- General CV-ZX.
- Exact certification of the finite-squeezing channel. That is a float64 claim; the exact check covers the ideal limit.
- Multi-mode displaced Gaussian-unitary stability bound.
