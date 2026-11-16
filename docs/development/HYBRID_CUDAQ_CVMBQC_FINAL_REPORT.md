# Hybrid Gaussian–Fock CV-MBQC with the CUDA-Q Dynamics adapter: final report

## Summary

**Delivered: an experimental hybrid executor.** It runs complete CV-MBQC patterns, with
Gaussian and non-Gaussian preparations, gates, loss, physical homodyne and photon-counting
instruments, and causal feed-forward. States use CUDA-Q's per-mode truncated
representation, and all of this is validated on CPU against independent engines. Its
primary acceptance experiment passes: three-mode cubic-phase CV-MBQC against native-Fock
with cutoff sweeps (HYB-05).

**Not delivered: real CUDA-Q execution.** The CUDA-Q runtime milestone is **BLOCKED**:
- CUDA-Q is Linux-only;
- the machine runs Windows without WSL;
- installing WSL is a system-wide change and was not made.

`engine="cudaq"` is implemented, never falls back to NumPy, and its integration tests are
kept, but **it has never executed**. There are no GPU results.

## 1. Starting repository revision

- Branch: `upgraded`.
- HEAD: `9dea4abf59dae4680fe0d30f6b37d367e1098f1c` ("original CV-MBQC and CV-ZX").

All work is local, unstaged and uncommitted. No branch was switched, created, staged,
committed, pushed or reset.

## 2. Starting Git status

The working tree was clean at the start: `git status --short` printed nothing.

## 3. Existing functionality reused

| Reused component | Where it is used |
|---|---|
| `photographiq.cudaq_dynamics` gate generators | every gate in the executor and the bridge's symplectic generators |
| `to_matrix`, `per_mode_to_total_photon` / `total_photon_to_per_mode` | `FockRep` conversions |
| `reference_evolve`, `evolve`, `_require_cudaq` | CUDA-Q engine; reference for thermal loss |
| `Pattern` IR, `Pattern.validate()` causality, `expressions.resolve` | feed-forward |
| `GaussianInput`, `FockInput`, `CubicPhaseResource`, `non_gaussian.photon_subtraction` | preparations and benchmarks |
| Gaussian, native-fock, native-mixed-fock and piquasso-fock backends | validation references |
| `paper_experiments/common.py` and `22_cv_diagrams/cvzx_support.py` | experiment saving and figure style |
| `NonGaussianProgram.pattern` | PhotoGraphiQML forward hook |

## 4. New modules and APIs (experimental, not registered as a backend)

| Module | API |
|---|---|
| `photographiq.hybrid.states` | `FockRep` (contract, `reduced`, `moments`, `boundary_population`), `to_total_photon`, `from_total_photon`, `williamson`, `gaussian_to_fock`, `moment_matched_gaussian` (labelled approximate), `fock_input` |
| `photographiq.hybrid.homodyne` | `quadrature_wavefunctions`, `density`, `window`, `measure`, `HomodyneOutcome` |
| `photographiq.hybrid.planner` | `classify`, `plan`, `ExecutionPlan`, `Segment` |
| `photographiq.hybrid.executor` | `run`, `run_shots`, `ensemble_statistics`, `HybridResult`, `ENGINES` |
| `photographiqml.cv.hybrid` | `hybrid_moments` (forward inference only) |

The public `photographiq` / `photographiqml` namespaces, the backend registry and
`simulate()` are unchanged.

## 5. Mathematical derivations

Conventions: ħ = 2, q = a + a†, p = −i(a − a†), vacuum covariance I, interleaved (q, p),
gates U = exp(−iH).

- **Symplectic generator.** With G = ½xᵀAx and [x_j, x_k] = 2iΩ_jk, the Heisenberg action is U†xU = exp(2ΩA)x. Choosing A = −½ΩL realises S = e^L. Each factor of the polar decomposition S = PQ has a real logarithm.
- **Williamson.** K = V^{−1/2}ΩV^{−1/2} is antisymmetric. Its real Schur form QᵀKQ has blocks [[0, 1/ν], [−1/ν, 0]] once oriented, which gives S = V^{1/2}QD^{−1/2} and V = S diag(ν)Sᵀ.
- **Homodyne.** ⟨x_θ|n⟩ = e^{−inθ}ψ_n(x), with ψ₀ = (2π)^{−1/4}e^{−x²/4} and ψ_{n+1} = (xψ_n − √n ψ_{n−1})/√(n+1). This holds because x_θ = R(−θ)†qR(−θ) with PhotoGraphiQ's R(φ) = exp(iφn). The density is p(x) = ‖ψ̃(x)‖², and for a mixed state ρ̃(x) = ⟨x_θ|ρ|x_θ⟩.
- **Bins.** P(B) = ∫_B p and ρ_B = ∫_B p(x)ρ(x)dx / P(B). The error relative to the ideal instrument is O(w²) (midpoint expansion; measured slope 1.99).
- **Loss.** The Lindbladian with L = √(1+n)a and √n a† at t = −ln η gives V → ηV + (1−η)(2n+1)I. For n = 0 the truncated space is invariant under a, so expm(t𝓛) equals the binomial Kraus channel exactly.
- **Rare-outcome amplification.** The conditional state ψ̃(x)/√p(x) carries the absolute truncation error δψ̃ as a relative error δψ̃/√p(x). This motivates the `conditional_boundary` diagnostic ‖ψ̃_top(x)‖²/p(x).

## 6. Gaussian-to-Fock conversion (HYB-02)

The bridge reproduces moments, with errors decreasing geometrically in d. At the largest d:

| State | Max covariance error | Lost weight |
|---|---|---|
| 1-mode pure squeezed coherent | 6.7e-8 | 4.6e-10 |
| 1-mode mixed | 7.7e-13 | 3.2e-15 |
| 2-mode pure | 9.8e-7 | 5.9e-9 |
| 2-mode mixed | 2.3e-5 | 1.6e-7 |

Normalisation is explicit: `retained_weight` records the truncated probability, and every
mixed output is positive semidefinite.

**Deviation:** the first run stopped the two-mode sweeps at d = 12 and 16 and failed the
declared 1e-4 threshold because of truncation. The sweeps were extended to 24.

## 7. CUDA-Q runtime validation

**BLOCKED.** Evidence is in `docs/validation/CUDAQ_REAL_RUNTIME_VALIDATION.md` and
`HYB01_cudaq_real_runtime.json`.

- `pytest tests/cudaq_adapter`: 28 passed, 1 skipped (the real-CUDA-Q test).
- The executable integration tests are kept and skip-guarded:
  - `test_real_cudaq_evolve_matches_reference`;
  - `test_real_cudaq_engine_matches_numpy_reference` (three-mode cubic MBQC, plus lossy teleportation on CUDA-Q).
- Acceptance criterion A is therefore **not met**.

## 8. Homodyne instrument validation (HYB-03, HYB-09, HYB-10)

| Check | Result |
|---|---|
| closed forms (vacuum, coherent, squeezed, number n = 1–4) | < 1e-9 |
| Gaussian conditional states (pure and lossy, 7 outcomes) | moments ≤ 2.3e-5, densities ≤ 1e-7 |
| native-fock non-Gaussian conditional states | density ≤ 9e-8, infidelity ≤ 6e-9 |
| bins | O(w²) |
| sampling | KS 0.010 < 0.023, tail mass < 1e-10, reproducible seeds |

## 9. Feed-forward validation (HYB-04, tests)

Corrections are evaluated on the actual records, and `Pattern.validate` enforces
causality.

| Check | Result |
|---|---|
| fixed-outcome teleportation vs Gaussian at d = 28 | mean 2e-7, cov 3.4e-6 |
| 200 sampled shots | KS passes (0.058 < 0.115); per-shot conditional states at d = 32 ≤ 1.5e-4 |
| gains | g·m shift reproduced |
| replay of sampled records | identical state |

**Recorded failure:** the per-shot criterion at d = 20 fails (max 1.1e-2 at a 3.7σ
outcome; 99th percentile 3.3e-3). This led to the `conditional_boundary` diagnostic,
which ranks the bad shots (Spearman 0.85).

## 10. Complete pattern execution

`run` executes complete patterns with:
- Gaussian, Fock and cubic-resource preparations;
- all Gaussian gates, Kerr and cubic phase;
- `Loss`;
- homodyne (ideal or binned, postselected or sampled);
- photon counting;
- outcome-valued gates and homodyne angles;
- output reordering and reduction.

`HybridResult` carries the state, records, raw outcomes, densities and probabilities
(conditional statistics), `log_likelihood`, retained weight, tail mass, truncation
diagnostics, per-operation execution record, plan, runtime and engine. `run_shots` and
`ensemble_statistics` give sampled ensembles with standard errors, labelled as estimates.

## 11. Non-Gaussian benchmarks

**HYB-05 (primary): three-mode cubic-phase CV-MBQC.** The pattern has a coherent input,
CZ-weighted wire, a cubic resource, two dependent homodynes, and quadratic and nonlinear
feed-forward. It is compared with native-fock using cutoff sweeps (piquasso-fock is
recorded too). All 4 cases pass:

| Case | Hybrid 1 − F | Native floor | Density relative error |
|---|---|---|---|
| γ = 0.15, three outcome pairs, d = 28 | 3.1–6.3e-7 | 0.7–1.2e-7 | ≤ 1.1e-5 |
| γ = 0.3, d = 40 vs native 72 | 1.7e-5 | 4.2e-6 | 3.8e-5 |

Two deviations are recorded:
- the native cutoff sweep was extended because native's norm guard rejected low cutoffs;
- the γ = 0.3 sweeps were extended after the 48 / 28 run failed the density criterion while native itself was unconverged.

**HYB-07: Kerr-enhanced MBQC.** κ = 0.05, 0.1, 0.2 and η = 1, 0.85:
- trace distance to native ≤ 2.7e-4, within 3× the native engine's own cutoff change;
- the effect size is ≥ 740× the error.

**HYB-08: photon counting with a bridged squeezed input.** At d = 28, |ΔP| ≤ 6e-11 and
P·(1 − F) ≤ 2e-11. **Deviation:** the first criterion failed for rare counts at d = 20.

## 12. Lindblad noise results

| Check | Result |
|---|---|
| Λ vs exact Kraus channel | < 1e-10 |
| thermal Λ vs Lindblad reference | < 1e-8 |
| HYB-06 lossy teleportation vs Gaussian (η = 0.9 / 0.7 / 0.5; η = 0.8 with n = 0.1), d = 24 | mean ≤ 7.6e-7, cov ≤ 1.2e-5, density ≤ 7.4e-8 |
| lossy Kerr vs native-mixed-fock | ≤ 2.1e-4 |

**Bug found and fixed during validation:** `_apply_channel` used the inverse axis
permutation, which was wrong whenever the lossy mode was not first. Its density was
0.278 against native's 0.367. Loss on the unmeasured mode exposed it.

## 13. Fock cutoff convergence

Dimension sweeps appear in HYB-02, -04, -05, -06, -07 and -08:
- errors fall geometrically for Gaussian-like states;
- the strong cubic phase (γ = 0.3) converges slowly and non-monotonically in both engines;
- binned measurement converges as O(w²).

## 14. Numerical error analysis

1. **Truncation.** Gates use truncated-generator semantics, the same as CUDA-Q, and are exactly unitary on the truncated space. The norm therefore does not detect truncation. Two diagnostics are reported:
   - `max_boundary_population`, with a warning above 0.02 as in the native engines;
   - `max_conditional_boundary`, which catches rare outcomes.
2. **Different truncations.** Hybrid (per-mode) and native (total-photon) agree only in the converged limit. Acceptance thresholds are therefore tied to native's own cutoff-to-cutoff change.
3. **Sampling.** The 4001-point inverse CDF and the tail mass are reported.
4. **Bins.** 64-point midpoint rule.
5. **Thermal loss.** Uses the truncated a†.

## 15. Performance results (HYB-12, CPU only)

On this CPU, hybrid NumPy is slower than native:

- three-mode cubic pattern: 0.76 s (d = 28) vs 0.11 s (native cutoff 48), at matched accuracy;
- two-mode lossy pattern: 0.28 s (d = 24) vs 0.16 s (native-mixed cutoff 30).

The planner's memory estimate was a lower bound in every run. **No GPU or CUDA-Q
performance was measured, and no acceleration is claimed** (HYB-11 is blocked).

## 16. Actual device / target information

Every execution in this report ran on CPU through `numpy-reference`, and each execution
record states `{"engine": "numpy-reference", "device": "cpu"}`.

- OS: Windows 11 10.0.26200.
- Python: 3.12.14.
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, driver 581.86, 6141 MiB. The GPU was **not used**.
- CUDA-Q target: none, because CUDA-Q is not installed.

## 17. Test pass/fail/skip counts

| Suite | Result |
|---|---|
| `tests/cudaq_adapter` | 28 passed, 1 skipped (real CUDA-Q) |
| `tests/hybrid` | 26 passed, 1 skipped (real CUDA-Q) |
| full suite (`pytest -q`) | 1842 passed, 14 skipped, 0 failed (10 min 39 s). The skips are: mentpy and playwright missing; capability-gated cross-backend cases; Piquasso cross-Kerr; the two real-CUDA-Q tests. |
| `scripts/check_public_api.py` | 88 exports and API imports OK |
| `experiments/validate_release.py` (output to a temporary directory) | "All release gates passed": pytest (1842 passed, 14 skipped), ruff check, ruff format, mypy (113 files), build (sdist and win_amd64 wheel), physics evidence |
| `ruff check` | clean |
| `ruff format --check` (src, tests) | clean |
| `mypy src/photographiq` | 89 files, no issues |

## 18. Documentation produced

- `docs/hybrid-cudaq/`: architecture, installation, state-representations, gaussian-to-fock, dynamics-execution, homodyne-instruments, feedforward, trajectories, noise, validation, performance, limitations. The examples were executed.
- `docs/validation/CUDAQ_REAL_RUNTIME_VALIDATION.md`.
- `docs/development/HYBRID_CUDAQ_BASELINE_AUDIT.md`.
- This report.
- `paper_experiments/23_hybrid_cudaq/README.md`, which maps the suggested HYB list onto these IDs.
- `mkdocs.yml` navigation, plus a cross-link in `docs/cudaq.md`.

## 19. Files modified

| Change | Files |
|---|---|
| Modified (2) | `mkdocs.yml` (nav), `docs/cudaq.md` (cross-link) |
| New source | `src/photographiq/hybrid/{__init__,states,homodyne,planner,executor}.py`, `src/photographiqml/cv/hybrid.py` |
| New tests | `tests/hybrid/test_hybrid.py`, `tests/hybrid/test_ml_forward.py` |
| New experiments | `paper_experiments/23_hybrid_cudaq/` (7 scripts, support module, README) |
| New outputs | `HYB01`–`HYB12` CSV / JSON / metadata in `paper_experiments/results/`, and figures and captions for HYB02/03/04/05/07/12 in `paper_experiments/figures/` |
| New docs | the documents listed in §18 |

No existing experiment output was deleted or overwritten. MuTA, MuTAKernel,
MuTAClassifier, MuTARegressor, PhysicalMuTA and GKPBridge were not modified.

## 20. Remaining limitations

- Real CUDA-Q execution, GPU execution and performance are not demonstrated.
- The executor is not a registered backend.
- Unsupported:
  - inefficient or noisy homodyne;
  - cat, GKP and correlated-resource preparations;
  - other measurement types;
  - cross-Kerr (no pattern command).
- No arbitrary-angle physical GKP-MuTA, because no GKP instrument is provided.
- The CPU path is slower than the native engines.
- Convergence must be checked against the outcomes that actually occur (rare outcomes).
- The PhotoGraphiQML hook is forward-only; there is no loss or gradient path.

## Capability classification

| Capability | Classification |
|---|---|
| Gaussian→Fock bridge (pure and mixed, n modes) | **Validated** |
| `FockRep` contract and total-photon mapping | **Validated** |
| Fock homodyne instrument (ideal, binned, postselected, sampled; pure and mixed) | **Validated** |
| Photon counting | **Validated** |
| Feed-forward with outcome-valued gates and causality | **Validated** |
| Lindblad loss, pure (`numpy-reference`) | **Validated** (exact Kraus channel; Gaussian and native-mixed references) |
| Thermal loss | **Validated** (Lindblad reference and Gaussian backend at the tested dimensions) |
| End-to-end hybrid execution, NumPy engine (Gaussian, cubic, Kerr, lossy) | **Experimental**: validated against references; experimental API |
| Planner dry run and memory estimate | **Validated** (the estimate is a lower bound) |
| Truncation diagnostics | **Experimental** (heuristic ranking, not an error bound) |
| PhotoGraphiQML forward inference (`hybrid_moments`) | **Experimental** (one validated program) |
| Fock→Gaussian moment matching | **Restricted** (explicitly approximate) |
| `engine="cudaq"` (unitary and Lindblad on CUDA-Q) | **Implemented but unverified** |
| GPU execution and acceleration | **Unsupported here** (not measured) |
| Inefficient homodyne, cat, GKP and correlated resources, other measurements | **Unsupported** |
| Arbitrary-angle physical GKP-MuTA | **Unsupported** |

## Acceptance criteria

| Criterion | Status |
|---|---|
| A. CUDA-Q runtime | **Not met: BLOCKED** (no CUDA-Q execution) |
| B. State conversion | Met (HYB-02) |
| C. Homodyne measurement | Met (HYB-03, -09, -10) |
| D. Classical feed-forward | Met (HYB-04, -05, tests) |
| E. End-to-end MBQC vs native-Fock | Met on the NumPy engine (HYB-05) |
| F. Numerical convergence (cutoff and resolution) | Met (sweeps; HYB-09) |
| G. Reproducibility (revision, environment, seeds, tolerances) | Met (JSON `source_revision`, metadata, configuration, declared criteria) |
| H. Backward compatibility | Met: the full suite passes (1842 / 0 failed), the public API check passes, and the registry and the ML models are unchanged |
