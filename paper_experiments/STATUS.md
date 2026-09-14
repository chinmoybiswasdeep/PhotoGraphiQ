# Status

Authoritative latest run record for R1-R42, on this checkout
(`manuscript-experiments` branch, python 3.12.14, numpy 2.5.3, scipy 1.18.1,
piquasso 8.0.1, graphix 0.4, jax 0.11.1 -- see `requirements_notes.md`).
Runtimes are wall-clock observations on the machine this suite was built on
and are not comparative performance claims. The existing repository test
suite (`python -m pytest -q` from the repo root) was also re-run after
adding this suite and unmodified `src/photographiq`: **533 passed** in
~341s, confirming nothing in `src/` was touched or broken.

| ID | Implemented | Executed | Passed | Skipped | Reason | Runtime | Output paths |
|----|:---:|:---:|:---:|:---:|--------|---------|---------------|
| R1 | Y | Y | Y | - | - | ~1 min (piquasso backend, 6 alphas x 4000 shots x2 backends) | `results/{csv,json}/R1_wire_teleportation.*`, `figures/*/R1_wire_teleportation.*` |
| R2 | Y | Y | Y | - | - | <1s | `R2_identity_finite_squeezing.*` |
| R3 | Y | Y | Y | - | - | <1s | `R3_rotation_compile_*.*` |
| R4 | Y | Y | Y | - | - | <1s | `R4_squeezing_compile.*` |
| R5 | Y | Y | Y | - | - | ~15s (400 samples) | `R5_random_symplectic_compile.*` |
| R6 | Y | Y | Y | - | - | <1s | `R6_cz_compile.*` |
| R7 | Y | Y | Y | - | - | ~2s | `R7_beamsplitter_compile.*` |
| R8 | Y | Y | Y | - | - | ~3s | `R8_multimode_showcase_*`, `figures/*/R8_multimode_showcase.*` |
| R9 | Y | Y | Y | - | - | ~2s | `R9_compilation_trace.*`, `results/raw/R9_compilation_trace_summary.txt` |
| R10 | Y | Y | Y | - | - | ~5s | `R10_compiler_scaling.*` |
| R11 | Y | Y | Y | - | - | <1s | `R11_adaptive_feedforward.*` + 2 figures |
| R12 | Y | Y | Y | - | - | <1s | `R12_dependency_dag.*` |
| R13 | Y | Y | Y | - | - | <1s | `R13_invalid_causality.*` (2 dead-code error paths noted, see ISSUES_FOUND.md #1) |
| R14 | Y | Y | Y | - | - | <1s | `R14_cvflow.json` |
| R15 | Y | Y | Y | - | - | ~10s | `R15_raw_piquasso_gaussian*.*` |
| R16 | Y | Y | Y | - | - | ~2s | `R16_numpy_analytic_reference.*` |
| R17 | Y | Y | Y | - | - | ~2s (graphix available) | `R17_graphix_structure.*` |
| R18 | Y | Y | Y | - | - | ~1 min (mixed-fock, cutoff=14, 1/15 samples gracefully hit the norm guard) | `R18_gaussian_backends.*` |
| R19 | Y | Y | Y | - | - | ~30s (135 fixed-outcome cases) | `R19_photon_subtraction_probability.*` |
| R20 | Y | Y | Y | - | - | ~1-2 min (cutoff up to 36) | `R20_subtracted_squeezed_state_*` |
| R21 | Y | Y | Y | - | - | ~1-2 min (cutoff=32) | `R21_cat_state.*` |
| R22 | Y | Y | Y | - | - | ~1-2 min (cutoff up to 48) | `R22_cubic_phase_direct.*` |
| R23 | Y | Y | Y | - | - | ~2-3 min (10/18 cells hit the expected finite-cutoff norm guard at high r/low cutoff -- physical, not a bug) | `R23_cubic_injection.*` |
| R24 | Y | Y | Y | - | - | ~3s | `R24_kerr_direct.*` |
| R25 | Y | Y | Y | - | - | ~45s (steps up to 64) | `R25_kerr_synthesis.*` |
| R26 | Y | Y | Y | - | - | ~1-2 min (cutoff up to 96) | `R26_gkp_resource*.*` |
| R27 | Y | Y | Y | - | - | ~1 min (cutoff=48) | `R27_gkp_stabilizers*.*` |
| R28 | Y | Y | Y | - | - | ~9 min (16 seeds x 9 deltas, cutoff=80; redesigned around ensemble/circular statistics -- see ISSUES_FOUND.md #3) | `R28_gkp_syndrome.*` |
| R29 | Y | Y | Y | - | - | ~2s | `R29_finite_squeezing_*` |
| R30 | Y | Y | Y | - | - | <1s | `R30_loss_validation.*` |
| R31 | Y | Y | Y | - | - | <1s | `R31_thermal_noise.*` |
| R32 | Y | Y | Y | - | - | Gaussian sweep <5s; mixed-Fock cross-check ~2-3 min after reducing shot counts to keep the noise!=0 branch (which integrates a dense density matrix per shot) practical -- see ISSUES_FOUND.md and the script's own comment | `R32_noisy_measurement*.*` |
| R33 | Y | Y | Y | - | - | ~1-2 min (4 workloads, cutoff up to 96) | `R33_fock_cutoff.*` |
| R34 | Y | Y | Y | - | - | ~2 min (5x6 grid, cutoff up to 96) | `R34_cubic_convergence.*` |
| R35 | Y | Y | Y | - | - | <1s (analytical) | `R35_fock_dimension_scaling.*` |
| R36 | Y | Y | Y | - | - | ~4 min (200-mode Piquasso backend point dominates: 77s median x5 repeats) | `R36_gaussian_runtime.*` |
| R37 | Y | Y | Y | - | - | ~2s | `R37_compiler_runtime.*` |
| R38 | Y | Y | Y | - | - | ~5s | `R38_piquasso_overhead.*` |
| R39 | Y | Y | Y | - | - | <1s | `R39_serialization.json`, `results/raw/R39_serialization_pattern.json` |
| R40 | Y | Y | Y | - | - | ~1s | `R40_seeded_shots.*` |
| R41 | Y | Y | Y | - | - | ~5s (jax available) | `R41_jax_gradients.*` |
| R42 | Y | Y | Y | - | - | ~10s (jax available) | `R42_stochastic_gradients.json` |

**42/42 implemented, 42/42 executed, 42/42 passed, 0 unsupported.**

`python run_all_safe.py` (the lightweight suite) was run end-to-end as a
final check: **24 ran, 24 passed, 0 failed, 18 skipped** (each with a stated
reason: optional dependency, large Fock, or long benchmark), total wall
time ~244s. See `results/json/run_all_safe_summary.json` for the raw record.

## Genuine findings made while building this suite (see `ISSUES_FOUND.md`)

1. `Pattern.validate()`'s `"Self CZ is invalid"` and `"Duplicate outputs"`
   messages are unreachable for the simplest trigger case (an earlier
   dependency-graph self-loop check raises `"Cyclic command dependencies"`
   first). The pattern is still correctly rejected either way.
2. The native two-mode Fock `GaussianTransform` entangling gate showed
   numerically unstable (super-unity, then wildly divergent) norms for a
   sharp two-mode GKP-scale workload at cutoffs roughly 120-140; correctly
   intercepted by the existing norm-tolerance guard every time, never a
   silent bad result. R28 was sized (cutoff=80, peak_width=0.35) to a
   combination confirmed not to hit this.
3. Mixed-Fock noisy homodyne (`noise!=0`) costs roughly 8-14 seconds PER
   SHOT at cutoff 10-20 on this machine, because it integrates a dense
   density matrix with `scipy.integrate.quad_vec` inside a `brentq`
   root-find for every sampled outcome. This is genuine, reproducible cost
   of an experimental feature, not a hang; R32's shot counts for that branch
   were sized down (6 shots) accordingly, with the fast/exact Gaussian-backend
   sweep (6000 shots, <5s) carrying the statistical weight of that result.
4. Single-shot GKP syndrome-extraction outcomes are dominated by which comb
   peak of the (also finite-energy) ancilla was sampled; only the ENSEMBLE
   (circular) mean over repeated seeds is meaningfully centered on the true
   imposed shift. R28 was redesigned around this once understood.

## Assessment for the manuscript

**Strongest results** (near-machine-precision agreement with an independent
reference, large sample count, or a clean qualitative signature):
R5 (400-sample random symplectic compiler validation), R15/R16 (raw
Piquasso / independent NumPy cross-checks), R19 (exact binomial photon
subtraction probability, 135 cases), R22/R24 (cubic phase / Kerr vs. dense
matrix-exponential and exact phase rule), R29 (finite-squeezing noise floor
scan), R39/R40 (serialization and seed reproducibility), R41 (JAX autodiff
vs. finite differences, 6 gate types).

**Numerically converged** (fidelity to the highest tested reference/cutoff
above 0.999, trend visibly flattening): R33's cat/photon-subtraction/cubic
workloads (though these use low-energy inputs where even the *lowest*
tested cutoff is already near-perfect -- a demanding high-energy input would
show a more informative convergence curve); R33's GKP workload shows a
genuine, visible convergence trend (0.999 to 1.0 over cutoff 24 to 96).

**Illustrative rather than tightly quantitative**: R8/R9 (compilation
showcase and provenance -- descriptive, not a pass/fail metric); R10/R37
(resource/runtime scaling -- trend, not precision); R28 (GKP syndrome
extraction -- correctly centered on average, but with substantial
single-shot noise at the tested finite-energy resource width); R32's
mixed-Fock cross-check (small shot count by necessity).

**Remain explicitly experimental** (per `docs/validation/feature-matrix.md`,
confirmed here): R18/R32 (mixed-Fock backend), R23/R25 (finite-resource
cubic injection / Kerr synthesis), R26-R28 (GKP), R41-R42 (JAX autodiff and
stochastic estimators). None of these support a fault-tolerance, universal-
compiler, or general-flow-finder claim, and none of the scripts here claim
one.

**Should be rerun on a more powerful machine before finalizing figures**:
R28 (9 minutes; more seeds would tighten the ensemble error bars), R36
(dominates the performance section's runtime; a machine with more cores/RAM
could extend the mode-count sweep), and R32's mixed-Fock branch if a
denser statistical estimate of the noisy-homodyne variance is wanted (its
per-shot cost currently forces a very small sample there).
