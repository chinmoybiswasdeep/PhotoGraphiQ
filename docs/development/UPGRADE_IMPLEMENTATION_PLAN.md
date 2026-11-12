# Upgrade implementation plan (branch `upgraded`)

Working checklist for the native-engine, sampling, calculus, flow and
visualization upgrade. Status marks: `[x]` implemented and validated,
`[~]` implemented but partial or experimental, `[ ]` not done, `[-]` not feasible
here (blocker recorded). The final state of each item is the authority; see
[COMPLETE_FUNCTIONALITY.md](../COMPLETE_FUNCTIONALITY.md) for the user-facing inventory.

## 0. Starting point and environment

- Branch `upgraded` at `72bcf44`, carrying the unstaged PhotoGraphiQML integration
  (21 modified, 835 untracked files). Those changes are preserved and built on.
- Baseline: full suite 689 passed, 2 skipped (MentPy absent) on the integration tree.
- Toolchain found: MSVC 19.50 (VS 2026 Build Tools, C++20, OpenMP 2.0), CMake 4.4
  and Ninja (pip), pybind11 3.1, scikit-build-core 1.1, Eigen 3.4.0 (fetched at a
  pinned SHA-256, MPL-2.0 only via `EIGEN_MPL2_ONLY`).
- GPU: NVIDIA RTX 4050 Laptop (compute 8.9, driver 581.86, 6 GB). **No CUDA toolkit
  (`nvcc`) is installed**, so compiled CUDA kernels cannot be built here.
- CUDA-Q publishes Linux-only Python wheels; it cannot be installed on this native
  Windows environment, so CUDA-Q Dynamics integration cannot be executed here.

## 1. Conventions every engine follows

hbar = 2, `[q, p] = 2i`, interleaved quadratures `(q0, p0, q1, p1, ...)`, statistical
covariance with vacuum `V = I`, symplectic form `Omega = (+)[[0, 1], [-1, 0]]`,
coherent mean `(2 Re a, 2 Im a)`, p-squeezed resource `diag(e^{2r}, e^{-2r})`,
homodyne `q cos t + p sin t`, CZ `p_u += g q_v`, calibrated inefficient homodyne
(added variance `(1-eta)/eta + noise`), Fock truncation by **total** photon number
`sum n < cutoff`. Native code must reproduce the Python reference conventions exactly.

## 2. Architecture decisions

| Decision | Reason |
| --- | --- |
| One extension `photographiq._native` with components `gaussian`, `fock`, `sampling` (CMake `PGQ_COMPONENTS`), static `pgq_core` shared by bindings and C++ tests | Components build and test independently; no duplicated kernels |
| Native backends subclass the validated Python backends and replace only state storage and numerical kernels | Measurement, preflight, GKP and diagnostics logic is reused, not duplicated |
| `photographiq.ir.OpenGraph` is the graph-level IR (typed `Model`: cv-gaussian, cv-nongaussian, qubit, gkp-logical); `Pattern` stays the CV command IR; qubit commands live in `photographiq.calculus` | Smallest coherent extension of existing APIs |
| Piquasso imported lazily (`photographiq._optional`), then made an optional extra once native backends are validated | Core runs without Piquasso; Piquasso remains an isolated reference |
| Visualization renders data computed in Python (Cytoscape.js HTML for interaction, Matplotlib for static exports) | No physics in the frontend; static exports need no browser |
| GPU work only through optional packages detected at runtime; explicit reporting of the device actually used | No silent CPU fallback |

## 3. Stages, dependencies and acceptance criteria

| Stage | Depends on | Acceptance criteria |
| --- | --- | --- |
| S0 Build system (scikit-build-core, CMake, loader) | - | `pip install -e .` builds `_native`; core imports without the extension and without Piquasso |
| S1 IR and backend registry | S0 | `OpenGraph` round trips; registry lists capabilities, device, precision, approximations; unsupported operations raise |
| S2 Native Gaussian engine | S0 | Symplectic updates, channels, homodyne/heterodyne/generaldyne conditioning, diagnostics; agrees with NumPy backend trajectories and closed forms |
| S3 Native Fock + mixed engines | S0 | Exact gate matrix elements, total-photon basis, measurements via existing logic; agrees with Piquasso and analytic results; cutoff convergence |
| S4 Sampling (permanent, hafnian, loop hafnian, torontonian; BS and GBS APIs) | S0 | Agreement with brute-force definitions and Piquasso; probability normalization on complete small systems |
| S5 Flows (causal, gflow, Pauli flow, CV analysis) | S1 | Defining conditions verified; agreement with Graphix where semantics match; CV corrections verified on several outcome sets |
| S6 Calculus (qubit measurement calculus, local Clifford, ZX subset) | S1 | Every rewrite checked by statevector or tensor equality |
| S7 Interactive visualization | S1, S5 | Construct → flow → corrections → execution stages from real data; layouts; exports |
| S8 GPU (optional) | S2, S3 | Real device execution or explicit unavailability; CPU/GPU agreement |
| S9 Benchmarks, BS/GBS notebook | S2-S4 | Machine-readable results with environment metadata; notebook executes |
| S10 Architecture figures, COMPLETE_FUNCTIONALITY.md, README | all | Figures generated from a structured description of implemented modules |
| S11 Integration audit | all | Full test suite, PhotoGraphiQML tests, representative manuscript experiments, Git status unchanged except local edits |

## 4. Checklist

- [x] S0 build system: scikit-build-core + CMake + pybind11, one `photographiq._native`
  extension; core imports without the extension and without Piquasso.
- [~] S1 IR and registry: `OpenGraph`, `BackendInfo` registry and `select_backend` work and are
  tested, but only behaviourally (no independent reference applies), so they are marked
  experimental.
- [x] S2 native Gaussian: agrees with the NumPy backend trajectories and closed forms; batched
  OpenMP kernels.
- [~] S3 native Fock and mixed: validated against Piquasso, closed forms and exact Gaussian
  moments, but Fock paths stay experimental by project policy. Combined squeeze plus
  displacement in one unitary is not fully stable at large |alpha| and cutoff (documented).
- [~] S4 sampling: permanent, (loop) hafnian, (loop) torontonian, BS and GBS APIs validated
  against definitions and Piquasso. Threshold probabilities with many clicks are less
  accurate than Piquasso in double precision; a conservative rounding bound, a warning and
  an mpmath path are provided.
- [x] S5 flows: causal flow, gflow, Pauli flow (checked against Graphix) and CV determinism.
  CV determinism can be undecided for resource squeezing r >~ 3.5.
- [x] S6 calculus: qubit measurement calculus, local Cliffords and a ZX subset, every rewrite
  checked by tensor or statevector equality. Bialgebra is 2x2 only.
- [~] S7 visualization: construct, flow, corrections and execution stages from real data, plus
  scientific plots. Layout quality and rendered appearance have no reference, so these are
  experimental.
- [~] S8 GPU: batched Gaussian kernels on CuPy, validated on the RTX 4050 against NumPy and
  native; no silent CPU fallback. Not done: Fock or sampling kernels on the GPU, and custom
  CUDA kernels (no `nvcc` here).
- [-] S8 CUDA-Q Dynamics execution: CUDA-Q has Linux wheels only; only a probe and a
  registered placeholder that raises exist.
- [x] S9 benchmarks and notebook: six benchmarks with environment metadata, correctness gates
  and figures; the BS/GBS notebook (14 sections) executes without errors.
- [x] S10 documentation and figures: `COMPLETE_FUNCTIONALITY.md` (36 sections), user guides,
  API pages, figures A-H generated from `docs/architecture/figures.json`.
- [x] S11 integration audit: see section 5.

## 5. Final status

Recorded on 2026-11-11 in the development environment of section 0. No commits were made;
every change is local and unstaged on branch `upgraded`.

| Check | Result |
| --- | --- |
| Full test suite (all optional packages installed) | 1273 passed, 14 skipped (MentPy absent, CUDA-Q absent, capability gaps), 0 failed |
| Core-only emulation (Piquasso, Graphix, JAX, mpmath, CuPy, MentPy, CUDA-Q blocked) | 761 passed, 339 skipped, 0 failed |
| Fresh wheel install with only NumPy, SciPy and NetworkX | imports; backends `gaussian`, `native-gaussian`, `native-fock`, `native-mixed-fock`; optional features raise `ImportError` naming the extra; no optional module imported |
| `ruff check`, `ruff format --check` (src, tests, benchmarks, scripts) | clean |
| `mypy` (src/photographiq, src/photographiqml) | no issues in 87 source files |
| `mkdocs build --strict` | clean |
| Architecture figures | 8 figures, 484 code references valid, print check passed at 7.0 in |
| Benchmarks (idle machine, 2026-11-11 09:26-09:45) | boson sampling, GBS, native Gaussian, GPU, Graphix, Fock and the 50-digit threshold reference; results and figures under `benchmarks/*/results/` |
| BS/GBS notebook | executed, 0 errors |
| Manuscript experiments (scratch copies, safe subsets) | PhotoGraphiQ core: 24 run, 24 passed, 18 skipped by the runner (long benchmarks, JAX). PhotoGraphiQML: 37 run, 28 passed (three of them after installing scikit-learn from the `ml-experiments` extra); the 9 failures are all MentPy imports (extra `ml-validation`, not installed here); 22 skipped by the runner (moderate or heavy). Git provenance hashing was disabled in the scratch copy, which is not a repository; the recorded results in the repository were not modified |

Behaviour changes users will notice: Piquasso is optional (`photographiq[piquasso]`), and
`simulate` defaults to `backend="gaussian"` (identical seeded trajectories to the former
default). Known limitations are listed in section 35 of
[COMPLETE_FUNCTIONALITY.md](../COMPLETE_FUNCTIONALITY.md).
