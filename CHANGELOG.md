# Changelog

### Unreleased: CI repair, documentation and manuscript experiments R43-R60 (2026-11-13)

- **CI.**
  - mypy failed with mypy 2.4 and NumPy 2.5.4 stubs (`np.isscalar` narrowing). The five scalar checks now share `photographiq._checks.is_finite_real_scalar` (same semantics).
  - The headless-browser test passes `--no-sandbox` on Linux, where GitHub's runners have Chrome but no usable sandbox.
- **Native engines.** Built with `EIGEN_DONT_PARALLELIZE`: the kernels parallelise themselves, and Eigen's own threads made small mixed-state gates up to 8.6 times slower on an idle machine.
- **Registry.** `native-fock` and `native-mixed-fock` now list renormalisation among their approximations, as they renormalise after every operation.
- **Interactive views.** Routed edges and arrows that avoid nodes, stage and layout selectors, order stepping, SVG/PNG/HTML downloads, a colour-blind-safe visual language and labels that fit their nodes. `from_pattern` annotates constant displacements and adaptive gates instead of dropping them. Fourteen demonstration figures in `docs/figures/demos/`.
- **Documentation.** New README, installation guide with real backend checks, backend comparison, theory pages (Gaussian, compilation, cluster states, flow, calculus, non-Gaussian, sampling), PhotoGraphiQML guide; historical pages labelled as such.
- **Manuscript experiments.**
  - R1-R42 migrated to the native engines and `photographiq.flows`; historical outputs kept in `paper_experiments/legacy_reference/441de9a/`.
  - New R43-R60: native Gaussian, Fock and mixed-Fock validation and performance, boson sampling, GBS, threshold GBS, CV flow finding and determinism, qubit flows against Graphix, measurement calculus, local Clifford, ZX, interactive visualization, GPU Gaussian, PhotoGraphiQML physical execution and an end-to-end showcase.
  - Provenance records the source SHA-256 and dirty state; `run_all_safe.py` has smoke, safe and heavy tiers; the notebook runs the scripts unchanged.
- **Validation tooling.** `scripts/audit_manuscript_consistency.py` (12 documentation/experiment consistency checks), `reproduce_manuscript.py --mode smoke|scientific|full`, and the `manuscript` workflow.

### Unreleased: validation, hardening and publication readiness (2026-11-12)

- **CI.**
  - Fix the lint and format failures in three example notebooks.
  - Constrain `numba<0.68` in every extra that installs Piquasso. With numba 0.68.1 and llvmlite 0.50.0, Piquasso's cached hafnian kernels abort on reload with `LLVM ERROR: Symbol not found`. This was the documentation-CI failure.
  - Remove `pythonpath = ["src"]` from the pytest configuration, so tests run against the installed package.
- **Documentation examples** (tutorials, projects, notebooks) run on the native backends. Only the autodiff tutorial needs Piquasso. `cutoff_convergence` and `measurement_convergence` accept `native-fock`; their defaults are unchanged.
- **Native Fock:** one-mode Gaussian unitaries use a stable rotation–squeeze–rotation–displacement factorisation. Errors drop from up to 1.4e2 to ≤ 1.9e-14 against 60-digit references.
- **Backends:** `measure()` honours `Homodyne.angle` when no explicit angle is passed (previously a direct call measured q), and raises on contradicting angles.
- **Sampling:**
  - `threshold_probability(precision="auto", rtol=..., return_info=...)` and `ThresholdInfo`.
  - A 382-case multiprecision reference matrix (`threshold_reference_matrix.py`).
  - A faster mpmath path.
- **Flows:** CV flow results are classified as certified (with the regime stated), impossible (with exact rational witnesses) or numerically uncertain. New `shift=` and `exact=` options and `flows.exact_ideal_limit`. An uncertain float64 rank decision now gives `unknown` instead of `not_certified`.
- **Calculus:** `photographiq.calculus.GaussianGraph`, the Gaussian-pure-state graphical calculus of Menicucci, Flammia and van Loock (2011).
- **GPU:** `photographiq.gpu.fock` (batched Fock gates on CuPy/cuSPARSE).
- **CUDA-Q:** the `cudaq-dynamics` pattern backend, which always raised, is removed. It is replaced by the Hamiltonian-level adapter `photographiq.cudaq_dynamics`; CUDA-Q execution is unverified.
- **Interactive views:** fix struck-through correction labels, faded outcome text, a stale selection panel and the static-SVG font fallback. Add real-browser checks (`scripts/check_interactive_browser.py`).
- **Validation:**
  - Independent Gaussian and Fock validation suite (`tests/validation`).
  - `scripts/reproduce_manuscript.py` and `scripts/compare_recorded_results.py`.
  - Audit documents under `docs/validation/` (Phases 1–5 and the master status).
  - The 2026-11-12 benchmark bundle `benchmarks/results_20261112/`.

### Unreleased: scientific software upgrade

- **Breaking:** Piquasso is no longer a runtime dependency. Install
  `photographiq[piquasso]` for the `piquasso`, `piquasso-fock` and
  `piquasso-mixed-fock` reference backends and for the default physical
  PhotoGraphiQML backend. A missing optional package raises `ImportError` with the
  install command.
- **Breaking:** `simulate` defaults to `backend="gaussian"` instead of `"piquasso"`.
  Both are exact Gaussian simulators that consume random numbers identically, so
  seeded trajectories agree to rounding.
- Add native C++20 engines (Eigen, pybind11, OpenMP; built by scikit-build-core):
  `native-gaussian`, `native-fock` and `native-mixed-fock`. They are cross-checked
  against the NumPy backends, Piquasso and closed forms.
- Add a backend capability registry: `available_backends`, `describe_backend` and
  `select_backend`.
- Add a unified typed IR (`photographiq.ir`).
- Add `photographiq.sampling` for boson sampling and GBS. It provides permanents,
  hafnians, loop hafnians, torontonians, loop torontonians, exact distributions
  and samplers, with native kernels.
- Threshold probabilities are evaluated in a dark-mode-conditioned form, report
  an error estimate and raise `ThresholdPrecisionWarning` above 1e-6. Add a
  `precision=` multiprecision path (extra `precision`).
- Add `photographiq.flows` (causal flow, gflow, Pauli flow, CV determinism,
  `FlowAnalysisResult`).
- Add `photographiq.calculus` (qubit and CV measurement calculus, local
  Cliffords, a ZX subset).
- Add `photographiq.interactive`: a graph editor, flow and correction overlays,
  execution traces and scientific plots.
- Add optional CuPy GPU kernels (`photographiq.gpu`; extras `gpu-cuda12` and
  `gpu-cuda13`) with explicit device reporting and no silent CPU fallback. Add a
  Linux-only CUDA-Q probe backend.
- Add benchmarks with environment metadata and equivalent-task comparisons
  (`benchmarks/`), manuscript architecture figures (`docs/architecture`) and
  `docs/COMPLETE_FUNCTIONALITY.md`.
- PhotoGraphiQML physical execution accepts `backend="native-fock"`, validated
  against `piquasso-fock`, which remains the default.

### Unreleased: unified repository

- Ship PhotoGraphiQML as the `photographiqml` package from this distribution
  (`pip install -e ".[dev]"` installs both); add `ml-validation` and
  `ml-experiments` extras. Its tests, docs, examples and manuscript suite live
  under `tests/ml`, `docs/ml`, `examples/ml` and `paper_experiments_ml`.
- Gaussian backends support postselection (`measurement_outcomes`) and record
  exact outcome densities, so `Result.log_likelihood` works on `gaussian` and
  `piquasso`.
- Fix spurious `Covariance must be symmetric` failures in deep or strongly
  squeezed Gaussian circuits by re-symmetrising after every gate.
- Add `GaussianState.fidelity` (mixed-state Uhlmann), `purity`,
  `symplectic_eigenvalues`, `entropy` and `log_negativity`; support
  `QuadraticPhase` in `gaussian_channel`; export `pg.cvflow`.
- Document that `WignerGrid.negative_volume` equals half the Kenfack-Zyczkowski
  indicator. See `docs/development/merged-integration-report.md`.

### v0.3 scientific hardening

- Correct JAX beamsplitter boundary ordering and package phase convention.
- Reject unresolved JAX tensor/addition normalization; expose retained masses
  and boundary diagnostics, with explicit eager/traced invalid-branch behavior.
- Canonicalize mixed-state occupation integers for Windows/Numba preparation.
- Correct GKP floating-point half-cell tie parity.
- Preserve nonlinear approximation metadata and warn during Kerr compilation.
- Add independent channel, GKP, gradient and synthesis tests and measured evidence.
- Restore missing tutorial transcripts to clean documentation builds; withhold
  the coverage badge until Codecov receives a verified main-branch upload.

Changes follow semantic versioning. Historical entries below are preserved.

## Unreleased — 0.3.0 development

### Added

- MkDocs Material website, source API reference, input/output guides, glossary,
  24 executable tutorials, 10 demo projects and five curated notebooks.
- Circuit drawings, richer MBQC diagrams and compilation provenance with SVG,
  PNG and PDF export.
- Experimental mixed-Fock density matrices, thermal attenuation and conditional
  counting/homodyne; calibrated inefficient/noisy detection.
- Cubic-resource compilation and approximate quartic/Kerr synthesis over a
  Gaussian-plus-cubic target gate set, with explicit refinement controls.
- Finite-energy square-lattice GKP resources, stabilizer diagnostics and elementary
  modular syndrome correction infrastructure.
- Optional JAX finite-Fock execution, fixed-branch gradients, Gaussian sampling
  reparameterization and a discrete likelihood-ratio estimator primitive.
- Documentation CI/Pages configuration, Codecov uploads and community templates.

### Changed

- Source installation paths, metadata, contribution guidance and result explanations.
- Cutoff studies may select the mixed-Fock backend.
- New research paths remain experimental; no publication or remote deployment is
  implied by this development version. Existing v0.2 evidence remains historical.

## 0.2.0 — experimental non-Gaussian MBQC

- Add sparse correlated Fock resources and public native pure-state import.
- Add capability preflight and explicit resource descriptions for cats and finite
  cubic-phase states, preserving CP(gamma)=exp(i gamma q³/6) at hbar=2.
- Add native Kerr/quadratic gates and mathematical photon addition/subtraction
  with square-root occupation factors and explicit normalization diagnostics.
- Add conditional ideal Fock homodyne, explicit postselection, probability/density
  records, and reverse exponentiation for count-dependent expressions.
- Add physical vacuum-tap subtraction, general/cat resource filters and finite
  cubic injection with nonlinear Gaussian feedforward.
- Add occupation-aligned fidelity/trace distance, quadrature moments, Wigner
  marginals, finite-grid negativity and branch-aware cutoff studies.
- Add norm/boundary warnings and Fock-dimension allocation guards.
- Preserve Gaussian compiler, flow and channel-analysis behavior; add quadratic
  phase execution to Gaussian backends. The affine channel analyzer still rejects
  this newly introduced command until its separate analysis rule is implemented.
- Extend analytical, independent Fock, native Piquasso and Graphix structural
  validation; update tutorials, experiments and the Quantum-style manuscript.

Fock loss/noisy measurements, mixed-state inputs, dynamic representation switching,
GKP decoding and universal non-Gaussian compilation remain unsupported.
The revised PDF is `paper/PhotoGraphiQ-v0.2.pdf`; `paper/PhotoGraphiQ.pdf` retains
the prior version because the open PDF viewer locked it during the update.


## v0.2 validation hardening

- Retain the original failing injection cases while separating finite-Fock,
  Gaussian-decomposition and independent quadrature error budgets.
- Reject invalid/insufficient resources before earlier native preparations and
  prevent slightly mixed Gaussian inputs from being silently treated as pure.
- Add optional raw quadrature moments through order four and photon moments;
  avoid dense allocation for pure norms and pure/pure state metrics.
- Check partial homodyne CDF integration errors; expand analytical density,
  Wigner, mixed-state, sensitivity, heralding and label-aware Graphix validation.
- Run every release gate on Python 3.11?3.14, retain per-version artifacts and
  disable fail-fast cancellations. See docs/release-hardening-report.md for
  actual outcomes; a configured workflow alone is not a release certification.
