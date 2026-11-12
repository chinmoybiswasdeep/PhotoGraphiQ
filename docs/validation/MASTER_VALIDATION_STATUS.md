# Master validation status

Cross-phase tracking for the 2026-11-12 validation, hardening and publication-readiness work on
branch `upgraded`. The local working tree is HEAD `f30321f` plus unstaged changes. "Last result"
refers to the 2026-11-12 reproduction run (`benchmarks/results_20261112/summary.json`: full suite
1721 passed, 13 skipped, 0 failed) and to the final regression run recorded in the
[Phase 5 audit](PHASE5_FINAL_RELEASE_AUDIT.md).

Status values:
- **Validated**: independent or cross-implementation reference, passing.
- **Regression-tested**: behavioural tests only.
- **Experimental**
- **Blocked**: external constraint.
- **Unsupported**

## Requirements

| ID | Requirement | Phase | Implementation | Tests | Independent reference | Status | Last result | Blockers | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| R1.1 | CI failures reproduced and root-caused | 1 | `experiments/validate_release.py`, notebooks, `pyproject.toml` | gate run | GitHub Actions run 43145860716 / 43145860723 metadata | Validated (locally) | all local gates pass | remote CI not re-run (no push) | [Phase 1](PHASE1_RELEASE_AUDIT.md) |
| R1.2 | Docs CI independent of Piquasso's Numba JIT | 1 | `examples/`, `notebooks/`, `pyproject.toml` (`numba<0.68`) | `scripts/check_docs_examples.py`, `tests/piquasso_reference/test_tutorial_backends.py` | Piquasso vs native outputs | Validated | 52 docs examples pass; only tutorial 24 needs Piquasso | upstream Numba 0.68 cache bug | Phase 1 |
| R1.3 | Native-only smoke | 1 | — | `tests/unit/test_native_only_smoke.py` | n/a | Regression-tested | pass | — | Phase 1 |
| R1.4 | Clean wheel, sdist and editable install | 1, 5 | `pyproject.toml` | fresh-venv install | n/a | Validated (Windows) | core-only wheel OK; sdist 30.2 MB; wheel 0.69 MB | Linux not available | [Phase 5](PHASE5_FINAL_RELEASE_AUDIT.md) |
| R1.5 | Python 3.11–3.14 | 1 | — | full suite per version | n/a | Validated (Windows) | 1645 passed on 3.11/3.13/3.14 after `numba<0.68` | Linux not tested | Phase 1 |
| R1.6 | Static checks | 1, 5 | — | ruff, ruff format, mypy, `mkdocs --strict` | n/a | Validated | clean | — | Phase 5 |
| R2.1 | Native Gaussian engine | 2 | `backends/native_gaussian.py`, `cpp/src/gaussian` | `tests/validation/test_val_gaussian.py`, `tests/native` | analytic moments, `expm` symplectics, dilations | Validated | max error 6.5e-13 | — | [matrix](independent-validation-matrix.md) |
| R2.2 | Native Fock and mixed-Fock engines | 2 | `backends/native_fock.py`, `cpp/src/fock` | `tests/validation/test_val_fock.py` | mpmath amplitudes, enlarged-space `expm` | Validated | max error 5e-16 (gates) | — | matrix |
| R2.3 | Combined squeeze+displace stability | 2 | `cpp/src/fock/gates.cpp` | `tests/native/test_native_fock_gaussian_unitary_stability.py` | 60-digit mpmath | Validated (one mode) | ≤ 1.9e-14 (was up to 1.4e2) | multi-mode displaced: no bound | [report](native-fock-gaussian-unitary.md) |
| R2.4 | Boson sampling | 2 | `sampling/` | `tests/sampling` | definitions, Piquasso | Validated | pass | — | [Phase 2](PHASE2_SCIENTIFIC_AUDIT.md) |
| R2.5 | GBS incl. threshold precision | 2 | `sampling/gbs.py` | `tests/sampling`, reference matrix | 40/50-digit mpmath, Piquasso | Validated (bound calibrated for few clicks) | 382 cases, 0 bound violations | double precision fails at r ≤ 0.5 with many clicks (use `precision="auto"`) | [GBS accuracy](gbs-threshold-accuracy.md) |
| R2.6 | Graph flows | 2, 3 | `flows/` | `tests/flow`, `tests/graphix_reference` | Graphix 0.4; exact rational witnesses | Validated | 360/360 Graphix agreement; CV classification tests pass | finite-squeezing certificate is a float64 claim | [CV flow](cv-flow-robustness.md) |
| R2.7 | Calculus (measurement calculus, LC, ZX subset) | 2 | `calculus/` | `tests/calculus` | tensor evaluation, branch simulation | Validated | pass | general CV-ZX unsupported | Phase 2 |
| R2.8 | PhotoGraphiQML incl. MentPy | 2 | `src/photographiqml` | `tests/ml`, `tests/ml/mentpy_reference` | MentPy at commit `63c3d83e` | Validated | 154/154 + 34/34; 24 experiments bit-identical | MentPy absent from the project venv | [MentPy](photographiqml-mentpy-validation.md) |
| R2.9 | Measurement-angle API defect | 2 | `backends/base.py:resolve_angle` | `tests/regression/test_measure_angle.py` | analytic density | Validated | pass | — | Phase 2 |
| R3.1 | CUDA-Q Dynamics adapter | 3 | `cudaq_dynamics.py` | `tests/cudaq_adapter` | native-fock, SciPy Lindblad | Experimental (translation validated; execution unverified) | 28 passed, 1 skipped | CUDA-Q Linux-only; no WSL/Docker | [CUDA-Q](../cudaq.md) |
| R3.2 | GPU Fock acceleration | 3 | `gpu/fock.py` | `tests/gpu/test_gpu_fock.py` | native kernels | Validated (correctness); speed workload-dependent | max deviation 7.2e-16 | small single states slower on GPU | [GPU](../gpu.md) |
| R3.3 | GPU sampling acceleration | 3 | — | measurement only | n/a | Unsupported (measured, not justified) | single large matrices 0.2–0.85× | — | Phase 3 |
| R3.4 | Restricted CV graphical calculus | 3 | `calculus/gaussian_graph.py` | `tests/calculus/test_gaussian_graph.py` | `GaussianBackend` numerics | Validated | 213 passed, max error 2e-14 | — | [guide](../user-guide/gaussian-graphical-calculus.md) |
| R3.5 | CV-flow robustness | 3 | `flows/cv.py`, `flows/_exact.py` | `tests/flow/test_flows_cv_robustness.py` | exact rational arithmetic | Validated | 383 flow/calculus tests pass | exact check opt-in for large r | CV flow |
| R3.6 | Interactive visualization in a real browser | 3 | `interactive/` | `tests/visualization`, `scripts/check_interactive_browser.py` | real input events, screenshots | Validated (Edge only) | 9/9 checks pass | Firefox, WebKit, touch not covered | [browser](interactive-visualization-browser.md) |
| R4.1 | Contribution matrix | 4 | — | — | — | Done (needs authors' literature check) | — | — | [Phase 4](PHASE4_MANUSCRIPT_REPRODUCIBILITY.md) |
| R4.2 | Manuscript experiments reproduced | 4 | `scripts/reproduce_manuscript.py`, `scripts/compare_recorded_results.py` | `tests/unit/test_compare_recorded_results.py` | recorded results | Validated for the safe subsets | every shared value reproduced; schema drift in 10 core files | heavy experiments not re-run | Phase 4 |
| R4.3–4.8 | Benchmarks (Gaussian, Fock, BS/GBS, Graphix, GPU) | 4 | `benchmarks/` | smoke tests | correctness check before timing | Validated (one machine) | 2026-11-12 bundle | single Windows laptop | `benchmarks/results_20261112/` |
| R4.9 | Figures | 4 | `docs/architecture/figures.json`, `benchmarks/plot_results.py` | `tests/unit/test_architecture_figures.py`, `tests/unit/test_benchmark_plots.py` | code-reference check | Validated | 532 refs, print check OK | — | Phase 4 |
| R4.10 | BS/GBS notebook | 4 | `examples/notebooks/boson_sampling_and_gbs_benchmark.ipynb` | clean-venv execution | n/a | Validated | 0 errors (Python 3.13 clean venv; project venv) | — | Phase 4 |
| R5.x | Final release audit | 5 | — | — | — | see Phase 5 | — | — | Phase 5 |

## The eight original upgrade requests

| # | Request | Status | Basis |
|---|---|---|---|
| 1 | Native C++ replacements for Piquasso | **Fully met** | Native Gaussian, Fock and mixed-Fock engines are independently validated (R2.1–R2.3). Piquasso is optional; the core and docs run without it, except autodiff, which declares it. |
| 2 | Dynamic, draggable MBQC graphs and correct flow/correction visualization | **Fully met** (Edge-verified) | R3.6. Remaining layout limits are documented. |
| 3 | Manuscript-quality architecture schematics | **Fully met** | Figures A–H validated against code (R4.9). |
| 4 | Faster Fock/Gaussian simulation with CUDA-Q and optional GPU | **Partially met** | Native CPU speedups measured (R4.3–4.4). The CuPy GPU Gaussian and Fock kernels are validated and faster for batched workloads (R3.2). CUDA-Q is an adapter with execution unverified (R3.1, blocked by platform). No custom CUDA kernels (no `nvcc`). |
| 5 | BS/GBS notebook and meaningful Piquasso/Graphix benchmarks | **Fully met** | R4.10 and R4.5–4.7. The benchmarks report where PhotoGraphiQ is slower. |
| 6 | Measurement calculus, local Clifford, ZX calculus | **Fully met for the stated scope** | Qubit measurement calculus, LC and a ZX subset (R2.7), plus the Gaussian graphical calculus (R3.4). General CV-ZX is unsupported. |
| 7 | Determinism and graph-flow algorithms | **Fully met** | Causal flow, gflow, Pauli flow (Graphix-validated) and CV flow with exact classification (R2.6, R3.5). |
| 8 | Complete functionality documentation | **Fully met** | `docs/COMPLETE_FUNCTIONALITY.md` (36 sections), updated 2026-11-12. |
