# Manuscript mapping

Suggested placement of experiment outputs in the Quantum manuscript "PhotoGraphiQ: A photonic
continuous variable software platform to perform measurement-based quantum computing". This is
a suggestion for the authors, not a claim about a submitted manuscript. Every figure below is
produced by the named script; its caption (`figures/captions/<name>.caption.md`) quotes numbers
computed by that script. Status of each experiment: `STATUS.md`. The R1-R42 mapping of
2026-09 is superseded; it is retrievable with `git show 441de9a:paper_experiments/MANUSCRIPT_MAP.md`.

## Narrative: five contributions

| Contribution | Claim the evidence supports | Experiments |
|---|---|---|
| A. Unified CV-MBQC programming model | Graphs, patterns, compilation, flow, feed-forward and measurement calculus in one IR; compiled maps equal analytic targets to rounding; finite squeezing appears only as a separately reported noise matrix | R1-R14, R51-R54 |
| B. Accurate native simulation | Native Gaussian, Fock and mixed-Fock engines agree with first-principles references (closed forms, enlarged-space `expm`, mpmath) at rounding level; truncation is always reported | R18-R34, R43, R45, R47 |
| C. Photonic sampling and performance | Permanents, hafnians and torontonians with explicit error control; measured speedups of the native engines; GPU gains only for batches | R35-R38, R44, R46, R48-R50, R58 |
| D. Explainability and reproducibility | Interactive and static views that display exactly the validated graph, flow and corrections; machine-readable results with provenance; one end-to-end showcase | R39, R40, R57, R60 |
| E. Extension to photonic machine learning | PhotoGraphiQML physical execution on the native Fock engine agrees with Piquasso | R59 and the PhotoGraphiQML suite (`../paper_experiments_ml/MANUSCRIPT_MAP.md`) |

## Main-text figures

| Figure | Source | Files | Notes |
|---|---|---|---|
| 1. Platform architecture | `scripts/generate_architecture_figures.py` | `docs/figures/architecture/figure_A_complete_architecture.*` (repository root) | Code-validated architecture figure (figure references checked by `--check`). |
| 2. Graph, flow and correction | R51, R57 | `figures/png/R51_cv_flow_examples.png`, `figures/png/R57_interactive_visualization.png` | Certified, impossible and uncertain CV graphs; four views of one validated graph. Demo figures 06 and 07 in `docs/figures/demos/` are alternatives. |
| 3. Compilation and finite squeezing | R5, R29, R52 | `figures/png/R5_random_symplectic_compile.png`, `figures/png/R29_finite_squeezing.png`, `figures/png/R52_cv_mbqc_determinism.png` | Compiled map error near machine precision; noise N(r) reported separately, slope -2 in r. |
| 4. Native engine accuracy | R43, R45 | `figures/png/R43_native_gaussian_validation.png`, `figures/png/R45_native_fock_accuracy.png` | Errors against independent references across modes, depths and cutoffs. |
| 5. Boson sampling and GBS | R48, R49, R50 | `figures/png/R48_boson_sampling.png`, `figures/png/R49_gaussian_boson_sampling.png`, `figures/png/R50_threshold_gbs.png` | R50 shows where double precision fails and that the reported bound holds. |
| 6. Performance and scaling | R44, R35, R58 | `figures/png/R44_native_gaussian_performance.png`, `figures/png/R35_fock_dimension_scaling.png`, `figures/png/R58_gpu_gaussian.png` | Use only timings from the idle-machine run recorded in `STATUS.md`. |
| 7. End-to-end showcase | R60 | `figures/png/R60_end_to_end_showcase.png` | Panels A-F: target circuit, cluster graph, flow layers, corrections, outputs, accuracy and N(r). |
| 8. (optional) Non-Gaussian, GKP or ML | R23 or R26-R28, or R59 | `figures/png/R23_cubic_injection.png`, `figures/png/R28_gkp_syndrome.png`, `figures/png/R59_photographiqml_physical.png` | Finite-energy approximations; no fault-tolerance claim. |

## Tables

| Table | Source | Notes |
|---|---|---|
| Capability matrix | `tables/native_feature_status.csv` | Upgraded engines; V validated, E experimental. The historical `tables/feature_status.csv` covers the 2026-09 backends. |
| Numerical validation summary | `tables/compiler_accuracy.csv`, `tables/backend_agreement.csv`, `tables/non_gaussian_validation.csv` | R3-R7, R18, R19, R21, R22, R24. |
| Flow comparison with Graphix | `tables/R53_qubit_flows_graphix.md`, `tables/R51_cv_flow_finding.md` | Qubit notions only for Graphix; CV classification separately. |
| Performance | `tables/performance_summary.csv` | R35-R38; idle-machine timings only. |
| Provenance | `tables/experiment_provenance.csv` | Commit, dirty flag and source SHA-256 of every result. |

## Appendices

| Appendix section | Source experiments |
|---|---|
| Gate sweeps and compiler details | R3, R4, R6, R7, R8, R9, R10 |
| Flow, causality and determinism | R11, R12, R13, R14, R51, R52 |
| Qubit flows, measurement calculus, local Clifford, ZX (qubit subset) | R53, R54, R55, R56 |
| External references (Piquasso, Graphix) and backend cross-checks | R15, R16, R17, R18 |
| Native Fock and mixed-Fock engineering | R46, R47 |
| Non-Gaussian resources and synthesis | R19-R25 |
| GKP | R26, R27, R28 |
| Noise models | R30, R31, R32 |
| Convergence | R33, R34 |
| Serialization and reproducibility | R39, R40 |
| Automatic differentiation | R41, R42 |
| GPU kernels | R58 |
| PhotoGraphiQML physical layer | R59 |

## What NOT to claim

- R17 and R53 compare with Graphix for discrete-variable structure and flows only; Graphix is
  never a CV amplitude, quadrature or Wigner-function reference.
- R23, R25 and R26-R28 are finite-resource or finite-energy approximations; their fidelities are
  not exact-gate or fault-tolerant results. R59's agreement is between two engines that share
  the Python preparation and decoding layers.
- Finite-squeezing channels are noisy maps, never ideal unitaries (R2, R29, R52, R60).
- R50: double-precision threshold probabilities are not accurate in the many-click, low-squeezing
  regime; claim the error bound and the multiprecision path, not full double precision.
- R54-R56: the measurement calculus, local Clifford and ZX results are qubit results; there is
  no general CV measurement calculus or CV-ZX.
- R44, R46, R58 and R35-R38 timings are single-machine observations; quote them with the
  hardware, thread count and the run date from `STATUS.md`. GPU gains apply to batches only.
- CUDA-Q: no experiment executes on CUDA-Q; the adapter's execution path is unverified.
- No experiment claims a GKP fault-tolerance threshold, a universal non-Gaussian compiler or an
  optimal general CV-flow finder.
