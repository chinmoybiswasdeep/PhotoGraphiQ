# Proposed QUANTUM-journal manuscript map

This is a suggestion for how this suite's results could be organized into a
manuscript, not a claim about any actual submitted paper. Figure/table
numbers below match the `Destination` column of `EXPERIMENT_INDEX.md`.

## Main text

| Figure | Source experiment(s) | Notes |
|---|---|---|
| Fig. 1 | R1, R2, R3 | Triangle/MuTA graph anatomy and Table-I identities against analytic oracles. |
| Fig. 2–3 | R7, R8, R9 | MentPy semantic-topology, flow, and density-matrix agreement (the headline independent logical validation). |
| Fig. 4 | R11, R25, R26 | Eq. 5 kernel: MentPy agreement, independent-contraction agreement, and PSD/conditioning properties. |
| Fig. 5 | R18 | Parameter-shift vs. finite-difference gradient verification. |
| Fig. 6 | R20, R21, R22 | Haar and Ising-XX gate-learning statistics (20 seeds each) and sensitivity sweep. |
| Fig. 7 | R23, R24, R27 | Classifier/regressor verification and kernel-SVM classification vs. classical baselines. |
| Fig. 8 | R33, R34 | Local Fisher spectra and finite-GKP codeword projection/convergence. |
| Fig. 9 | R35, R37, R38 | Physical capability boundary, exhaustive Pauli-frame validation, and independent public-Pattern cross-check. |
| Fig. 10 | R38, R39, R40 | Raw Piquasso state-preparation and CZ validation (independent of PhotoGraphiQML's lowering helper). |
| Fig. 11 | R43, R44, R45 | Decoded logical-vs-physical statistics, shot convergence, and joint-readout correlations. |
| Fig. 12 | R47, R48 | Resource-axis convergence and discrete physical-training instability (an explicit negative/instability result). |
| Fig. 13 | R13, R42, R_PERF | Aggregate runtime/resource-scaling and abstraction-overhead figure. |
| Fig. 14 | R51 | Expanded local/nonlocal gate-family learning, with failures retained. |
| Fig. 15 | R52 | Leakage-safe repeated classification against majority, linear, and nonlinear baselines. |
| Fig. 16 | R53 | Regression interpolation, extrapolation, residuals, and classical baselines. |
| Fig. 19 | R56 | Supported logical-to-physical probability and observable gaps. |
| Fig. 20 | R57 | Disjoint-shot physical selection, validation, and fresh-test evaluation. |

## Tables

| Table | Source | Notes |
|---|---|---|
| Table I | R1, R2, R3 | Structural/analytic identities defining the paper ansatz. |
| Table II | R7, R8 | MentPy semantic-topology and flow-agreement summary. |
| Table III | R16 | Fail-fast invalid-input capability matrix. |
| Table IV | R32 | Pauli Lie-closure dimension cross-check (known generator sets). |
| Table V | R50 | Canonical Euler/Ising-XX reproduction and explicit unsupported source-paper claims. |

## Appendices

| Appendix section | Source experiments |
|---|---|
| A. Logical foundations (independent cross-checks) | R4, R5, R6 |
| B. MentPy validation details | R10, R12, R13 |
| C. API contracts | R14, R15, R16, R17 |
| D. Gradients and training | R19 |
| E. Kernels and learning robustness | R28 |
| F. Expressivity/diagnostics | R29, R30, R31 |
| G. Physical lowering | R36 |
| H. Piquasso validation (broader coverage) | R41, R42 |
| I. Physical statistics and decoding | R46 |
| J. Kernel and expressivity ablations | R54, R55 |
| K. Physical/resource boundaries and scaling | R58, R59 |

## What NOT to claim

- R7–R13 (MentPy) validate the **ideal logical qubit layer only**. MentPy is
  never a continuous-variable or finite-GKP oracle (see
  `docs/research/muta-mapping.md`); do not cite any MentPy-agreement result
  as evidence for physical/GKP correctness.
- No experiment in this suite claims arbitrary-angle physical MuTA,
  continuous physical training, an `alpha -> homodyne angle` substitution,
  universality, fault tolerance, or quantum advantage. R48 in particular is
  an explicit **instability/negative result**, not a robust-classifier
  claim (see `docs/physical/physical-training.md`).
- R38–R42 (Piquasso/Pattern comparisons) are classified independence **class
  C**, not B: the finite-GKP resource (cutoff/width/envelope/peaks/grid and
  the encoded input amplitudes) is a **shared input** between the two sides
  being compared, obtained from PhotoGraphiQ since raw Piquasso has no
  native GKP-resource constructor. Only the gate/pattern **orchestration**
  is independent.
- R43's `compare_logical_physical` never reports a decoded-state fidelity
  (`finite_state_fidelity` is always `None`, verified by R43 itself); do not
  substitute TV distance or observable differences for a fidelity claim.
- R34's GKP diagnostics never assert `physical_muta_validated=True` (checked
  directly by R34); resource-only diagnostics do not by themselves establish
  a physical implementation of arbitrary logical MuTA.
- R47's convergence studies never set `certified=True` (checked directly);
  numerical grid/cutoff stability on the two numerical axes (cutoff,
  grid_points) does not remove finite-resource physical error on the
  resource axes (peak_width, envelope, peaks).
- R13's and R42's timings are **local, single-machine observations** (this
  Windows host only), not general performance or algorithmic-complexity
  claims, and not a claim that PhotoGraphiQML or MentPy is "faster" in
  general.
- R23's classifier result is scoped to the single-feature threshold task;
  see `ISSUES_FOUND.md` #3 for the composite two-feature boundary that did
  **not** train reliably — do not cite `MuTAClassifier` as validated for
  arbitrary composite decision boundaries.
- R27/R28's kernel-classification results make **no quantum-advantage
  claim**; classical baselines are fit on the identical raw coordinates.
- R50 reproduces only source-paper configurations that map unambiguously to
  the public API; unmapped learning curves are listed as unsupported.
- R54 does not invent kernel depth, triangle, shot, or physical-noise axes:
  the fixed Eq. 5 feature map does not expose them.
- R55's finite-system gradient samples are not evidence of an asymptotic
  barren plateau. R59's empirical fits apply only to the measured ranges.
- R56-R58 exercise the restricted signed-X physical path only. They do not
  imply support for arbitrary angles, loss, detector inefficiency, or soft
  adaptive decoding.
