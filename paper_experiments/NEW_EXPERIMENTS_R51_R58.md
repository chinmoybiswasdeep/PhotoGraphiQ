# New manuscript experiments R51–R56 and R58

These experiments cover flow, calculus and GPU validation. R57 (interactive visualization) is
handled elsewhere. Each script runs from its own directory with `../../.venv/Scripts/python <script>`.
It declares its acceptance constants before it evaluates anything. It saves its results through
`common.save_result`, which writes the CSV, the JSON and the `.metadata.json` provenance file,
with the seed and the parameters in `extra`. Figures are written through
`common.save_figure` and `common.save_caption`, and every number in a caption is formatted from
the results. On any failure a script writes `<ID>_FAILURES.csv` and raises `AssertionError`.

All numbers below come from the runs recorded in `results/json/R5x_*.json` (date 2026-11-13).
These runs were made on a shared machine. Wall times include about 5–20 s of import time.

| ID | Script | Default runtime | Outcome |
|---|---|---|---|
| R51 | `17_flow_calculus/51_cv_flow_finding.py` | ~16 s | all checks passed |
| R52 | `17_flow_calculus/52_cv_mbqc_determinism.py` | ~26 s | all checks passed |
| R53 | `17_flow_calculus/53_qubit_flows_graphix.py` | ~32 s | all checks passed |
| R54 | `17_flow_calculus/54_measurement_calculus.py` | ~44 s | all checks passed |
| R55 | `17_flow_calculus/55_local_clifford.py` | ~7 s | all checks passed |
| R56 | `17_flow_calculus/56_zx_calculus.py` | ~13 s | all checks passed |
| R58 | `19_gpu/58_gpu_gaussian.py` (quick) | ~70 s | all accuracy/device checks passed; timings PROVISIONAL |

---

## R51: CV flow finding

**Question.** Does `photographiq.flows.find_cv_flow` decide CV-flow existence correctly? The
method is backward real-linear layering, following Booth and Markham (Quantum 7, 1146, 2023). The
check covers linear wires, 2D clusters, branched graphs, weighted random graphs and graphs that
have no certificate. It also asks whether the reported layers, correction coefficients and
impossibility witnesses are valid.

**Assumptions and model.** The model is `Model.CV_GAUSSIAN` open graphs with a weighted adjacency
A. Every measured node uses homodyne at atan2(1, k), and resource squeezing is r = 1. The flow
condition: for each measured u there is a c_u supported on non-input nodes measured later, with
(A c_u)_v = δ_uv for every v measured up to u. The corrections are q_t += c_q m and p_t += c_p m,
with m = record/sin θ.

**Backend, seed and parameters.** The experiment uses pure linear algebra plus the library's
float64 Gaussian determinism check (`gaussian` backend). The seed is 20261151. The graphs are:

- 13 named graphs;
- 30 weighted G(n, p) graphs with n = 4..7, signed weights with |w| log-uniform in [0.2, 3], and
  random inputs and outputs;
- 15 random weighted "brickwork" clusters.

`--full` raises this to 120 G(n, p) graphs and 60 clusters.

**Reference, independent of the library.**

1. An exact forward search over measured prefix sets in rational arithmetic, using a rank test
   for e_u ∈ range(M). It decides existence completely and runs in the opposite direction to the
   library's search.
2. A least-squares re-derivation of c_u on the reported order, with its residual and rank gap.
3. A re-check of the layer vectors: their support, their residual on u's own and earlier layers,
   and the correction map = (−c, −A c).
4. An exact check of every impossibility witness: yᵀ A[W, C] = 0 with y_u ≠ 0.

**Acceptance criteria.**

- Least-squares residual ≤ 1e-9 with rank gap 0.
- Layered residual ≤ 1e-10.
- Correction-map error ≤ 1e-12.
- Each named graph matches its expected decision.
- No certified graph lacks an order, and no impossible graph has one, according to the exact
  brute force.
- Every exact witness is valid.
- The library's determinism check passes on every certified graph.
- The uncertain case is resolved by `exact=True`.

**Actual results.**

- 58 graphs: 24 certified, 33 impossible, 1 numerically uncertain.
- The library agrees with the exact brute force on existence for 57/57 decided graphs.
- 20/20 exact witnesses are valid.
- Worst least-squares residual 6.2e-15; worst correction-map error 1.1e-16.

Named cases:

| Graph | Decision | Detail |
|---|---|---|
| Wires n = 3, 6, 10 | certified | 2, 5, 9 layers; residual 0 |
| grid 2×3 | certified | 2 layers, order [(0,0),(1,0),(0,1),(1,1)], residual 1.5e-16 |
| grid 3×4 | certified | 3 layers |
| branched_Y and branched_tree (1 input → 2 outputs, isometries) | certified | |
| hexagon (invertible over ℝ but not over GF(2)) | certified | 1 layer |
| unit bridge K₂,₂ | impossible | exact witness |
| weighted singular hexagon | impossible | exact witness |
| \|O\| < \|I\| | impossible | exact witness |
| isolated measured node | impossible | combinatorial obstruction |
| bridge with det 1e-10 | numerically uncertain | margin 4.75 decades; with `exact=True`: certified, `ideal_limit` only, correction gain 1.0e10 (warning) |

Rank-decision margins on all other named graphs are 14.4–15.4 decades. Of the 30 random G(n, p)
graphs, 29 were impossible. The brickwork family exercises the certified path on random weights:
15/15 certified.

**Figures.**

- `R51_cv_flow_examples` shows the weighted graph, the measurement layers and a table of
  correction coefficients, for a 2×3 cluster and a branched tree.
- `R51_cv_flow_summary` shows the decisions per family, and the independent residual against the
  largest correction coefficient. Tables are in `tables/R51_cv_flow_finding.md` and
  `results/csv/R51_cv_flow_corrections.csv`, which lists every coefficient.

**Limitations.**

- The brute force is exponential and is used only up to 16 measured nodes.
- float64 weights are treated as exact rationals.
- "Impossible" means that no admissible order exists for the real-linear CV-flow condition. It is
  not a proof of nondeterminism by other means.
- Homodyne efficiency is 1 and resources are lossless.

**Manuscript placement.** Main text, in the section on CV flow and determinism analysis. Use the
example figure and a one-line summary of the agreement and witness counts. The summary figure
can go to an appendix.

## R52: CV-MBQC determinism

**Question.** For CV-flow corrected Gaussian patterns, is the outcome-averaged channel (S, N)
equal to the intended S at every finite r, with N reported separately and going to 0 as r → ∞?
Is the corrected conditional mean independent of the outcomes in the ideal limit? Do the
library's channel and postselected branches match an independent model?

**Cases.**

- One-step teleportation with k = 0.35.
- A weighted 4-node wire with w = (0.7, 1.3, −0.9).
- 4-mode graph teleportation: inputs R_i, a weighted cycle cluster C_i, random shears.
- A weighted 2×3 cluster.
- A branched 1 → 2 isometry.

Corrections come from `find_cv_flow`, and the patterns are built with
`flows.cv.corrected_pattern`.

**Backend, seed and parameters.**

- `gaussian_channel`, `simulate(backend='gaussian', measurement_outcomes=…)` (exact
  postselection: each run is one branch) and 40 seeded sampled trajectories.
- r ∈ {0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5}.
- 6 postselected record sets per r, plus a 13-point record sweep for the teleportation case.
- Coherent inputs with a random mean. The seed is 20261152.

**Reference, independent of the library.** A linear Wigner-variable model:

- Variables are the input quadratures and (q, p) of each resource node, with covariance
  diag(V_in, e^{2r}, e^{−2r}).
- The model applies CZ shears, records m_u = k q_u + p_u and the correction map, in the
  library's convention.
- It gives S = L_in and the exact split N = e^{2r} N₊ + e^{−2r} N₋.
- It gives Gaussian conditioning with gain G = LΣRᵀ(RΣRᵀ)⁻¹.

There are also closed forms for S: T(k) per step, [[−k/w, −1/w], [w, 0]] per weighted step,
and C_Z(A)⊕F D(s) for graph teleportation.

**Acceptance criteria.**

- S matches the closed form and the model to ≤ 1e-10 (relative).
- N matches the model to ≤ 1e-9.
- Postselected moments match the model to ≤ 1e-9.
- The conditional covariance does not depend on the outcomes (≤ 1e-9).
- rank N₊ = |O| − |I|.
- N_finite > 0 at every finite r.
- Slope of d ln‖P_F N P_F‖/dr over r ≥ 1 in [−2.05, −1.95].
- Slope of d ln‖P_F G‖/dr over r ≥ 2 in [−2.15, −1.85].
- The spread of the seeded trajectories is within 40% of the model's prediction (a statistical
  check with 40 samples).
- `verify_cv_determinism` passes for every case.

**Actual results.**

- All 40 rows pass.
- S error ≤ 3.8e-16, N error ≤ 1.9e-16, postselected moments vs the model ≤ 1.4e-15.
- The conditional covariance does not depend on the outcomes (0 difference).
- N slope is −2.000 for every case.
- G slopes: teleport −1.987, weighted wire −1.908, graph teleport −1.981, cluster −1.957,
  isometry −1.987.
- The library's decay exponents κ are 0.9994–0.99993.

Finite noise is never zero:

| Case | max\|P_F N P_F\| at r = 1 | at r = 3.5 |
|---|---|---|
| teleport | 0.135 | 9.1e-4 |
| weighted wire | 0.216 | 1.45e-3 |

For the isometry, rank N₊ = 1, as expected.

The seeded trajectory spread agrees with the model, for example teleport at r = 1: 0.378 vs
0.385, and at r = 3.5: 0.0333 vs 0.0339.

**Figures.** `R52_cv_mbqc_determinism`:

- (a) N against r, with e^{−2r} for reference;
- (b) residual outcome gain;
- (c) E[q_out | record] against the record in units of its standard deviation, for
  r = 0.5, 1, 2, 3. The curves flatten towards the ideal (Sμ)_q.

**Limitations.**

- Homodyne efficiency is 1 and resources are lossless.
- Postselection is used only as a numerical certificate.
- At finite r the channel is noisy and is never called an ideal unitary.
- The slope of the gain is pre-asymptotic below r ≈ 2.

**Manuscript placement.** Main text, as the central determinism figure of the CV-MBQC section.

## R53: Causal flow, gflow and Pauli flow vs Graphix 0.4

**Question.** Does flow existence agree with Graphix 0.4, which uses the same semantics (same
planes and observables)? Do both libraries' flows satisfy the defining conditions and give
strongly deterministic patterns?

**Graphs.** There are 9 named cases:

- line;
- 2×3 cluster;
- gflow without causal flow;
- YZ state preparation;
- XZ at the start of a line;
- XZ inside an input line, which fails;
- X Paulis labelled in the XZ plane, where only Pauli flow exists;
- Y/Z Paulis;
- \|I\| > \|O\|, which fails.

There are also 80 seeded random open graphs per notion (n = 3..8). `--full` uses 300. The seed
is 20261153.

**References.**

- Graphix `OpenGraph.to_{causalflow,gflow,pauliflow}_or_none`.
- An independent set-algebra checker of F1–F3, G1–G3 and P1–P9.
- An independent dense Choi-vector statevector simulator of every branch, used to check strong
  determinism. It is applied to both our certificates and Graphix's.

**Acceptance criteria.**

- 100% existence agreement with Graphix.
- Named cases as expected.
- Negative answers come from complete algorithms.
- Neither library's flow violates any condition.
- Determinism deviation ≤ 1e-12.
- Our flow is never deeper than Graphix's.
- `analyze(cv, 'causal'|'gflow'|'pauli')` returns `unsupported`.

**Actual results.**

| Notion | Agreement with Graphix | Flow / no flow | Identical layers | Identical correction sets |
|---|---|---|---|---|
| Causal flow | 89/89 | 23 / 66 | 23/23 | 14/23 |
| gflow | 89/89 | 25 / 64 | 25/25 | 19/25 |
| Pauli flow | 89/89 | 31 / 58 | 29/31 | 19/31 |

- Flows are not unique, so different correction sets can both be valid. Both libraries' sets
  were valid in every case.
- 158 flow patterns were simulated over all their branches, with a worst deviation of 6.7e-16.
- Our flow was never deeper than Graphix's.
- The CV guard returns `unsupported` for all three notions.

**Figures.** `R53_qubit_flows_graphix`: (a) existence counts; (b) counts of jointly certified
flows, identical layers, identical correction sets and statevector-checked patterns. The table
is `tables/R53_qubit_flows_graphix.md`.

**Limitations.**

- Statevector checks are limited to 12 qubits.
- Correction sets are compared only by validity.
- Graphix is used only where the semantics coincide.

**Manuscript placement.** Main text, in a short paragraph in the qubit-flow subsection, with
the table. The figure can go to an appendix.

## R54: Measurement calculus

**Question.** Do standardization (to NEMC form) and signal shifting preserve the full branch
semantics? This covers adaptive measurements, corrections and dependency tracking. For CV, the
experiment validates only command classification and the extraction of affine corrections.

**Patterns.**

- 30 seeded random valid non-standard DKP patterns (`--full`: 120). Each has 2 inputs,
  3 ancillas and 2 outputs, with random planes, angles and s/t domains.
- 5 flow-derived patterns:
  - causal flow: line and 2×3 cluster;
  - gflow, including XZ and YZ planes;
  - Pauli flow with Y and Z measurements.
- CV patterns: the corrected wire and 2×3 cluster from `find_cv_flow`, plus a deliberately
  nonlinear feed-forward pattern.

The seed is 20261154.

**References.**

- An independent Choi-vector simulator. It measures the conjugated observable
  Z^t X^s O X^s Z^t, built as a matrix rather than from an angle table. Branches are keyed by
  the final signals.
- Graphix 0.4 statevector simulation with `FixedBranchSelector` on a random input.
- Library `run_branch`.
- For CV: direct evaluation of the original expressions on random records.

**Acceptance criteria.**

- Branch maps agree up to the phase of each branch to ≤ 1e-10.
- Probabilities agree to ≤ 1e-12, with no phase freedom.
- Graphix agreement: 1 − |overlap| ≤ 1e-10.
- Dependency DAG = domains, and the schedule is valid.
- Flow patterns are strongly deterministic.
- The library's determinism verdict matches the independent simulator.
- Negative control: dropping one output correction is detected (deviation > 1e-3) in ≥ 80% of
  the patterns.
- CV role counts are exact, the affine forms match the expressions to ≤ 1e-12, and the
  nonlinear feed-forward is reported as non-affine.

**Actual results.**

- 35 patterns with 312 branches; all 312 were also compared with Graphix.
- Worst deviations:
  - standardize: 5.6e-17;
  - signal shift: 5.6e-17;
  - probabilities: 8.3e-17;
  - library vs independent simulator: 2.8e-16;
  - Graphix: 5.6e-16.
- Rewrite rules applied: commute 629, EZ 88, MZ 52, EX 46, MX 31.
- Negative control detected in 20/21 patterns.
- All three CV patterns passed. The wire and the cluster have only affine corrections; the
  nonlinear pattern is reported as non-affine.
- Standardization can increase the depth of the measurement-dependency graph, because
  corrections are absorbed into measurement domains (MX/MZ). Signal shifting can then reduce it
  (figure panel b). This is expected DKP behaviour, not a defect.

**Figures.** `R54_measurement_calculus`: (a) branch deviations; (b) adaptive depth before and
after the rewrites; (c) counts of each rewrite rule.

**Limitations.**

- Dense simulation is used.
- Graphix is compared on the original patterns only, because Graphix has no S command.
- No CV measurement calculus is claimed.

**Manuscript placement.** Main text, as one paragraph on the qubit calculus. The figure can go
to an appendix.

## R55: Local Clifford and local complementation

**Question.**

- Does U_v = exp(−iπ/4 X_v) ∏_{w∈N(v)} exp(+iπ/4 Z_w) map |G⟩ to |τ_v G⟩, up to a global
  phase?
- Are the LC sequences found by `lc_equivalence` correct witnesses?
- Are non-equivalences confirmed?
- For CV: are the implemented `GaussianGraph` rules consistent with phase-space evolution?

**Graphs.** P4, S4, C5, K4, the 2×3 cluster and 6 random connected graphs. There are 51 (graph,
vertex) pairs and 44 random LC sequences of length 1–6. For CV there are 6 random 3–4-mode
weighted clusters with r ∈ [0.3, 1]. The seed is 20261155.

**Phase convention.** |G⟩ = ∏ CZ |+⟩^n, big-endian. States are compared up to a global phase,
and the phase is reported.

**References.**

- Our own graph state built from CZ phases.
- Our own edge toggling.
- U_v built from cos/sin of the Pauli matrices.
- The vector of Schmidt ranks over all bipartitions, which is an LU invariant.
- For CV: S V Sᵀ, and ideal homodyne by Schur conditioning of the covariance.

**Acceptance criteria.**

- State deviation ≤ 1e-12 for single LCs and for sequence witnesses.
- Replayed edges match the target.
- Expected non-equivalences are confirmed both by the library and by differing invariants.
- Positive control: S4 ~ K4 (the GHZ class).
- CV covariance and mean agree to ≤ 1e-10 (relative).

**Actual results.**

- Single LC: worst 1 − |⟨τG|U|G⟩| is 2.2e-16 in magnitude (the stored value is −2.2e-16, a
  rounding artefact). The library's Cliffords give 4.4e-16.
- Global phases are in {0, 1, 2} × π/4.
- All 44 sequences were recovered, with witness deviation ≤ 7.8e-16.
- Non-equivalences are confirmed by differing Schmidt-rank vectors:
  - P4 vs S4;
  - C5 vs K5;
  - 2×3 cluster vs K6.
- S4 ~ K4 was found.
- CV rules (cluster, CZ, shear, squeeze, rotate, beam splitter, `measure_q`, `measure_p`):
  covariance ≤ 1.3e-15, mean ≤ 2.3e-16.
- `cv_local_complementation` raises `NotImplementedError` and is recorded as not implemented.

**Figures.** `R55_local_clifford`: C5 → τ₀ → τ₂τ₀, with the toggled edges highlighted and the
global phase shown.

**Limitations.**

- The checks are exhaustive only for small graphs (the orbit search covers ≤ 10 nodes).
- There is no relabelling or isomorphism.
- No CV local complementation is implemented.

**Manuscript placement.** Appendix, or one sentence and the figure in the calculus section.

## R56: ZX calculus (qubit subset only)

**Question.** Do the implemented rewrites preserve the linear map exactly, including the
scalar? The rewrites are:

- spider fusion;
- identity removal, including H·H = I;
- colour change;
- Pauli state copy;
- 2×2 bialgebra;
- Hopf;
- cancellation of parallel Hadamard edges;
- removal of Hadamard self-loops;
- `simplify_spiders`.

The experiment also asks whether graph-state diagrams and MBQC branch diagrams have the intended
semantics.

**Seed and parameters.** The seed is 20261156. There are 150 random (diagram, rewrite) trials
(`--full`: 600), 9 random graph states (n = 3–5), and the 4 branches of a 3-qubit XY/YZ pattern.

**Reference.** An independent einsum evaluator. It builds X spiders directly from |+⟩/|−⟩
projectors, not by conjugating Z spiders with H. It is checked against closed forms (fusion,
colour change, identity), against CZ statevectors for graph states, and against `run_branch`
Kraus operators. **Scalars and phase:** the matrices before and after a rewrite are compared
exactly, with no global-phase or normalization freedom, because every rewrite records its exact
factor.

**Acceptance criteria.** Every deviation is ≤ 1e-12 relative, and at least 75 random rewrites
were actually applied. (The threshold is the smaller of `MIN_RANDOM_APPLIED` = 100 and half the
number of trials, so 75 in the default run.)

**Actual results.**

- 171 checks, all passing.
- Worst deviation after vs before a rewrite: 7.8e-16.
- Library vs independent evaluator: 2.2e-16.
- Closed forms: 2.2e-16.
- Graph-state and MBQC semantics: 2.3e-16.
- Random rewrites applied: colour change 67, fuse 35, Hopf 17, identity 14, Hadamard pair 7,
  self-loop 6, copy 4.
- The random generator never satisfied the bialgebra precondition (phase 0, exactly 2+2 legs).
  The bialgebra rule is covered by the textbook case only.
- Recorded factors: Hopf 0.5, bialgebra √2, Hadamard pair 0.5, Hadamard loop 1/√2, copy
  0.44+0.554j.

**Figures.** `R56_zx_calculus`: before and after diagrams for eight rules, with the scalar
factor and the deviation of each.

**Limitations.** This is a qubit ZX subset only. There is no CV-ZX, π-copy, or ZX local
complementation or pivoting. Evaluation is dense.

**Manuscript placement.** Appendix, with one sentence in the main text that states the scope.

## R58: GPU Gaussian acceleration (PROVISIONAL timings)

**Question.** How do the NumPy CPU path, the native C++/OpenMP batch kernels and the CuPy GPU
path compare in accuracy and in time? GPU time is separated into host preparation, H2D, kernel
and D2H. A further question is whether the GPU actually executed.

**Model, backend and parameters.**

- B copies of an n-mode squeezed CZ chain with thermal loss.
- Operations:
  - local beam splitter;
  - a dense random symplectic per state (Haar unitary × squeezers, built by batched QR);
  - homodyne conditioning on a given outcome (angle 0.4, η = 0.9);
  - purity.
- Quick grid: n ∈ {4, 16}, B ∈ {64, 1024, 8192}, repeats 5, warm-up 2.
- `--full`: n up to 64, B up to 16384, repeats 21.
- float64, plus explicit float32 GPU rows.
- The seed is 20261158.
- Device: NVIDIA GeForce RTX 4050 Laptop GPU (CC 8.9), CuPy 14.2.0, CUDA runtime 13.2; native
  max_threads 12.

**Method.**

- GPU stages are timed with CUDA events, synchronized.
- The kernel is timed on device-resident data with `on_device=True`.
- Host timing uses `perf_counter`.
- First calls are excluded.

**Device verification.** Every GPU row must have `execution.device == 'cuda:0'` and backend
`cupy`, and `pointerGetAttributes(...).type == memoryTypeDevice` for its device outputs.
Otherwise the run fails, so a CPU fallback is never counted.

**Acceptance criteria.** float64 engines vs NumPy float64 ≤ 1e-12 (purity ≤ 1e-11), and float32
≤ 1e-3.

**Actual results.**

- All 90 rows pass, and GPU execution was verified on every GPU row.
- Worst float64 deviation 3.3e-16; worst float32 deviation 3.1e-7.

PROVISIONAL timings, shared machine, n = 16, B = 8192, in ms:

| Operation | NumPy | native | GPU H2D / kernel / D2H | GPU end-to-end (f64) | GPU kernel f32 |
|---|---|---|---|---|---|
| local beam splitter | 85 | 41 | 16 / 6.3 / 38 | 50 | 5.1 |
| dense symplectic | 1449 | 497 | 40 / 28 / 27 | 104 | 18 |
| homodyne | 329 | 16 | 24 / 16 / 26 | 69 | 4.8 |
| purity | 115 | — | 18 / 7.7 / 0.04 | 32 | 1.9 |

- Host data preparation: 57 ms (state), plus 0.9 s for the 8192 random symplectics.
- At B = 64 the GPU kernel is slower than NumPy (ratio 0.04–0.8).
- Data transfers dominate the GPU end-to-end time for the O(B n) and O(B n²) operations.
- Native C++ beats GPU end-to-end for homodyne and the local beam splitter. GPU wins clearly only
  for the dense O(B n³) case.
- The stage times are not additive on a shared machine (for example, end-to-end 50 ms vs 61 ms
  summed for the beam splitter).

**Figures.** `R58_gpu_gaussian`: (a) separated stages at the largest configuration; (b) NumPy
time / GPU kernel time against B.

**Limitations.**

- Timings are provisional; rerun on an idle machine with `--full --repeats 21`.
- GeForce FP64 runs at 1/64 of the FP32 rate.
- The kernel time includes the library's validity-check syncs.
- CUDA-Q is not used and not claimed.

**Manuscript placement.** Software and performance section, or an appendix. Quote numbers only
from the idle-machine rerun.

---

## Recommended `run_all_safe` classification

| ID | Classification |
|---|---|
| R51, R52, R55, R56 | safe; default config < 30 s; no optional dependencies |
| R53, R54 | safe when Graphix is installed (`common.require_module` raises `SkippedExperiment` otherwise) |
| R58 | timing/GPU: exclude from the default safe run, or run `--quick` only when `photographiq.gpu.available()`; its accuracy and device checks are deterministic, its timings are not |
