# CV-native machine learning: source audit

Audit of the `upgraded` branch before `photographiqml.cv` was added. Starting revision
`c8d053a8c388d075086a7b642d52eddf3b318568` ("arbitary angle gkp"); the working tree was clean
(`git status --short` empty) when this work began.

## 1. Reusable APIs

| Need | Existing API | Reused how |
|---|---|---|
| Gaussian gate matrices, hbar = 2 | `photographiq.gaussian`: `omega`, `rotation`, `squeezing`, `cz`, `beamsplitter`, `embed`, `teleportation_matrix`, `wire_channel` | Independent references in tests. The CV module writes its own complex-step-safe copies (`_rotation`, ...), which the tests check against these. |
| Trainable optical circuit and its MBQC compilation | `photographiq.Circuit` (`rotate`, `squeeze`, `gaussian`, `cz`, `beamsplitter`, `displace`), `compile_circuit`, `decompose_symplectic` | `GaussianProgram.circuit()` builds the `Circuit`, and `.compile(squeezing=r)` produces the executable Pattern. The fast channel mirrors the compiler's lowering step by step: the same four-or-five-step decomposition, the same SUM-based beam splitter, and the same Fourier steps around CZ and displacement gates. |
| Exact unconditional channel of a fixed-angle Gaussian pattern | `photographiq.analysis.gaussian_channel` → `GaussianChannel(matrix, noise, displacement)` | Independent oracle for both compiled programs and clusters. It costs about 1 s for 100 measurements and about 10 s for 1000, so it is used for validation, not in training loops. |
| Trajectory execution with exact Gaussian conditioning | `photographiq.simulate`, `run_shots`, `GaussianBackend.measure` (Schur complement, outcome postselection, densities) | Conditional-vs-ensemble validation of compiled patterns and clusters. |
| Weighted CV graphs and flow | `CVGraph`, `certify_cv_flow`, `flow_pattern` (Booth–Markham real-linear CV flow, supplied total order) | The `CVCluster` cluster is a `CVGraph`. Its corrections come from `certify_cv_flow`, and its executable pattern from `flow_pattern(graph, order, shears={node: cot θ})`. |
| Gaussian state functionals | `GaussianState.overlap` (Tr ρσ), `fidelity` (Uhlmann via Banchi–Braunstein–Pirandola), `purity`, `symplectic_eigenvalues` | `CVKernel` uses them directly. |
| Fock execution of non-Gaussian gates | `native-fock` / `native-mixed-fock` backends: `kerr`, `cubic_phase`, `ladder`, `quadratic_phase`, Gaussian inputs, norm-tolerance guard, `retained_norms` | `NonGaussianProgram.run`. |
| Fock moments and Wigner functions | `fock_analysis.quadrature_moment` (boundary-correct), `wigner`, `WignerGrid.negative_volume` | Readout features and the R75 diagnostics. |
| Finite-energy cubic injection | `photographiq.non_gaussian.cubic_injection`, `resource_injection`, `CubicPhaseResource` | Optional lowering of cubic gates. |
| Optimisers | `photographiqml.training.Trainer` (Adam, SGD, L-BFGS-B; gradient callable; validation and callback hooks), `History` | Every CV model trains through `Trainer`. |
| Pattern structure | `Pattern.dependencies()` (command DAG), `Pattern.schedule()` | Measurement depth and feedforward counts in `pattern_resources`. |
| Evidence I/O and figure style | `paper_experiments_ml/common.py` (`save_result`, `save_figure`, Times New Roman manuscript style) | R68–R75. |

## 2. Conventions

- **Quadrature convention:** ħ = 2, with [q_j, p_k] = 2i δ_jk.
- **Ordering and vacuum:** interleaved (q₁, p₁, …), with vacuum covariance I.
- **Squeezing:** `squeezing(r)` = diag(e^{−r}, e^{r}), so positive r squeezes q. Resource modes are prepared p-squeezed with `GaussianInput.squeezed(−r)`, giving variances (e^{2r}, e^{−2r}).
- **CZ and quadratic phase:** `cz(g)` maps p₁ += g q₂ and p₂ += g q₁. `QuadraticPhase(s)` maps p → p + s q.
- **Homodyne:** a homodyne measures q cos θ + p sin θ. The teleportation step `T(k) = [[−k, −1], [1, 0]]` corresponds to θ = atan2(1, k), with the correction −m / sin θ.
- **Fock gates:** Kerr is exp(iκn²) and the cubic phase exp(iγq³/6).

The CV module uses all of these unchanged.

## 3. Supported operations

- **Gaussian** (all Gaussian backends): rotation, squeezing, beam splitter, weighted CZ, quadratic phase, displacement, loss, and homodyne, heterodyne and general-dyne measurement, including noisy homodyne.
- **Non-Gaussian** (Fock backends only): Kerr, cross-Kerr, cubic phase, photon addition and subtraction, photon counting, cat and GKP inputs, and resource injection. The finite-energy GKP logical measurements are separate (`encoded.py`).

## 4. Differentiation before this work

- `photographiq.autodiff` provides JAX execution of fixed-topology, fixed-outcome Fock patterns. It requires JAX and Piquasso, and has no Gaussian-moment path.
- `photographiqml.training.finite_difference` provides central differences.
- `physical_gradient` provides paired-CRN finite differences for the GKP physical MuTA.

There was no exact derivative for Gaussian moments. The new module adds complex-step derivatives instead of a JAX path. They need no optional dependency and run on the same NumPy code that the tests validate.

## 5. Existing ML models and benchmarks

The existing families are:

- **`MuTA`, `MuTAKernel`, `MuTAClassifier`, `MuTARegressor`:** ideal logical-qubit models.
- **`PhysicalMuTA`, `GKPBridge`, `lower_muta_to_gkp`:** finite-energy GKP lowering. Signed-X is the default; the XY family is new on this branch.

The benchmarks (R1–R59, R_PERF, and the R60–R67 extension) cover the logical and GKP families only. No model encoded data into CV states or trained CV-MBQC resources directly. `MuTAKernel` is a logical-qubit feature map.

## 6. Limitations found

- **Measurement angles:** `gaussian_channel` rejects outcome-dependent angles and nonlinear feedforward. The compiled patterns and flow patterns use fixed angles with affine corrections, so it applies.
- **Expression arithmetic:** `Circuit` gates and Pattern expressions evaluate to Python floats (`Expr.evaluate` casts to `float`). Complex-step derivatives therefore cannot run through `Pattern`, and the CV module keeps its own analytic propagation. Tests tie that propagation to the Pattern path.
- **Cost of full-pattern analysis:** cost grows with the number of commands. A two-mode beam splitter alone lowers to about 100 teleportation steps, and a 4-mode, 4-layer program to about 1500 measurements.
- **Flow certificates:** a certificate holds for one numerical weight assignment. `flow_pattern` freezes the weights, so changing a trainable weight requires a new certificate.
- **Fock truncation:** Fock backends raise an error when the retained norm drops below 1 − 10⁻³. Callers must choose a cutoff, or bound the energy, explicitly.

## 7. Finite squeezing in the existing code

- **Finite resource squeezing:** this is a per-node attribute of `CVGraph` and an argument of `compile_circuit`.
- **Added noise:** `wire_channel` gives the closed-form noise of a teleportation wire, diag(0, e^{−2r}) per step propagated through later steps.
- **Existing tests:** `tests/analytical/test_channels.py` checks compiled gate channels against the ideal matrices and exact wire noise.

## 8. Experimental evidence before this work

- **Compilation and channels:** compiled Gaussian gates and their channels, cubic and Kerr synthesis, native Fock/Gaussian accuracy, and CV-flow robustness, in the core R-series under `paper_experiments/`.
- **Learning:** no CV learning experiment existed.

## 9. Gaps that required new code

1. A data-encoding layer that separates data features from trainable parameters (`CVFeatureMap`, `Feature`).
2. Trainable Gaussian programs with both an ideal and a physical (compiled-channel) propagation, cheap enough for training (`GaussianProgram`, `layered_program`).
3. A directly trainable weighted cluster with homodyne angles as parameters, flow recertification when weights change, and an exact channel (`CVCluster`).
4. Readouts, explicit classical heads, an exact shot-noise estimator, losses, and resource penalties (`CVRegressor`, `CVClassifier`).
5. Exact derivatives (complex step) and a training history with provenance (`CVTrainingHistory`).
6. Closed-form Gaussian kernels on top of `GaussianState` (`CVKernel`).
7. Non-Gaussian programs on the Fock engine, with an optional cubic-injection lowering (`NonGaussianProgram`).
8. Resource reports computed from the executable patterns (`resource_report`, `pattern_resources`).
9. An independent dense matrix-exponential Fock reference (`tests/validation/cv_fock_reference.py`).

## 10. Naming

The repository already reserves "CVMuTA" for a planned proposal: MuTA's triangle-cell
topology with continuous-variable cells, with its own cell-map derivation. That proposal is
roadmap item 3 and is listed as **Planned** in `COMPLETE_FUNCTIONALITY.md` Section 28, and
`MuTA(representation=...)` still rejects it.

The directly trainable cluster added here is a different construction: a generic weighted
grid. To avoid claiming that proposal, it is named `CVCluster`. `MuTA` and its error messages
are unchanged.

## 11. Deliberately not duplicated

- No new numerical engine: execution uses the existing backends.
- No new graph or flow algorithm: flow comes from `photographiq.cvflow`.
- No new optimiser: training uses `Trainer`.
- No new Gaussian state functionals.

The only duplicated logic is the compiler's lowering arithmetic, reproduced as channel arithmetic. It is pinned to `gaussian_channel(compile_circuit(...))` by tests, so the two cannot drift apart unnoticed.
