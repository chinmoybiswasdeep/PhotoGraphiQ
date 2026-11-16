# CV-native measurement-based quantum machine learning: final report

## Summary

`photographiqml.cv` adds a CV-native learning family to PhotoGraphiQML, separate from logical MuTA and the GKP physical MuTA:

- Classical data is encoded into CV optical states.
- The states are processed by Gaussian programs in one of two forms:
  - a direct trainable circuit, or the unconditional channel of its compiled finite-squeezing CV-MBQC pattern (Path A);
  - a directly trainable weighted cluster with trainable homodyne angles and CV-flow feedforward (Path B, `CVCluster`).
- An experimental non-Gaussian path runs Kerr and cubic-phase programs on the native Fock engine.
- Every step except the classical head runs on PhotoGraphiQ's own patterns and engines.

**Validation and evidence:**

- Validation is against independent references: closed forms, `gaussian_channel` of the executed Pattern, trajectory ensembles, an independent dense Fock `expm` reference, and OLS/logistic baselines.
- Eight extension experiments, R68–R75, produce figures CVML-1 to CVML-8.
- No quantum advantage is claimed. With displacement encoding and a linear mean readout, a Gaussian model is affine in its inputs and efficiently classically simulable. The tests and R71 verify this.

## 1. Repository revision and starting state

- **Branch:** `upgraded`, confirmed by `git branch --show-current`.
- **Starting revision:** `c8d053a8c388d075086a7b642d52eddf3b318568`. `git status --short` was empty, so there were no pre-existing modifications.
- **Git operations:** none that write. Nothing was staged, committed or pushed, and no branch was created or switched. Every change is in the local working tree, unstaged and uncommitted.
- **End state (read-only checks after all runs):**
  - Branch `upgraded`; HEAD is still `c8d053a8c388d075086a7b642d52eddf3b318568`.
  - `git diff --cached` is empty.
  - Six tracked files are modified: `README.md`, `docs/COMPLETE_FUNCTIONALITY.md`, `docs/ml/guide.md`, `docs/ml/index.md`, `mkdocs.yml` and `src/photographiqml/__init__.py`.
  - Everything else is new and untracked: `src/photographiqml/cv/`, `tests/ml/cv_native/`, `tests/validation/cv_fock_reference.py`, `paper_experiments_ml/15_cv_native/`, `examples/ml/cv_native/`, `docs/ml/cv-native/` and the two development documents.
  - The R68–R75 results and figures are 48 new untracked files under `paper_experiments_ml/results/` and `paper_experiments_ml/figures/`.

## 2. New and modified files

**New library code (`src/photographiqml/cv/`):**

| File | Contents |
|---|---|
| `__init__.py` | Exports |
| `gaussian.py` | `CVFeatureMap`, `Feature`, `GaussianProgram`, `layered_program`; ideal and compiled-channel propagation |
| `mbqc.py` | `CVCluster` |
| `learning.py` | `CVRegressor`, `CVClassifier`, `CVTrainingHistory`, `complex_step_gradient`, `central_difference` |
| `kernels.py` | `CVKernel` |
| `resources.py` | `resource_report`, `pattern_resources` |
| `nongaussian.py` | `NonGaussianProgram` |

**Modified:**

- `src/photographiqml/__init__.py`: exports the `cv` subpackage. No existing name changed.

**New tests:**

- `tests/ml/cv_native/test_cv_gaussian_programs.py`
- `tests/ml/cv_native/test_cv_cluster.py`
- `tests/ml/cv_native/test_cv_learning.py`
- `tests/ml/cv_native/test_cv_kernels_nongaussian.py`
- `tests/validation/cv_fock_reference.py`: independent dense matrix-exponential Fock reference.

**New experiments (`paper_experiments_ml/15_cv_native/`):**

- `README.md` and `cv_support.py` (shared helpers).
- `68_cv_architecture.py`, `69_circuit_vs_mbqc.py`, `70_gaussian_regression.py`, `71_nonlinear_classification.py`, `72_finite_squeezing_learning.py`, `73_gradients.py`, `74_resource_scaling.py`, `75_non_gaussian_learning.py`.
- Outputs: `paper_experiments_ml/results/{csv,json,...}/R68_*`…`R75_*` and `paper_experiments_ml/figures/{pdf,png,svg}/R68_*`…`R75_*`.

**Documentation:**

- New: `docs/development/CV_NATIVE_ML_AUDIT.md` and this report.
- New: `docs/ml/cv-native/` (15 pages).
- New: `examples/ml/cv_native/01…08_*.py` (tutorials).
- Updated: `docs/ml/guide.md`, `docs/ml/index.md`, `mkdocs.yml`, `README.md`, `docs/COMPLETE_FUNCTIONALITY.md`.

**Untouched:**

- `MuTA`, `MuTAKernel`, `MuTAClassifier`, `MuTARegressor`, `PhysicalMuTA`, `GKPBridge`, the GKP lowering and the signed-X models.
- All core `photographiq` modules.
- The canonical R1–R59 evidence contract and the historical results.

## 3. Architecture

```
classical data x ──► CVFeatureMap ──► Gaussian program ──────────────► output moments ──► readout ──► classical head ──► ŷ
 (classical)        (quantum: D, S    (quantum)                        (mean d, cov V)    q/p/qp/     (affine /
                     or R encoding)    Path A: GaussianProgram                              moments,     logistic /
                                         ideal  : d→Sd+t, V→SVSᵀ                            S-shot       softmax), or the
                                         physical: compile_circuit                          means        physical homodyne
                                                   → X=S, Y(r)=e^{−2r}Y₁                                head Φ((μ−b)/σ)
                                       Path B: CVCluster (w×c grid)
                                         homodyne angles θᵢ, CZ rungs g,
                                         CV-flow corrections, X(θ,g), Y(r)
                                       NonGaussianProgram (experimental):
                                         Kerr / cubic on the Fock engine
                ▲                                                                                              │
                └──────── optimiser (Trainer: Adam / SGD / L-BFGS) ◄── complex-step or central-difference ◄──┘
```

Figure CVML-1 (R68) draws this pipeline from two trained models, annotated with their measured resource reports.

## 4. Mathematical conventions

- **PhotoGraphiQ conventions, used unchanged:**
  - ħ = 2, with [q_j, p_k] = 2iδ_jk.
  - Interleaved ordering (q₁, p₁, …, q_n, p_n), with vacuum covariance I.
  - A unitary acts as d → Sd + t, V → SVSᵀ.
  - A channel acts as d → Xd + t, V → XVXᵀ + Y.
- **Physicality:**
  - The ideal maps are verified symplectic, SΩSᵀ = Ω, to 1e-10 or better.
  - Physicality is enforced by construction: every trainable parameter enters through a gate matrix, never through a free matrix entry.
- **Intended map vs physical channel (Path A):**
  - The intended map S and the finite-resource channel (X, Y) are separate outputs.
  - X equals S exactly (R69: largest deviation 7.6e-14 over 36 random programs).
  - Y = e^{−2r}·Y₁ exactly.
- **Conditional vs ensemble quantities:**
  - Conditional means, conditional covariances, measurement records and ensemble quantities are kept apart.
  - The conditional covariance does not depend on the outcomes; the conditional mean does.
  - The ensemble covariance equals the conditional covariance plus the spread of the conditional means. The tests check this against trajectory runs on the Gaussian backend.

## 5. API examples

```python
from photographiqml.cv import CVCluster, CVClassifier, CVFeatureMap, CVKernel, CVRegressor, layered_program

# Path A: trainable Gaussian circuit executed through its compiled finite-squeezing CV-MBQC channel
model = CVRegressor(layered_program(2, 1), mode="physical", squeezing=1.0, epochs=120).fit(X, y)
model.predict(X_test, shots=100, seed=0)           # S-shot homodyne sample means
model.resource_report(X)                            # resource modes, measurements, depth, dB, ...

# Path B: directly trainable weighted cluster with trainable homodyne angles
clf = CVClassifier(CVCluster(2, 3, squeezing=1.0), head="homodyne", mode="physical").fit(X, labels)

# Gaussian kernel for classical kernel methods
gram = CVKernel(CVFeatureMap(2, scale=1.3), kind="overlap").gram_matrix(X)
```

All eight tutorials in `examples/ml/cv_native/` run in under a minute and check their key claim with asserts.

## 6. Implemented capabilities

**Data encoding (`CVFeatureMap`):**

- Displacement (q or p), squeezing (bounded) and rotation encodings.
- Data features (`Feature`) are kept distinct from trainable parameters (strings).
- Data can be re-uploaded through `Feature` arguments on later gates.
- The mean input photon number is reported.

**Path A: trainable Gaussian programs:**

- Gates: rotation, squeezing (optional `bound·tanh` budget), shear, displacement (optional bound), beam splitter and weighted CZ.
- `layered_program` builds a standard ansatz.
- `mode="ideal"` gives the intended symplectic map. `mode="physical"` gives the channel of the compiled pattern.
- `circuit()` and `gate_pattern()` return the executable objects.

**Path B: `CVCluster`:**

- A w × c grid with trainable homodyne angles. The default bounded parameterisation covers one period without the singular angle.
- Rung weights can optionally be trained, with the CV-flow certificate recertified for each weight tuple and uncertifiable layouts rejected.
- The exact unconditional channel and the executable `flow_pattern` are both available.
- Graph topology, edge weights, squeezing, homodyne angles and input encoding are separate parameters with separate roles.

**Readouts and heads:**

- Readouts: q, p, qp or moments.
- Heads: affine (regression); logistic or softmax (classification); the physical homodyne head (Born probability of a binned q outcome).
- Quantum and classical parts are labelled throughout.

**Training:**

- The existing `Trainer` (Adam, SGD, L-BFGS-B).
- Gradients: complex-step (exact to rounding; the default for Gaussian programs), central differences (Fock programs) or a user callable.
- `CVTrainingHistory` records parameters, loss, validation loss, gradient norm, runtime, simulator evaluations and settings (mode, squeezing, shots, seed, steps, optimiser, penalty).

**Shot-aware learning:**

- The S-shot sample mean is sampled exactly as μ + chol(C/S)z with common random numbers, giving a pathwise estimator.
- `qp` readouts split the shots.
- The physical homodyne head can be evaluated from S shots.

**Finite-squeezing-aware objectives:**

- Ideal training followed by physical evaluation.
- Physical (shot) training.
- Noise penalty (mean added noise, vacuum units) and squeezing penalty (Σ sinh² r, photons). With a penalty, the task MSE is divided by Var y, so both terms are dimensionless.

**Kernels (`CVKernel`):**

- Hilbert–Schmidt overlap (PSD), normalised overlap, and Uhlmann fidelity, which is distinct for mixed states.
- Ideal or finite-squeezing processing of the states.

**Resource accounting (`resource_report`, `pattern_resources`):**

- Resource modes, entangling edges, weighted edges, measurements, measurement depth and feedforward corrections.
- Peak live modes and trajectory memory; moment memory.
- Resource squeezing (r, dB), gate squeezing, added noise, input photons and runtime per prediction.
- Fock cutoff and dimension for non-Gaussian programs.

## 7. Experimental capabilities

- **`NonGaussianProgram` (Kerr, cubic phase) on `native-fock`:**
  - Cutoff is mandatory, and the engine refuses truncation beyond its norm tolerance. The minimum retained norm is reported.
  - Training uses central differences.
- **Cubic gate lowering to finite-energy cubic-phase resource injection:**
  - The lowering tracks the ancilla preparation, the inverse SUM, the ancilla homodyne outcome, and the quadratic-phase and displacement feedforward.
  - `injection_channel` integrates the outcome record into the unconditional output.
  - This is an approximate conditional filter, not an exact gate. No heralding is needed: every outcome is accepted and corrected, so the heralding probability is 1.

## 8. Unsupported or restricted capabilities

These are deliberately rejected or not implemented:

- **Shots with non-mean readouts:** shot-trained `moments` readouts, the homodyne head with training shots, and non-Gaussian programs with shots raise `ValueError`. The sample means of non-Gaussian outcomes are not Gaussian, so the exact shot model does not apply.
- **No MBQC lowering for Kerr:** Kerr runs as a Fock gate. Only cubic gates have an (approximate) lowering. No exact universal non-Gaussian compilation is claimed.
- **No adaptive angles in clusters:** cluster homodyne angles are fixed per evaluation, and feedforward corrections are affine displacements from CV flow.
- **No JAX path:** derivatives use complex steps on NumPy instead.
- **No GPU and no CUDA-Q** in this family.
- **CVMuTA stays planned:** the roadmap reserves "CVMuTA" for MuTA's triangle cells with CV gates. That remains **Planned**. `CVCluster` is a different construction and does not implement it.
- **Multiclass:** only through the classical softmax head. The physical homodyne head is binary.

## 9. Validation methodology

Every capability is compared with a reference it does not share code with, wherever one exists:

| Capability | Reference | Where |
|---|---|---|
| Encoders | Analytic moments and photon numbers | `test_cv_gaussian_programs.py` |
| Ideal Gaussian programs | Products of `photographiq.gaussian` matrices; gate-level Pattern executed on `gaussian` and `native-gaussian` | `test_cv_gaussian_programs.py` |
| Compiled CV-MBQC channel | `gaussian_channel` of the actual compiled Pattern (an independent propagation in `photographiq.analysis`); trajectory ensembles (Schur-complement backend) | tests; R69 |
| Cluster channel | `gaussian_channel(flow_pattern)`; product of T(cot θ); trajectory ensembles (conditional vs ensemble) | `test_cv_cluster.py`; R69 |
| Homodyne conditioning | Hand-written Schur complement | `test_cv_cluster.py` |
| Gradients | Closed form (rotation chain); central differences; h-sweep | `test_cv_learning.py`; R73 |
| Shot estimator | Exact expected MSE from moments vs 200 sampled evaluations | `test_cv_learning.py`; R72 |
| Kernels | RBF closed form; unitary invariance; PSD; purity; fidelity vs overlap | `test_cv_kernels_nongaussian.py` |
| Non-Gaussian gates | Independent dense `expm` reference (`tests/validation/cv_fock_reference.py`) at two cutoffs; cutoff sweeps | tests; R75 |
| Learning | OLS, linear and quadratic logistic regression, cubic polynomial least squares, on held-out test sets | R70, R71, R75 |

Independence caveat: the Path A fast channel mirrors the compiler's lowering arithmetic. Its agreement with `gaussian_channel` is therefore a check of the mirroring, not of `compile_circuit` itself. The compiler's correctness is covered by the existing `tests/analytical/test_channels.py` and by the trajectory comparison in R69.

## 10. Numerical agreement

| Quantity | Result |
|---|---|
| Compiled X, t vs ideal S; Y vs `gaussian_channel(Pattern)` (R69, 1–3 modes × 1–4 layers × 3 seeds) | ≤ 7.6e-14 |
| Y·e^{2r} constant across r (R69) | ≤ 1e-9 relative |
| Trajectory ensembles vs channel (R69, 1000 shots, two programs) | max \|z\| = 1.44 |
| Cluster channel vs `flow_pattern` channel (tests) | ≤ 1e-11 |
| Kernel vs RBF closed form | ≤ 1e-13 |
| Fock engine vs `expm` reference (tests, cutoffs 40 and 60) | ≤ 2e-6 (⟨q⟩, ⟨p⟩), ≤ 2e-5 (variances) |
| Mirrored channel vs `gaussian_channel(Pattern)` cost (R69) | 26–833× faster, at equal results |

## 11. ML benchmark outcomes

All results are on deterministic held-out test sets. Figures CVML-3, CVML-4 and CVML-8 show them.

**R70: affine regression.** The quantum models match OLS, as expected for an affine target. They show trainability, not an advantage.

| Model | Test MSE (median of 5 initialisation seeds) | OLS |
|---|---|---|
| Layered, ideal (linear task) | 3.948e-4 | 3.948e-4 |
| Layered, compiled (r = 1) | 3.948e-4 | 3.948e-4 |
| 2×3 cluster (r = 1) | 3.903e-4 | 3.948e-4 |
| 3×3 cluster (r = 1), multivariate task | 4.361e-4 | 4.363e-4 |

Optimisers (2×3 cluster): L-BFGS converged in about 130 evaluations, against about 1000 for Adam. SGD was still decreasing after 120 epochs.

**R71: circles, test accuracy (one seed, L-BFGS).**

| Model | Source of nonlinearity | Accuracy |
|---|---|---|
| Linear logistic regression | none (classical) | 0.629 |
| Gaussian, displacement encoding, linear readout | none (affine) | 0.629 |
| (1) Classical x² preprocessing | classical encoding | 1.000 |
| (2) Rotation encoding | input-dependent Gaussian gates | 0.986 |
| (3) Squeezing encoding with variance readout | covariance observables | 1.000 |
| (4) Classical quadratic head | classical head | 0.929 |
| Quadratic logistic regression | classical | 0.929 |
| (5) Kerr (non-Gaussian) | quantum non-Gaussian | 0.829 |

- The affine Gaussian model reproduces linear logistic regression exactly.
- On linearly separable blobs, every head reaches 0.975, including the physical homodyne head.
- The Kerr model is the weakest of the nonlinear models on this task with this budget. No advantage is claimed.

**R75: nonlinear 1D regression** (target sin(2.5x)/2 + 0.4x², 45/15/30 split). All models share the displacement encoding, the qp mean readout and the affine head, and use L-BFGS with 40 iterations and one seed.

| Model | Test MSE |
|---|---|
| Gaussian D R S (affine) | 0.0407, equal to OLS (0.0407) |
| Gaussian with rotation encoding (input-dependent Gaussian gate) | 0.0119 |
| Cubic D(b) CP R (Fock cutoff 30) | 0.0229 |
| Kerr D(b) K R (Fock cutoff 30) | 0.0004 |
| Classical cubic polynomial least squares | 0.0005 |

- The single Kerr gate lifts the affine restriction, reaching the level of a classical cubic polynomial.
- The minimum retained norm during Kerr training was 1 − 6e-16.
- This is one controlled comparison, not evidence of an advantage.

## 12. Gradient verification (R73)

- **Closed form (rotation chain, depth 1–8):** ≤ 2.2e-16 absolute.
- **Complex step vs central differences (h = 1e-6):** ≤ 8.8e-9 relative. This covers the physical homodyne-head loss, compiled programs and clusters at depths 1–6, and r ∈ {0.25, 1, 2}, with 5 parameter draws each. The h-sweep shows the usual finite-difference V-curve, minimum around h = 1e-5. Complex step is flat at about 1e-16 for h ≤ 1e-9.
- **Near the homodyne margin** (θ up to 3.0915, within 0.05 rad of the singular angle): gradients stay finite and agree with finite differences to ≤ 4.2e-7 relative.
- **Gradient statistics with depth (40 random initialisations):**
  - The variance of ∂L/∂(first parameter) *grows* with depth: compiled 0.74 → 8.0e2; cluster 4.5e2 → 8.1e11 over depths 1–6.
  - Random angles near the margin produce large shears, hence large gain. These Gaussian models show exploding rather than vanishing gradients, not a barren plateau. The bounded parameterisation and the margin control this.
- **Shot gradients (S = 100):** the CRN pathwise estimator has 12–181× lower standard deviation than finite differences with fresh shot noise.
- **Not claimed:** deterministic differentiability of sampled outcomes. Only the reparameterised estimator is differentiated.

## 13. Finite-squeezing results (R72, R69, R74)

1. **Means are unaffected.** For flow-corrected Gaussian MBQC, the unconditional output mean equals the ideal mean at every resource squeezing (tests; R69). Finite squeezing enters only through the covariance. A mean-readout model with exact moments is therefore indifferent to squeezing. This was found, not assumed.
2. **Shot-limited error.** The shot-limited test error is (μ − y)² + wᵀC(r)w/S. The sampled estimator matches the exact expectation (max \|z\| = 1.36). The error falls monotonically with squeezing and with S. For the ideal-trained 2×4 cluster at S = 100 it goes from 0.0246 at 2.2 dB to 0.0122 at 17.4 dB.
3. **Shot-aware physical training helps.** At S = 100 it lowers the expected test MSE by 1.6–2.3× relative to ideal training: 0.0109 vs 0.0246 at 2.2 dB, and 0.0075 vs 0.0122 at 17.4 dB. It trades a small bias for lower output noise.
4. **The noise penalty works.** At every r it lowered the added channel noise relative to ideal training, improving the S = 100 error to 0.0177 at 2.2 dB.
5. **The physical homodyne classifier is encoding-limited.** Its accuracy is independent of r (single shot 0.677, many shots 0.975). Training settles on a gain of about 10 that amplifies signal and input vacuum noise together, so the resource noise is at most about 0.5% of the output variance.
6. **Added noise grows with size.** At r = 1 (8.7 dB) it reaches 39.8 vacuum units for a 4-mode, 4-layer compiled program with 1424 measurements (R74). In the R74 trade-off, larger clusters did not reduce the error. The best expected MSE (about 0.0079) needs 13 dB at 4–6 columns.

## 14. Non-Gaussian convergence results (R75)

- **Engine vs independent dense `expm` reference** (cutoff 120). Errors in ⟨q⟩ for two one-mode programs, D(x)·R·S·K(0.15) and D(x)·CP(0.05)·S, at input energies of 0.25–4 photons:

  | Program, input photons | Cutoff 8 | 12 | 16 | 24 | 32 | 48–64 |
  |---|---|---|---|---|---|---|
  | Kerr, 1 | 1.6e-4 | 2.2e-7 | 2.9e-9 | 1.6e-14 | 1.3e-15 | 1.3e-15 |
  | Kerr, 4 | refused | 1.6e-2 | 2.6e-4 | 6.3e-9 | 1.9e-14 | ≤ 1.7e-16 |
  | Cubic, 1 | 4.8e-5 | 4.3e-6 | 8.9e-8 | 1.7e-11 | 1.3e-14 | 5e-15 |
  | Cubic, 4 | refused | refused | 2.2e-4 | 6.8e-7 | 1.6e-9 | ≤ 9.5e-14 |

  - "Refused" means the engine raised an error because the retained norm fell below 1 − 1e-3. It did not return a silently truncated result.
  - 1 − (retained norm) tracks the error, for example 9.2e-4 for Kerr at 4 photons and cutoff 12.
- **Wigner negativity** of D(1)·K(κ)|0⟩ and D(1)·CP(γ)|0⟩, on a 121 × 121 grid over [−6, 6]²:
  - It is zero for Gaussian states (≤ 6e-18).
  - It grows strictly with gate strength: Kerr κ = 0.05 → 0.4 gives 1.7e-6 → 0.126; cubic γ = 0.02 → 0.2 gives 9e-18 → 1.75e-3.
  - Weak cubic gates create negativity below the grid resolution. This is consistent with Hudson's theorem, and it is the reason for the R75 criterion revision.
- **Cubic-injection lowering** (γ = 0.1, cutoff 48, 81 postselected outcomes integrated over [−8, 8]). Fidelity of the unconditional output with the exact gate output:

  | Resource squeezing r | Fidelity | Outcome integral |
  |---|---|---|
  | 0.2 | 0.865 | 1 − 3e-7 |
  | 0.4 | 0.904 | 1 − 7e-6 |
  | 0.6 | 0.932 | 1 − 1e-4 |
  | 0.8 | 0.953 | 1 − 8e-4 |

  - The minimum retained norm was 1 − 2e-4.
  - Higher resource squeezing improves the gate but needs larger cutoffs. The growing outcome-integral deficit at r = 0.8 is truncation of the more energetic resource.
  - Each point took about 1.7 s.

## 15. Runtime and memory

Timings are provisional: one Windows workstation, not idle (R74).

**Exact moment channel per evaluation:**

| Model | Size | Time |
|---|---|---|
| Compiled programs | 17 → 1424 measurements | about 0.3 ms → 27 ms |
| Clusters | up to 28 measurements | 0.07–5 ms |

**Other evaluations:**

- **Full-pattern `gaussian_channel`:** about 10 ms to 3 s (up to 525 measurements).
- **One Gaussian-backend trajectory:** up to about 1 s at 1424 measurements.

**Memory:**

- Moment propagation stores three (2n)² matrices.
- A trajectory holds at most 3 (compiled) or w·c (cluster) live modes. `peak_live_modes` and `trajectory_covariance_bytes` are in every resource report.
- Fock: a two-mode cutoff-24 program has dimension 300.
- Training wall-clock:
  - Gaussian fits: 1–3 s.
  - Two-mode Kerr classifier (L-BFGS, 100 samples): about 10 min.
  - One-mode Kerr regressor: about 95 s.
  - Cubic-injection channel integration: about 1.7 s per resource squeezing (81 outcomes, cutoff 48).

## 16. Test results

All counts are from runs in this working tree, Python 3.12.14, Windows 11.

| Run | Result |
|---|---|
| `tests/ml/cv_native` (new) | 48 passed, 0 skipped, 0 failed |
| `experiments/validate_release.py` (all release gates) | **All release gates passed**; the rows below are its components |
| └ pytest, whole suite | 1787 passed, 13 skipped, 0 failed (18 min) |
| └ coverage | 93.58% (floor 90%) |
| └ ruff check | pass (src, tests, examples, experiments, scripts) |
| └ ruff format --check | pass |
| └ mypy (src/photographiq, src/photographiqml) | pass |
| └ package build | pass |
| └ physics-evidence | pass |
| `mkdocs build --strict` | pass (exit 0) |
| Tutorials `examples/ml/cv_native/01…08` | all 8 pass their asserts, 2–10 s each |
| Experiments R68–R75 | all pass their declared acceptance criteria (revisions disclosed) |

**Coverage of the new modules:**

| Module | Coverage |
|---|---|
| `gaussian.py` | 96.9% |
| `resources.py` | 96.7% |
| `mbqc.py` | 94.0% |
| `nongaussian.py` | 93.9% |
| `kernels.py` | 92.3% |
| `learning.py` | 89.6% |

**Skips.** None of the 13 skips are in the new code. They are environmental or capability-based:

- MentPy not installed (2).
- Playwright not installed (1).
- CUDA-Q is Linux-only (1).
- Cross-backend tests for capabilities a backend lacks (7).
- Piquasso has no cross-Kerr (2).

**Python versions.** Only Python 3.12 is installed here. The repository declares 3.11–3.14, but the other versions were not tested. The new code uses no version-specific features beyond what the package already uses.

## 17. Known limitations

- **Simulability:** Gaussian CV learning with Gaussian inputs is efficiently classically simulable. R71 attributes every nonlinearity explicitly.
- **Single runs:** R71's attribution and R75's regression are one seed on one dataset. They are descriptive, not statistical, comparisons.
- **Fock cost:** Fock-engine training is two to three orders of magnitude slower than Gaussian training. L-BFGS line searches can request energies beyond the cutoff. The engine refuses these; bounded displacements prevent them.
- **Mirroring dependency:** the Path A fast channel mirrors the compiler. A future change to `compile_circuit`'s lowering must be mirrored too. Tests pin the two together.
- **Revisions after first runs:** acceptance criteria and set-ups were revised in R71, R72, R73 and R75. Each revision is disclosed in its script and in `15_cv_native/README.md`.
- **Warnings from `NonGaussianProgram.run`:** the engine's small-truncation `RuntimeWarning` is silenced. The minimum retained norm is reported instead, and truncation beyond 1e-3 still raises an error.
- **`resource_penalty`:** it returns 0 when no penalty is configured. Read added noise from `resource_report` or `program.channel(..., mode="physical")[2]` instead.

**Post-implementation review.** Writing and running the documentation exposed ten issues. Each is listed with its resolution:

1. **Trainable rung weights had zero complex-step gradient.** The flow certificate is solved for real weights, so under the default gradient the weights never moved. *Fixed:* `gradient="auto"` now uses central differences when `trainable_weights=True`, and `gradient="complex-step"` is rejected. A regression test checks that the weights train.
2. **Rung weights started near 0 instead of at `rung_weight`.** *Fixed:* `fit` starts every quantum parameter at the program's default, plus initialisation noise. A test covers this.
3. **The flow-certificate cache grew without bound.** *Fixed:* it is cleared after 256 weight tuples.
4. **The resource report showed r = 1 for ideal-mode models.** *Fixed:* `resource_squeezing_r` is now `None` for those models, and `structure_compiled_at_r` is reported separately. A test covers this.
5. **`penalty="squeezing"` was a silent no-op for clusters.** *Fixed:* it is now rejected.
6. **`resource_penalty` returns 0 without a penalty.** Documented (above); no change.
7. **The feature-map docstring was garbled.** *Fixed.*
8. **Prediction-time shots were allowed for Fock programs.** *Fixed:* they are now rejected.
9. **Truncation warnings are silenced.** Documented (above); no change.
10. **`CVKernel` did not validate `mode`, and one error message had a typo.** *Fixed.*

R70–R75 do not train rung weights and rely on default initialisation only through the angle defaults, which were already 0. Their results are therefore unaffected. Tutorial 03, which trains rung weights, was rerun, and its recorded output was updated in `docs/ml/cv-native/examples.md`.

## 18. Scientific novelty relative to existing CVQML tools

Existing CVQML frameworks (for example circuit-model CV neural networks with Kerr layers in Fock simulators) train *circuit* parameters of an ideal gate model. What is new here, within PhotoGraphiQ:

1. **Two physical execution levels for the same program.** The intended map and the finite-squeezing MBQC channel of a trainable Gaussian program are both available, the channel exact and cheap enough to train through.
2. **Direct training of a CV cluster's homodyne angles and edge weights.** Flow is recertified as the weights change.
3. **Analysis of how finite squeezing enters learning.** It affects only covariances; shot-aware training helps; homodyne classifiers are encoding-limited.
4. **Explicit attribution of nonlinearity** to its source.

These are software and analysis contributions. Nothing here demonstrates a computational advantage.

## 19. Outstanding work

- The CVMuTA proposal itself: triangle cells with CV gates and their own derivation.
- Adaptive (outcome-dependent) homodyne angles in trainable clusters.
- MBQC lowering of Kerr gates and validated non-Gaussian compiled training.
- Multi-seed statistics for R71 and R75.
- Integration of R60–R75 into the canonical evidence contract.
- A JAX execution mode, if larger models need reverse-mode gradients.

## 20. Capability matrix

| Capability | Implemented | Independently validated | Experimental | Unsupported |
|---|---|---|---|---|
| CV feature encoding (displacement, squeezing, rotation, re-uploading) | ✓ | ✓ analytic moments | | |
| Trainable Gaussian circuit (ideal) | ✓ | ✓ gate products; gate-level Pattern on two backends | | |
| Compiled finite-squeezing CV-MBQC channel (Path A) | ✓ | ✓ `gaussian_channel(Pattern)`; trajectories (R69) | | |
| Direct trainable cluster, homodyne angles (Path B) | ✓ | ✓ `flow_pattern` channel; trajectories; symplecticity | | |
| Trainable edge weights with flow recertification | ✓ | ✓ recertified-channel check (tests) | | |
| Adaptive (outcome-dependent) cluster angles | | | | ✓ |
| Conditional vs ensemble outputs | ✓ | ✓ trajectories vs channel | | |
| Regression (affine head) | ✓ | ✓ equals OLS (R70) | | |
| Binary classification (logistic, physical homodyne head) | ✓ | ✓ equals linear logistic on blobs and circles (R71) | | |
| Multiclass (classical softmax head) | ✓ | tested (accuracy ≥ 0.8 on a 3-class split) | | physical multiclass head |
| Complex-step gradients | ✓ | ✓ closed form, finite differences (R73) | | |
| Shot-aware training (CRN, mean readouts) | ✓ | ✓ exact expected error (R72) | | shots with moment readout or non-Gaussian programs |
| Noise and squeezing penalties | ✓ | ✓ penalty lowers noise (R72) | | |
| Gaussian kernels (overlap, normalised, fidelity) | ✓ | ✓ RBF closed form; PSD; unitary invariance | | |
| Resource reports | ✓ | ✓ counts vs structure (R68, R74) | | |
| Non-Gaussian Kerr / cubic programs (Fock) | ✓ | ✓ engine vs `expm` reference | ✓ | |
| Cubic-injection MBQC lowering | ✓ | normalisation checked; fidelity vs gate reported | ✓ | |
| Kerr MBQC lowering | | | | ✓ |
| GPU / CUDA-Q execution of CV learning | | | | ✓ |
| CVMuTA proposal (triangle cells with CV gates) | | | | planned |
| Quantum advantage | | | | not claimed |
