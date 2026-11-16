# R68–R75: CV-native measurement-based learning (extension catalogue)

These studies validate `photographiqml.cv`:

- **Path A:** trainable Gaussian programs and their compiled finite-squeezing CV-MBQC channels.
- **Path B:** directly trainable weighted clusters (`CVCluster`).
- **Learning tools:** CV kernels, gradients and resource reports.
- **Experimental:** non-Gaussian Fock programs.

Like R60–R67, this is an **extension catalogue**. The canonical evidence contract of this suite still covers R1–R59 and R_PERF only. That contract consists of `publication.EXPECTED_IDS`, `EXPERIMENT_INDEX.md`, `EXPERIMENT_CONTRACTS.json`, `EXPERIMENT_MANIFEST.json`, the generated notebook and `STATUS.md`. Run these scripts individually. They write CSV, JSON, metadata (including the source revision and environment) and figures under `../results` and `../figures`, with an `R6x_` or `R7x_` prefix.

Design and audit: [CV_NATIVE_ML_AUDIT.md](../../docs/development/CV_NATIVE_ML_AUDIT.md).
Results and capability matrix: [CV_NATIVE_ML_FINAL_REPORT.md](../../docs/development/CV_NATIVE_ML_FINAL_REPORT.md).

## Experiments

| ID | Figure | Script | Question | Reference / oracle | Acceptance (declared in the script) | Cost |
|---|---|---|---|---|---|---|
| R68 | CVML-1 | `68_cv_architecture.py` | Architecture, drawn from trained, executable models | Pattern structure (A) | Resource counts match the structure; logical map exact | ~1 min |
| R69 | CVML-2 | `69_circuit_vs_mbqc.py` | Direct circuit vs compiled CV-MBQC: logical map, finite-squeezing channel, trajectories | `gaussian_channel(Pattern)`, trajectory ensembles (B) | Errors ≤ 1e-9; Y ∝ e^{−2r}; trajectories within 4 SE | ~3 min |
| R70 | CVML-3 | `70_gaussian_regression.py` | Affine and multivariate regression, held-out data, optimisers | OLS (A) | Median test MSE ≤ 2 × OLS + 1e-4 | ~2 min |
| R71 | CVML-4 | `71_nonlinear_classification.py` | Where nonlinear classification comes from | Linear and quadratic logistic regression (A) | Affine Gaussian model = linear logistic (±0.05); heads on blobs (±0.05); attribution descriptive | ~12 min |
| R72 | CVML-5 | `72_finite_squeezing_learning.py` | Finite squeezing × finite shots; ideal, shot-trained and noise-penalised training | Exact expected MSE from moments (A) | Sampled = exact within 4 SE; monotone in r and S; penalty lowers noise | ~1 min |
| R73 | CVML-6 | `73_gradients.py` | Complex-step vs finite-difference vs closed-form gradients; margin; depth statistics; CRN | Closed form (A) | Closed form ≤ 1e-12; CS vs FD ≤ 1e-6; margin finite; CRN variance below independent noise | ~3 min |
| R74 | CVML-7 | `74_resource_scaling.py` | Resources, simulation cost, noise vs size, accuracy–squeezing–size trade-off | Structure (A) | Counts follow structure; depth ≤ measurements; timings provisional | ~1 min |
| R75 | CVML-8 | `75_non_gaussian_learning.py` | Fock convergence, Wigner negativity, non-Gaussian regression, cubic-injection lowering | Dense matrix-exponential reference (B) | Engine = reference ≤ 1e-6 at cutoff 64; negativity zero only for Gaussian; injection normalised | ~20 min |

## Revision notes

Revisions made after a first run are disclosed in the scripts:

- **R71:** all models now use L-BFGS, after Adam left the Kerr classifier under-trained. The Kerr displacement is bounded to stay inside the Fock cutoff.
- **R72:** panel (d) shows the variance split, which explains why the homodyne classifier's accuracy is flat in r.
- **R73:** the squeezing sweep uses a covariance-dependent loss, because a mean-readout loss is independent of r.
- **R75:** the fixed Wigner-negativity threshold for weak gates was replaced by "zero for Gaussian, monotone in strength, clear at the strongest gate". Weak gates create real but tiny negativity (Hudson's theorem).

Run one study with, for example, `python 70_gaussian_regression.py` from this folder. The native extension is required for R75.
