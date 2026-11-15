> Historical v0.1 assessment. See the [current release report](release-report.md).

# Release assessment

**Research foundation, not a complete physical v0.1 release.** The faithful
logical MuTA stage is implemented and tested. The GKP stage provides actual
PhotoGraphiQ finite-resource preparation and diagnostics, but general physical
MuTA execution remains unsupported. CVMuTA is a separate derivation proposal.
The missing logical measurement/injection and decoding protocols are a
scientific blocker to claiming the full requested stack is complete.

## Repository and changed files

```text
src/photographiqml/
  ansatz/triangle.py         one paper layer and semantic geometry
  ansatz/muta.py             composed logical model, flow and serialization
  logical.py                Appendix B ideal logical target execution
  parameters.py             metadata around PhotoGraphiQ Parameter
  gkp.py                    finite resource bridge and explicit stage boundary
  training.py               deterministic Adam/SGD/L-BFGS and histories
  models.py                 encoding, classifiers, regressors and instruments
  kernels.py                paper Eq. 5 kernel
  diagnostics.py            logical concurrence and pure-state QFI
  expressivity.py           Pauli Lie closure and local state Fisher matrix
  validation.py             independent contraction and optional MentPy adapter
tests/                      logical, training, GKP and reference tests
docs/research/              paper analysis, both audits, mapping, design plan
docs/tutorials/             30 concise tutorials, including research boundaries
examples/tutorials/         30 executable examples
notebooks/                  5 executed notebooks
experiments/
  paper_reproduction/       two Fig. 3 logical gate targets, 20 seeds each
  kernel_classification/    three datasets with classical baselines, 3 seeds
scripts/                    figures, learning materials and validation
paper/PhotoGraphiQML.tex     research draft
.github/workflows/ci.yml     core Python matrix and pinned-reference job
```

The [file manifest](file-manifest.txt) records the exact changed/created
repository paths. README.md was modified. The pre-existing license and supplied
paper PDF were preserved. The `.venv` and `.references` directories are local,
ignored tooling/resources; upstream source is not copied into this package.

## Paper mapping and reference matrix

The [paper notes](research/muta-paper-notes.md) cover all main sections and
Appendices A–E. The [mapping table](research/muta-mapping.md) connects each
concept to implementation and validation. The paper's bias is architectural
and measurement-angle restriction, not a hidden classical offset.

| Comparison | Cases | Outcome |
|---|---|---|
| Exact semantic edges, inputs, outputs, node counts | (1,1), (2,1), (3,1), (2,2), (3,2), both one_column settings | Pass |
| Measurement ordering | All structural cases | Reference total order satisfies our flow dependency DAG |
| Unrestricted trainability | All structural cases | Exact agreement |
| Restricted trainability | Both flags and all structural cases | Expected upstream discrepancy explicitly asserted |
| Logical density matrices | 12 width/depth/restriction configurations | Pass at 1e-10 tolerance |
| Kernel overlaps | Two independent feature inputs | MentPy agreement |
| Instrument branch probabilities and states | Final Z instrument | MentPy agreement; not learned teleportation |
| Adaptive outcomes | Every one of 256 branches of a two-wire cell | Corrected states agree, probabilities sum to one |
| Table I and depth inclusion | XX gate, Euler gate, widths 1–3 | Pass |
| Raw CV states vs MentPy | None | Deliberately not a valid comparison |

MentPy commit: `63c3d83e495696b4491c9d376dab7e3e6cf6c863` (0.1.0a15).
PhotoGraphiQ commit: `f15957d297f65f1e4761007e189f11119282dfdd` (0.3.0).
Piquasso: 8.0.1. See [MentPy audit](research/mentpy-audit.md).

The upstream restriction mutates only a trainable-node list. Cross-edge
insertion and stacking rebuild that list from still-trainable Ment objects.
This package fixes the measurement values explicitly and preserves those
constraints. Numerical reference comparisons explicitly repair reference
measurements to compare identical logical models. This discrepancy is never
silently interpreted as faithful agreement with the unmodified template.

## Validation and CI

- 88 tests passed on Windows with Python 3.14.4.
- Line coverage: 96.09% (565 of 588 executable statements).
- Core execution tested in a subprocess that deliberately blocks MentPy imports.
  Separately, 54 tests passed with the reference directory excluded and 90.82%
  coverage, confirming the core CI job can meet its threshold without reference tests.
- All 30 example scripts and five notebook kernels executed successfully.
- Ruff lint, formatting, mypy, wheel/sdist build, strict documentation build,
  tutorial execution and notebook execution all passed. Every check returned
  zero in the repository-root `validation-results.json`.
- Python 3.11, 3.12 and 3.13 are configured in CI, not locally executed here.
- Hosted GitHub CI, publication and deployment have not been performed.

`coverage.json` records the measured coverage. The release script checks the
90% threshold and fails on nonzero check statuses. Optional MentPy tests skip
when absent; core import does not load MentPy. The TeX draft is provided as
source; compilation of that draft is not a release check.

## Tutorials, projects, notebooks and figures

There are **30 concise tutorials**, **30 corresponding runnable scripts**,
**five executed curated notebooks**, and **two configured larger experiment
projects**. The ten larger projects requested in the brief are not complete.
Autodiff, paper instrument learning, full physical GKP and CVMuTA tutorials
explain missing capabilities and execute prerequisite/contract examples;
their presence is not evidence those features are implemented.

Six SVG figures were generated from scripts: logical triangle, composed MuTA,
concurrence, GKP resource projection, gate-learning curves and kernel
classification. All outputs are retained. No figure claims quantum advantage.

## Reproduced targets and observed results

Logical gate learning follows Fig. 3's target families, ten Haar states and
7/3 split, with **20 independent seeds per target** and 120 Adam steps.
The explicit learning rate and finite-difference gradient are local choices.
Mean final held-out infidelity (population standard deviation over runs):

| Target | Mean | Standard deviation |
|---|---:|---:|
| Haar single-qubit gate on first wire | 3.3990e-5 | 1.3028e-4 |
| IsingXX(pi/2) | 3.3963e-6 | 2.0807e-6 |

All individual runs remain in
`experiments/paper_reproduction/results/gate_learning.json`, including the
less-converged Haar seed 19. The largest MentPy checkpoint density discrepancy
is **1.67e-15**. MentPy was evaluated at checkpoints, not independently retrained;
these are numerical model comparisons, not separate optimizer replications.
The 40 training runs took about 75 seconds total in this local environment.

Kernel classification uses Eq. 5 with 160/40 splits and three seeds. Mean
held-out accuracy with raw feature coordinates:

| Dataset | MuTA kernel | RBF SVM | Logistic regression |
|---|---:|---:|---:|
| Circles | 0.9583 | 1.0000 | 0.4583 |
| Blobs | 0.5750 | 1.0000 | 1.0000 |
| Moons | 0.7750 | 1.0000 | 0.8750 |

These are locally specified generators, not an exact Fig. 8 reconstruction.
The poor blobs result is retained; a periodic feature map on unscaled
coordinates need not preserve the class geometry. Tiny negative Gram
eigenvalues (order 1e-14) are roundoff. Per-seed confusion matrices and metrics
are in `experiments/kernel_classification/results/metrics.json`.

The paper's noisy-label/resource plots, trained QFI classifier, learned
teleportation instrument and discrete HEA search/DQN have not been reproduced.

## GKP and CV limitations

GKPResource projection, finite superposition preparation and a PhotoGraphiQ
Fock execution of a prepared resource are tested. Separate one-mode cutoff
and grid refinement checks pass. At cutoff 48 for width=envelope=0.4,
captured weights exceed 0.9999996, while the normalized codeword overlap is
about 0.0130. High captured weight therefore does not imply orthogonal
codewords or an ideal encoding channel.

No arbitrary logical XY physical instrument, magic-state injection lowering,
full decoder, correlated multimode gate convergence or logical confusion
matrix is supplied. General `representation='gkp'` execution raises an
actionable error; `GKPBridge.logical_target` is explicitly labeled logical.
The [CV proposal](research/muta-mapping.md) derives only a teleportation-step
ingredient. There is no exported CVMuTA implementation, Gaussian universality
claim or finite-resource monotonic-expressivity claim.

## Training and scaling limitations

Training is deterministic and full-batch. Automatic differentiation,
score-function estimators, physical noisy training, tied parameter groups,
parameter bounds, mixed logical inputs, wrapper/optimizer persistence and
general intermediate instrument controls are pending. Pure-state metrics
are not extended to unsupported representations. The simple supervised
wrapper uses one Z readout plus an affine/logistic head.

Local statevector runs ranged from approximately 0.13–0.20 ms for one wire
at small depth to 10.3 ms for six wires and four requested layers (24 paper
layers, 582 graph nodes, 576 parameters). Measurements use a logical frontier,
so statevector storage is 16*2^n bytes; the six-wire vector occupies 1024 bytes.
This excludes graph metadata, temporary arrays and process overhead; it is
not a measured peak-memory claim. Dense unitary extraction scales as 4^n and
is capped at ten wires. Fock multimode performance is not benchmarked here.

## Recommended next work

The critical next step is a validated GKP logical measurement/injection and
decoding protocol in PhotoGraphiQ. Keep physical lowering disabled until
that exists, then validate decoded statistics and independent convergence
axes. In parallel, complete the remaining logical paper reproductions and
model-training contracts. Introduce CVMuTA only as a separate mathematically
derived family after the GKP stage. See the repository ROADMAP.md for the
remaining release acceptance work.
