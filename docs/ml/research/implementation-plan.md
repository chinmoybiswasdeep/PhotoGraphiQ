# Architecture, validation and delivery plan

Dependency stack: Piquasso -> PhotoGraphiQ -> PhotoGraphiQML. Logical MuTA is
a paper-derived target model; optical simulation is delegated to PhotoGraphiQ.
Build and validate the logical model first. GKP resource diagnostics and an
explicit unsupported physical execution boundary follow. CVMuTA remains a
separate derivation proposal until the bridge is validated.

Files: `src/photographiqml/ansatz/{triangle,muta}.py`, `logical.py`,
`parameters.py`, `gkp.py`, `training.py`, `models.py`, `kernels.py`,
`diagnostics.py`, `validation.py`, and a small `__init__.py`.
Support files: pyproject.toml, tests/{muta,mentpy_reference,gkp,training},
docs/research and tutorials, examples, experiments/paper_reproduction,
scripts, notebooks, CI, contribution policies and paper draft.

Public API proposal: MuTA, TriangleNeuron, GKPBridge, MuTAKernel,
MuTAClassifier, MuTARegressor, QuantumInstrumentModel, Trainer, Parameter.
MuTA defaults to explicit logical execution. `representation='gkp'` never
silently falls back to qubit simulation. Logical angles use radians, inputs
shape (2**n,), batches (N,2**n), and named parameter dictionaries or vectors
in documented semantic order. Predictions and histories have explicit shapes.

Tests: topology counts; exact semantic edge mapping and trainability
discrepancies; Table I and B2; normalized random states; all branch corrections
on small cells; identity layer inclusion; reference probabilities and
density matrices; parameter binding/freezing/serialization; gradients against
finite differences; Adam training; kernel equation and PSD; instrument
completeness; classifier and regressor fitting; GKP projection diagnostics.
Do not label resource checks as full physical GKP execution convergence.

Documentation plan: explain representation boundaries first; then triangle,
flow, trainability, data shapes, fitting, kernels, instrument semantics,
resource diagnostics and limitations. Tutorials must execute supported paths;
do not create fictional output for blocked physical APIs. Curated notebooks
cover quickstart, cells, gate learning, kernel and GKP diagnostics.

Reproduction plan: prioritize analytic Table I/B2 and depth inclusion, then
Fig. 3 logical gate learning and Eq. 5 kernels. Record seeds, configuration,
timings and differences from the published protocols. Resource-noise,
teleportation-instrument training, DQN and finite GKP figures must be reported
as pending until implemented and run. Full 30 tutorials/10 projects and all
paper figures remain release acceptance targets, not assumed accomplishments.
