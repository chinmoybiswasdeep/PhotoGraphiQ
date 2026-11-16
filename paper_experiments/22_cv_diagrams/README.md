# R61–R68: typed CV diagrams and CV-MBQC resource optimisation (extension catalogue)

These studies validate `photographiq.calculus.cv_diagrams`. It provides typed CV process
diagrams with hybrid Gaussian semantics, proved rewrites, Pattern translation, verified
resource optimisation, and a restricted non-Gaussian and GKP layer.

The IDs belong to the core suite (`paper_experiments/`, which previously ended at R60) and are
separate from the PhotoGraphiQML catalogue. Like the other extensions, the canonical evidence
contract is unchanged and the scripts run individually. Each script writes the following
under `../results` and `../figures` with an `R6x_cvzx_` prefix:

- CSV and JSON results; the JSON also includes `configuration`, `validation_report` and
  `source_revision`;
- a metadata companion (environment);
- the figure as PDF, PNG and SVG, plus a caption.

Theory and proofs: [docs/theory/CV_ZX_FOUNDATIONS.md](../../docs/theory/CV_ZX_FOUNDATIONS.md).
Audit: [CV_ZX_IMPLEMENTATION_AUDIT.md](../../docs/development/CV_ZX_IMPLEMENTATION_AUDIT.md).
Results: [CV_ZX_FINAL_IMPLEMENTATION_REPORT.md](../../docs/development/CV_ZX_FINAL_IMPLEMENTATION_REPORT.md).

| ID | Prompt item | Script | Question | Independent references | Cost |
|---|---|---|---|---|---|
| R61 | O1 (Figures 2–3) | `61_gaussian_rewrites.py` | Is every Gaussian rule exact in its class? | `gaussian_channel(Pattern)`, postselected trajectories, dense `expm` (Weyl phase) | ~1 min |
| R62 | O2 | `62_wire_shortening.py` | What do interior p measurements do to a cluster graph? Executable teleport-chain shortening. | `GaussianGraph` (MFvL) vs diagram conditioning | <1 min |
| R63 | O3 (Figure 4) | `63_random_compilation.py` | Random 1–3-mode circuits: compiled vs optimised resources, depth, noise, flow. | `gaussian_channel`, `find_cv_flow` | ~1 h |
| R64 | O4 + O5 | `64_scheduling_and_feedforward.py` | Feed-forward merging; commutation vs flow-layered schedules. | `gaussian_channel`, `find_cv_flow`, `corrected_pattern` | ~5 min |
| R65 | O6 (Figure 5) | `65_resource_vs_noise.py` | Resource reduction vs finite-squeezing noise; Pareto fronts; ancilla-elimination control. | exact channel semantics | <1 min |
| R66 | O7 (Figure 6) | `66_optimizer_scaling.py` | Candidates, verification cost and runtime vs program size; greedy vs beam. | — (provisional timings) | ~10 min |
| R67 | O8 | `67_end_to_end.py` | Circuit → compile → optimise → recompile → execute (2000 trajectories each). | trajectory ensembles vs each pattern's own channel, `find_cv_flow` | ~10 min |
| R68 | Figures 7–8 | `68_non_gaussian_and_gkp.py` | Cubic-phase rules vs a false-rule control; heralded subtraction; GKP readout and SUM syndrome. | dense `expm`, Poisson closed form, existing GKP readout, untruncated grid | ~10 min |

## Revision notes

Revisions made after a first run are disclosed in the scripts:

- **R66:** the flow check is limited to clusters of at most 80 nodes, because the exact
  rational fallback dominates the runtime above that size.
- **R68:**
  - the false-rule control is now a *q* displacement; a p displacement commutes with
    exp(i f(q));
  - the true-rule convergence criterion is ≤ 1e-7 with a ≥ 1e4 decrease;
  - the SUM syndrome is compared with an independent grid computation of the same protocol,
    not with the direct readout;
  - the readouts use cell-aligned outcome grids. An unaligned grid caused a 1.7e-3 artefact.
