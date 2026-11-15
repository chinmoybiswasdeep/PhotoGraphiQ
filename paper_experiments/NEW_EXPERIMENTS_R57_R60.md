# New manuscript experiments R57 and R60

R57 validates the interactive and static visualization. R60 is the end-to-end showcase and the
centrepiece figure. Run each script from its own directory with
`../../.venv/Scripts/python <script>`.

Each script:

- declares its acceptance constants before it evaluates anything;
- saves its results through `common.save_result`, which writes the CSV, the JSON and the
  `.metadata.json` provenance file;
- writes its figures through `common.save_figure` and `common.save_caption`, and every number in
  a caption is formatted from the results;
- on any failure, writes `<ID>_FAILURES.csv` and raises `AssertionError`.

All numbers below come from the runs recorded in `results/json/R57_*.json` and `R60_*.json`
(date 2026-11-13). These runs were made on a shared machine.

| ID | Script | Default runtime | Outcome |
|---|---|---|---|
| R57 | `18_visualization/57_interactive_visualization.py` | ~1 min without a browser; ~5 min with the Playwright/Edge checks | 20/20 checks passed, browser and notebook checks included |
| R60 | `21_showcase/60_end_to_end_showcase.py` | ~1 min (600 trajectories); `--full` (4000) ~3 min; `--quick` (100) ~25 s | 13/13 checks passed; timings PROVISIONAL |

---

## R57: Interactive visualization validation

**Objective.** Show that every view of one validated graph displays exactly the validated data.
The views are the complete graph, the flow, the corrections and the execution. Show also that
changing the visual layout never changes semantics, that exports and serialization are
faithful, and that the interactive page works under real input in a browser and in a notebook.

**Location.** `paper_experiments/18_visualization/57_interactive_visualization.py`. The browser
part reuses `scripts/check_interactive_browser.py`, which now has 12 real-input checks.

**Model.**

- One graph throughout: `Circuit(2).rotate(0, 0.4).squeeze(1, 0.3).cz(0, 1, 0.8)`, compiled at
  resource squeezing r = 1.5.
- The graph is built with `GraphView.from_pattern`: 19 nodes and 18 CZ edges, re-validated
  through `OpenGraph.from_dict`.
- The flow comes from `flows.analyze`: certified, 9 layers.
- One trajectory is simulated with `native-gaussian`, seed 57.

**Validation (all programmatic) and results.**

| Check | Criterion | Measured |
|---|---|---|
| flow status | certified | certified |
| drawn flow arrows vs `flow.correction_map` | equal | 34 = 34 |
| drawn correction arrows/labels vs `Displace` expressions, evaluated directly at unit records (independent of `pattern_corrections`) | equal pairs and labels | 17 = 17 |
| correction coefficients | ≤ 1e-12 | 0.0 |
| flow layers respect every dependency | all | 9 layers, all |
| flow-map pattern (`flows.cv.corrected_pattern`) channel vs compiled channel | ≤ 1e-9 | 5.9e-13 |
| semantics invariant under 12 layouts × 4 stages: 7 named layouts and 5 random manual layouts; graph JSON, payload without positions, bends and z-order, flow and correction arrows | all | 48/48 |
| pattern JSON and a seeded `native-gaussian` rerun unchanged | identical | identical, 17 outcomes |
| static SVG of each stage parsed with ElementTree (`node:`, `label:`, `edge:`, `flow:`, `correction:` ids) | node, edge, arrow and label sets equal to the model | 19 nodes, 18 edges, 0/34/17/17 arrows |
| routed curves keep clear of other nodes | ≥ 0.34 layout units | minimum 0.367 |
| editor JSON round trip (`apply_edits`) | graph equal; position drift ≤ 1e-12 | drift 0.0 |
| layout file and OpenGraph metadata round trip | exact | exact |
| browser suite `check_interactive_browser.py`, 12 checks | all pass | 12/12 |
| browser on the R57 graph: real drag, then `<select>` stage switch to flow, corrections and execution, layout switch to layered, Export JSON → `apply_edits` | arrows equal model; only the dragged node moved; graph unchanged; no page errors | all true |
| notebook: `nbclient` executes a notebook that displays the view | executes | executed |
| notebook output: the `<iframe srcdoc>` page it produced, opened in Edge | Cytoscape node and edge counts equal the model | 19 / 18, no errors |

The 12 browser checks are:

1. drag;
2. pan and zoom;
3. graph edits, with a zero weight rejected;
4. the construct, flow and corrections stages;
5. the execution animation;
6. the layout scale: the 6×6 cluster now renders at 10.9 px with no label overlaps;
7. static exports;
8. a stage switch with real `<select>` input and measurement-order stepping;
9. a layout switch plus **Reset view**;
10. routing and the browser exports:
    - the Cytoscape control points equal the Python routing to 1e-6 px;
    - every curve keeps ≥ 0.34 units from other nodes;
    - the downloaded SVG parses with the node, edge and arrow ids of the model;
    - the PNG is valid;
    - the downloaded HTML reopens with the dragged positions and the same stage.

**Outputs.**

- `results/{csv,json}/R57_interactive_visualization*`;
- `results/raw/R57_exports/{construct,flow,corrections,execution}.svg` and `layout.json`;
- `results/raw/R57_interactive.html`;
- `results/raw/R57_browser/`, with screenshots and `report.json`.

**Figure.** `R57_interactive_visualization`, four rows, one per view of the same graph:

- (a) the complete weighted graph;
- (b) the certified flow with `#k` orders;
- (c) the executed corrections;
- (d) one recorded run.

**Reproduction.** The browser and notebook parts need Playwright. Run with
`PYTHONPATH=<Edge Playwright venv>/Lib/site-packages` and with `PHOTOGRAPHIQ_CYTOSCAPE_JS`
pointing to a local Cytoscape.js 3.30.2. Without these, the script records `not_run` and never
`pass`. `--require-browser` turns `not_run` into a failure.

**Limitations.**

- The browser checks cover headless Edge (Chromium) only. Firefox, WebKit, touch input and
  high-DPI rendering are not tested.
- The notebook check executes the kernel and renders the produced iframe page in Edge. It does
  not drive JupyterLab itself.
- Routing avoids nodes but does not minimise edge crossings. Dense flow drawings (34–45 arrows)
  remain busy.
- The flow's standard-form correction map and the compiled feed-forward differ arrow by arrow
  (17 arrows here). Both are shown, and only the channel equality relates them.

**Manuscript placement.** Software section (visualization) or an appendix. Use the figure as a
compact "one graph, four views" panel.

---

## R60: End-to-end showcase

**Objective.** Follow one target operation through the whole stack, with every intermediate
object produced by the library and checked independently:

1. circuit;
2. MBQC compilation;
3. complete weighted graph;
4. flow and determinism;
5. measurement schedule;
6. correction map;
7. native simulation;
8. output-state analysis;
9. finite-squeezing noise;
10. interactive and static exports.

**Location.** `paper_experiments/21_showcase/60_end_to_end_showcase.py`.

**Model and parameters.**

- Target S = R₁(−0.6) · CZ(0.8) · [R₀(0.4) ⊕ S₁(0.3)].
- Conventions: hbar = 2; interleaved (q0, p0, q1, p1); vacuum covariance I; CZ(g):
  p_u += g q_v.
- `Circuit.compile(squeezing=1.5)` gives 23 modes, 22 CZ edges and 21 homodyne measurements.
- Inputs are coherent states α = 0.6+0.2i and −0.3+0.5i.
- `native-gaussian`, seeds 60…(60+K−1); K = 600 by default, 4000 with `--full`.
- Squeezing sweep r ∈ {0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3}.

**Validation.** The constants are declared before evaluation. Results from the `--full` run:

| Check | Criterion | Measured |
|---|---|---|
| flow status | certified | certified (13 layers, 45 dependencies) |
| ideal-limit determinism map (flow certificate) vs analytic S | ≤ 1e-9 | 1.2e-12 |
| flow order respects its own correction map | 0 violations | 0 |
| compiled command order respects the feed-forward it executes | 0 violations | 0 |
| channel of a pattern executing exactly the flow map vs compiled channel (S and N) | ≤ 1e-9 | 1.2e-12 |
| compiled S vs analytic product (NumPy, written in the script) | ≤ 1e-9 | 5.6e-16 |
| compiled S symplectic | ≤ 1e-9 | 1.1e-16 |
| conditional covariance identical for every outcome | ≤ 1e-10 | 0.0 |
| ensemble mean vs S μ, in standard errors | ≤ 5 | 1.43 |
| ensemble covariance vs S V Sᵀ + N, in standard errors | ≤ 5 | 1.87 (default run: 3.84 with K = 600) |
| `native-gaussian` vs NumPy `gaussian`, same seeds (outcomes, means, covariances) | ≤ 1e-10 | 1.8e-15 |
| N(r)·e^{2r} constant over the sweep | ≤ 1e-9 (relative) | 3.4e-16 |
| S independent of r | ≤ 1e-9 | 5.6e-16 |

**Results.**

- The intended map S is reproduced to machine precision at every r.
- The added noise is exactly N(r) = e^{−2r}·N(0): the fitted log-slope is −2.0000. At r = 1.5
  (13.0 dB), Tr N = 1.2252, and the noisy ensemble output has fidelity 0.7393 with the ideal
  output.
- The flow's standard-form correction map (45 entries) and the compiled feed-forward (21
  displacements) differ in form. The two compiled schedules order 2 flow pairs differently.
  Both give the same channel, to 1.2e-12.
- Each trajectory's conditional output mean is shifted by an outcome-dependent amount, as the
  finite-squeezing theory predicts. The spread of these means matches the predicted
  (S V Sᵀ + N − V_cond) ellipses.
- PROVISIONAL timings (shared machine): one `native-gaussian` run 24.6 ms, NumPy `gaussian`
  56.3 ms, flow analysis 2.5 s, 4000 native trajectories 75 s.

**Figure.** `R60_end_to_end_showcase` (A–F), every panel produced by the script:

- (A) the target circuit (`draw_circuit`);
- (B) the complete weighted cluster graph, with weights and squeezing;
- (C) the certified flow, with each node's layer Lk and all 45 dependencies;
- (D) the executed correction functions, with the output formulas
  `14: q += −1 m[13]` and `22: q += −1.047 m[21]` in the title;
- (E) the conditional output means of K runs, with the predicted 2σ ellipses, S μ and the
  sample mean;
- (F) Tr N(r) against e^{−2r}, 1 − F(ideal, noisy), and |S(r) − S_analytic|.

**Other outputs.**

- The interactive page with all four stages: `results/raw/R60_end_to_end_showcase.interactive.html`.
- The static SVG: `results/raw/R60_end_to_end_showcase_graph_corrections.svg`.
- The CSV and JSON contain S, N, d, the predicted and ensemble moments, and the sweep rows.

**Limitations.**

- Ideal homodyne detection and no loss.
- The flow certificate is an infinite-squeezing statement. At finite r the map is a noisy
  channel, and it is never presented as the ideal unitary.
- Timings are provisional. Rerun on an idle machine with `--full --repeats 21`.
- The statistical checks use 5σ thresholds on 10 covariance entries and 4 mean entries.

**Manuscript placement.** Main text, as the centrepiece figure for the paper's overview or
results section. Quote timings only from the idle-machine rerun.

---

## Demonstration figures (not numbered experiments)

`scripts/generate_demo_figures.py` writes 14 figures to `docs/figures/demos/`, each as
`.svg`, `.pdf`, `.png` and `.caption.md`. The 11 graph demos also come as `.interactive.html`.
The figures are:

1. teleportation;
2. a two-mode cluster, with simulated nullifiers vs e^{−2r};
3. a 3×4 cluster and its flow;
4. circuit-to-MBQC compilation, with gate-provenance boxes;
5. adaptive feed-forward: the angle used vs the closed form over 300 runs;
6. flow layering, qubit and CV;
7. correction maps, CV and qubit;
8. cubic-phase injection, with its Wigner function;
9. GKP syndrome extraction;
10. native vs NumPy execution;
11. boson sampling;
12. GBS;
13. PhotoGraphiQML training;
14. an end-to-end run.

Use `--quick` for fewer runs and shots. GKP (09) takes about 1 minute; the full set takes
about 6 minutes.

## Recommended `run_all_safe` classification

| ID | Classification |
|---|---|
| R57 | safe without a browser: it records `not_run` for the browser and notebook checks. Run it with `--require-browser` and the Playwright venv on the PYTHONPATH for the full validation. |
| R60 | safe with the default configuration (~1 min). `--full` is the publication configuration. Its timings are not deterministic. |
