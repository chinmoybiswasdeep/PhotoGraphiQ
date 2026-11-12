# Interactive visualization

`photographiq.interactive` draws a measurement-based computation in stages:

1. the resource graph;
2. its flow;
3. the corrections that flow implies;
4. one recorded execution.

It also provides plots of the states the computation produces. Python is the
authoritative model. Every number on the page is computed by PhotoGraphiQ; the
browser only displays it and lets you edit the graph. Browser edits come back to
Python as JSON and are validated again before you use them.

For the Matplotlib drawing functions of `photographiq.visualization` (`draw_graph`,
`pattern.draw()`, compilation figures), see [Visualization and export](visualization.md).

## Requirements

| Output | Needs |
|---|---|
| HTML graph pages | a web browser; Cytoscape.js 3.30.2 is loaded from `cdn.jsdelivr.net`, or inlined from a local file with `cytoscape_js=path` |
| HTML plot pages | a web browser; Plotly.js 2.35.2 is loaded from `cdn.jsdelivr.net`, or inlined with `plotly_js=path`. The Python `plotly` package is **not** used. |
| Notebook display | Jupyter; views and plots render as `srcdoc` iframes through `_repr_html_` |
| SVG / PDF / PNG | `matplotlib`, imported only when you save or draw |

Nothing in the package needs network access. Only a page that uses the CDN URLs needs it
when it is opened. For an offline page, download the two JavaScript files once and pass
their paths.

## The four stages

Each `GraphView` is immutable. `construct`, `with_flow`, `with_corrections` and
`with_execution` return new views, and `photographiq.interactive.STAGES` lists the
stage names.

```python
import photographiq as pg
from photographiq.cvflow import flow_pattern
from photographiq.flows import analyze
from photographiq.interactive import GraphView, execution_trace
from photographiq.ir import OpenGraph

graph = pg.CVGraph.line(4, squeezing=1.5, inputs=(0,), outputs=(3,))
shears = {0: 0.3, 1: -0.5, 2: 0.0}
open_graph = OpenGraph.from_cvgraph(graph, shears)

# A: graph, CZ weights, input/output/measured roles, squeezing, measurement metadata
view = GraphView(open_graph, title="Four-node CV wire").construct()

# B: flow overlay, read from a FlowAnalysisResult (or any object/dict with its fields)
flow = analyze(open_graph)
print(flow.status, flow.measurement_order)
flow_view = view.with_flow(flow)
print(sorted(flow_view.flow_arrows()))

# C: corrections as linear combinations of raw homodyne records m[k]
correction_view = flow_view.with_corrections()
for target, correction in correction_view.corrections.items():
    print(target, correction.formula())

# D: one recorded trajectory
pattern = flow_pattern(graph, flow.measurement_order, shears=shears)
result = pg.simulate(pattern, backend="gaussian", seed=7)
trace = execution_trace(pattern, result)
execution_view = correction_view.with_execution(trace)
print(len(trace.steps), trace.steps[0].label)
```

Output:

```text
certified (0, 1, 2)
[(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
1 q += -1.044 m[0]
2 q += -1.118 m[1]; p += -1.044 m[0]
3 q += -1 m[2]; p += -1.118 m[1]
15 Prepare 1 (squeezing=1.5)
```

**Flow (B).** The flow overlay shows what the flow analysis reported; it derives
nothing itself. Its inputs are the `correction_map`, the measurement order and
layers, the certification status, the algorithm and the stated assumptions. A
result for a different model, or one that names nodes absent from the graph, raises
`ValueError`.

**Corrections (C).** `with_corrections()` with no argument uses the flow's
correction map. You can instead pass a correction map, another flow result or a
`Pattern`. A `Pattern` shows the outcome-dependent `Displace` commands it actually
executes, extracted with `pattern_corrections`; feed-forward that is not affine in
the outcomes raises `ValueError`. CV coefficients refer to the raw record `m[k]`,
the same quantity the simulator records.

**Execution (D).** The execution stage shows one recorded trajectory. Its outcomes
and evaluated feed-forward arguments come from `Result.records`. On the HTML page,
the **Play**, **Pause**, **Step** and **Reset** buttons animate the steps.

You can also build a view directly from a pattern with
`GraphView.from_pattern(pattern)`. Anything that cannot be drawn faithfully is
listed in `view.notes` or as a node annotation rather than being approximated:
repeated or cancelling CZs, beam splitters, non-wire-step or adaptive homodyne
angles, and non-homodyne measurements.

## Postselection is labelled, never hidden

If you fix outcomes with `measurement_outcomes`, the trace marks those
measurements as conditional branches. The page notes include the same warning.

```python
import photographiq as pg
from photographiq.cvflow import flow_pattern
from photographiq.interactive import GraphView, execution_trace

graph = pg.CVGraph.line(4, squeezing=1.5, inputs=(0,), outputs=(3,))
pattern = flow_pattern(graph, (0, 1, 2), shears={0: 0.3, 1: -0.5, 2: 0.0})
post = pg.simulate(pattern, backend="gaussian", seed=7, measurement_outcomes={0: 0.0})
trace = execution_trace(pattern, post)
print(trace.postselected)
view = GraphView.from_pattern(pattern).with_corrections(pattern).with_execution(trace)
print(view.payload()["notes"][-1])
```

```text
(0,)
Postselected outcomes [0]: this is one conditional branch, not a deterministic run
```

## Open, save and embed a page

| Call | Effect |
|---|---|
| `view` (last expression in a notebook cell) | inline iframe |
| `view.show(stage=None, **options)` | writes a temporary `.html` file, opens the default browser, returns the path; `options` go to `to_html` |
| `view.save_html(path, stage=None, **options)` | writes a self-contained page; `options` go to `to_html` |
| `view.to_html(stage=None, *, editable=True, cytoscape_js=None)` | returns the page as a string |
| `view.payload(stage)` / `view.elements(stage)` | the JSON data the page renders (Cytoscape.js elements) |

`stage` selects any stage whose overlay exists. Asking for `"flow"` on a view without
a flow overlay raises `ValueError`. The page embeds its data as escaped JSON, so node
labels and titles cannot inject HTML or script.

## Layouts

Positions are display metadata and never change the graph. The available layouts
are `grid`, `circular`, `hierarchical` (breadth-first depth from the inputs),
`layered` (by flow layers or measurement order), `force` (NetworkX spring layout,
fixed `seed`; the function is `force_directed`, but `compute_layout` takes the name
`"force"`), `publication` (the default: an aligned integer grid that reduces but
does not minimise edge crossings) and `manual`. All of them are deterministic. An
unknown method name, a missing node or a non-finite position raises `ValueError`.

```python
import photographiq as pg
from photographiq.interactive import GraphView, compute_layout, export_layout, import_layout
from photographiq.ir import OpenGraph

open_graph = OpenGraph.from_cvgraph(pg.CVGraph.line(4, squeezing=1.5, inputs=(0,), outputs=(3,)), {})
print(compute_layout(open_graph, "grid").positions[1])
view = GraphView(open_graph).with_layout("circular")
text = export_layout(view.layout)                 # {"0": [x, y], ...}; also export_layout(layout, path)
print(import_layout(text, open_graph) == view.layout)
```

```text
(1.0, 0.0)
True
```

## Edit in the browser, validate in Python

With `editable=True` (the default), the page side panel can do the following:

- add or remove nodes;
- add edges and set their weights;
- toggle inputs and outputs;
- set squeezing, homodyne shear or qubit measurements.

**Export JSON** downloads `{"graph": ..., "layout": ...}`. `apply_edits` rebuilds the
graph with `OpenGraph.from_dict`, so an edit that would be invalid in Python is also
rejected here.

```python
import json
import photographiq as pg
from photographiq.interactive import GraphView, apply_edits
from photographiq.ir import OpenGraph

view = GraphView(OpenGraph.from_cvgraph(pg.CVGraph.line(4, squeezing=1.5, inputs=(0,), outputs=(3,)), {}))
exported = {"graph": view.graph.to_dict(), "layout": view.layout.to_dict()}  # what the page exports
graph, layout = apply_edits(json.dumps(exported))   # JSON text, a file path or a dict
print(graph.to_dict() == view.graph.to_dict())
exported["graph"]["edges"][0]["weight"] = 0
try:
    apply_edits(exported)
except ValueError as exc:
    print(exc)
```

```text
True
Edge (0, 1) needs a finite nonzero weight
```

## Static figures for manuscripts

`view.save(path, stage=None)` and `draw_static(view, path=None, stage=None, *, ax=None,
figsize=None, dpi=300, formulas=True, title=None)` render any stage with Matplotlib.
The suffix of `path` selects SVG, PDF or PNG. SVG keeps text as text and PDF embeds
Type 42 fonts. The styling matches the page:

- CZ edges: solid grey, with weight labels;
- flow dependencies: dashed blue;
- corrections: dotted orange, with coefficients;
- node shapes: square inputs, diamond outputs, circular measured nodes.

```python
import pathlib, tempfile
import photographiq as pg
from photographiq.cvflow import flow_pattern
from photographiq.interactive import GraphView, execution_trace

graph = pg.CVGraph.line(4, squeezing=1.5, inputs=(0,), outputs=(3,))
pattern = flow_pattern(graph, (0, 1, 2), shears={0: 0.3, 1: -0.5, 2: 0.0})
result = pg.simulate(pattern, backend="gaussian", seed=7)
view = GraphView.from_pattern(pattern).with_corrections(pattern).with_execution(execution_trace(pattern, result))
out = pathlib.Path(tempfile.mkdtemp())
print(sorted(view.save(out / f"wire_{s}.svg", stage=s).name for s in ("construct", "corrections", "execution")))
```

```text
['wire_construct.svg', 'wire_corrections.svg', 'wire_execution.svg']
```

## Scientific plots

Each plot function returns a `Plot`. Call `.show()`, `.save_html(path)`,
`.to_html(plotly_js=None)`, `.save(path)` (SVG/PDF/PNG with Matplotlib) or
`.draw(ax)`; in a notebook the `Plot` displays inline. No physics is evaluated in
JavaScript. Conventions are hbar=2, q = a + a^dag, vacuum covariance I, and homodyne at
angle theta measuring q cos(theta) + p sin(theta).

| Function | Shows | States |
|---|---|---|
| `photon_number_distribution(state, node)` | P(n) bars; the title gives the retained truncated mass | Fock (`native-fock`, `piquasso-fock`, mixed Fock) |
| `quadrature_distribution(state, node, angle, *, samples=None, x=None, bins=40)` | ideal homodyne density, optionally with a histogram of recorded outcomes | Gaussian or Fock |
| `wigner_plot(state, node=None, q=None, p=None, *, points=121, extent=5.0)` | signed Wigner heatmap; the title gives grid mass and negative volume | Gaussian (closed form) or Fock |
| `covariance_heatmap(state)` | covariance matrix, interleaved (q0, p0, q1, p1, ...) | Gaussian |
| `squeezing_ellipses(state, nodes=None)` | one-sigma ellipses with dashed vacuum circles | Gaussian |
| `entanglement_diagnostics(state)` | log-negativity of (mode, rest) in bits and reduced entropy in nats, per mode | Gaussian |
| `measurement_probabilities(source)` | recorded probabilities and densities; postselected keys labelled | any `Result`, or a mapping outcome -> probability |
| `fidelity_vs_cutoff(study, reference=None)` | infidelity versus Fock cutoff of a `CutoffStudy` | Fock |
| `optimization_curves(history, *, log=False)` | losses, validation losses and gradient norms of a `photographiqml` `History` | n/a |
| `runtime_scaling(sizes, timings)` | log-log plot of timings you supply (it measures nothing) | n/a |

```python
import math
import photographiq as pg
from photographiq.interactive import photon_number_distribution, photon_number_marginal, quadrature_distribution

coherent = pg.Pattern(inputs=(0,)).append(pg.Displace(0, q=2.0, p=0.0))       # alpha = (q + i p)/2 = 1
state = pg.simulate(coherent, backend="native-fock", cutoff=15).state
pn = photon_number_marginal(state, 0)
print([round(pn[n], 6) for n in range(3)], [round(math.exp(-1) / math.factorial(n), 6) for n in range(3)])
print(photon_number_distribution(state, 0).layout["title"]["text"])

shots = pg.run_shots(pg.Pattern(inputs=(0,)).append(pg.Displace(0, q=2.0))
                     .append(pg.Measure(0, pg.Homodyne(0.0), key="x")), 200, backend="gaussian", seed=1)
gaussian = pg.simulate(coherent, backend="gaussian").state
plot = quadrature_distribution(gaussian, 0, 0.0, samples=shots.values("x"))
print([trace["name"] for trace in plot.traces])
```

```text
[0.367879, 0.367879, 0.18394] [0.367879, 0.367879, 0.18394]
Photon-number distribution, mode 0 (retained mass 1.000000)
['200 recorded outcomes', 'ideal density']
```

The ideal homodyne density does not include detector inefficiency. A histogram of
recorded `samples` includes whatever detector noise the measurement had.

## Limits

- Overlays display results computed elsewhere: flows come from `photographiq.flows` or
  `photographiq.cvflow`, executions from `simulate`. The interactive module does not
  check a flow it was not given.
- The page itself runs no simulation. Edits in the browser change only the exported
  graph, and you must re-run the analysis in Python.
- Layout quality (edge crossings) and rendered appearance are not checked by tests.
  Tests check determinism, file validity, the displayed data, and the generated
  JavaScript, which is parsed with Node.js and run in headless Edge or Chrome when one is
  installed.

## Further examples

- `examples/notebooks/interactive_graph_visualization.ipynb` goes through all four
  stages, static export, the edit round trip and three plots. It is executed by
  `tests/visualization/test_interactive_notebook.py`.
- The tests in `tests/visualization/` compare displayed data with closed forms, the
  simulator and certified flows.
