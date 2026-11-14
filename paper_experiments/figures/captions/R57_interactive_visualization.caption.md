**R57: one validated graph, four views.** (a) Complete weighted CV cluster graph of the
circuit R(0.4) on mode 0, S(0.3) on mode 1, then CZ(0.8), compiled by
`Circuit.compile(squeezing=1.5)`: 19 nodes,
18 CZ edges (weights printed on the edges), blue squares are the inputs,
green diamonds the outputs, thick outlines mark measured nodes. (b) Flow found by
`flows.analyze` (status `certified`, 9 measurement layers, `#k` gives the
measurement order); blue arrows are exactly the reported `correction_map`
(34 dependencies). (c) Classical corrections the compiled pattern
executes (17 outcome-dependent displacements, orange dashed,
coefficients of the raw homodyne record), read from its `Displace` commands. (d) One
trajectory simulated with `native-gaussian` (seed 57); `m=` values are the recorded
outcomes. The flow's correction map is the standard-form map (corrections after all CZs);
the compiled pattern applies each correction before the next CZ, so the two arrow sets
differ by 17 arrows; a
pattern executing exactly the flow map has the same channel as the compiled pattern (largest
difference 5.9e-13).
Validation: 17/19 checks pass (layout invariance over
12 layouts, SVG export parsed against the model, JSON round
trips, routed-curve clearance >= 0.34 layout units, minimum
0.367); not run: ['browser real-input checks', 'notebook iframe renders in browser']. Positions are display
metadata only. Limitations: browser checks use headless Edge only (no touch, Firefox or
WebKit); finite squeezing r = 1.5 adds noise that this figure does not show.
Source: `paper_experiments/18_visualization/57_interactive_visualization.py`.
