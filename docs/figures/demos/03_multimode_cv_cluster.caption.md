**Multimode CV cluster.** (a) `CVGraph.rectangular(3, 4, squeezing=1)` with the first
column as inputs and the last as outputs; all other nodes are measured with homodyne shear
k = 0 (p measurements). (b) `flows.analyze` on this open graph: status `certified`,
3 measurement layers [['(0, 0)', '(1, 0)', '(2, 0)'], ['(0, 1)', '(1, 1)', '(2, 1)'], ['(0, 2)', '(1, 2)', '(2, 2)']],
27 classical dependencies from the reported correction map (blue
arrows). Flow certification is an infinite-squeezing determinism statement; at finite
squeezing every measured node adds noise. Limitations: square-lattice nearest-neighbour
graph with unit weights only.

Source: `scripts/generate_demo_figures.py` (`03_multimode_cv_cluster`).
