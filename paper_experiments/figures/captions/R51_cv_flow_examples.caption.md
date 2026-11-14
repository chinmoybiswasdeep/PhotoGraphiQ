**R51 - CV flow found on a weighted 2D cluster and a branched graph.** Columns: weighted
resource graph (squares inputs, diamonds outputs, circles homodyne-measured; edge labels are
CZ weights), measurement layers found by `photographiq.flows.find_cv_flow` (backward real-
linear layering, Booth-Markham 2023), and the displacement corrections q_t -= c_t m,
p_t -= (A c)_t m with m = record/sin(theta). grid_2x3: 2 layers, order
[(0, 0), (1, 0), (0, 1), (1, 1)], independent least-squares residual 1.5e-16;
branched_tree (1 input, 2 outputs: an ideal isometry): 3 layers, residual
0.0e+00. Resource squeezing r = 1.0 only enters the determinism
check; the flow itself is an infinite-squeezing statement. Source: R51 (51_cv_flow_finding.py).
