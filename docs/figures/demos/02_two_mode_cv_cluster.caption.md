**Two-mode CV cluster state.** (a) `CVGraph.line(2, squeezing=1.0)`: two p-squeezed modes
joined by CZ(1) (both are outputs; nothing is measured). (b) Covariance matrix of the state
prepared by `simulate(Pattern(graph), backend="gaussian")` (hbar = 2, vacuum = I). (c) The
nullifier variances Var(p0 - q1) and Var(p1 - q0) of the simulated states for r = 0...2
against the closed form e^(-2r) (largest deviation 2.7e-15); the state is an ideal cluster
only as r -> infinity. Limitations: pure Gaussian resource, no loss.

Source: `scripts/generate_demo_figures.py` (`02_two_mode_cv_cluster`).
