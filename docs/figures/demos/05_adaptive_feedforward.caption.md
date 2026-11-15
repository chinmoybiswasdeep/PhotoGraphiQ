**Adaptive feed-forward.** `protocols.adaptive(squeezing=1.5)` is a two-step wire whose
second homodyne angle is theta_2 = atan2(1, k + 0.1 m0), set by the first outcome m0
(k = 0.3). (a) One `native-gaussian` trajectory with its recorded outcomes. No correction
arrows are drawn: the second correction q -= m1 / sin(theta_2(m0)) is not affine in the
outcomes, and `pattern_corrections` rejects it instead of linearising it; the adaptive angle
is listed as a node annotation in the interactive page. (b) The angle actually
evaluated in 300 trajectories (from `execution_trace`, which re-evaluates the pattern's
expressions on the recorded outcomes) against the closed form (largest difference
0.0e+00 rad). Limitations: Gaussian feed-forward only; flow analysis is not applicable
because the measurement angle itself depends on an outcome.

Source: `scripts/generate_demo_figures.py` (`05_adaptive_feedforward`).
