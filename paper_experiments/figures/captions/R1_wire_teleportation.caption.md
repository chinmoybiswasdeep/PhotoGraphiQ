**R1: one-step CV wire teleportation.** Left/centre: deviation of the sampled ensemble mean and
variance of the output q (left marker) and p (right marker) quadratures from the analytic
unconditional channel (`gaussian_channel`), in units of the shot-noise standard error, for the
one-step Fourier wire (`protocols.wire([0.0])`, resource squeezing r = 1.0) and 6
coherent inputs; 4000 homodyne trajectories with displacement feed-forward per input on the NumPy
`gaussian`, C++ `native-gaussian` and Piquasso-backed `piquasso` engines. The three engines consume the
same random numbers for equal seeds, so their markers coincide: they are one statistical sample, not
three independent ones. Every ensemble first and second moment lies within 2.69 standard errors
of the analytic value (acceptance <= 5). The corrected output channel equals the independent Fourier map
(q, p) -> (-p, q) to 6.1e-17 and its added noise equals diag(0, e^(-2r)) to
0.0e+00. `native-gaussian` reproduces the NumPy trajectories seed by seed (max record
difference 0.0e+00). Right: added p
variance versus r (q receives none). Model: hbar = 2, vacuum covariance I; the noise is the physical
cost of finite squeezing, not a truncation artifact, and vanishes only as r -> infinity. Source:
`paper_experiments/01_core_mbqc/01_wire_teleportation.py`.
