**R29: finite resource squeezing sets the MBQC noise floor.** Left: total added variance Tr(N) of the unconditional
channel of two compiled circuits -- R(0.5) followed by S(0.3) on one mode, and
CZ(0.6) on two modes -- versus resource squeezing r = 0.1..3.0
(20 values); right: output q and p variances of the single-mode circuit for coherent input
alpha = (0.3+0.2j). The logical map equals the independent NumPy target to
6.0e-16 at every r: squeezing changes only the added noise,
which falls from Tr N = 9 to 0.0272
(single mode). The CZ noise equals the closed form of four identity-wire steps per output mode to
3.8e-33. At r = [0.3, 1.0, 2.0] both circuits were executed on `gaussian` and
`native-gaussian` (600 trajectories): sampled ensembles reproduce map and noise within 2.33 shot-noise
standard errors. Model: hbar = 2, vacuum covariance I; the noise is physical (finite squeezing), not a truncation or
compilation error. Source: `paper_experiments/09_noise/29_finite_squeezing.py`.
