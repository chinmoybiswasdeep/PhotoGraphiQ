**R2: four-step identity wire versus finite squeezing.** Total added output variance Tr(N) of
`protocols.identity` (four k = 0 teleportation steps, coherent input alpha = (0.4+0.3j)) versus
resource squeezing r = 0.2..2.0. Circles: unconditional channel from
`gaussian_channel`; dashed: closed form Tr(N) = 4 e^(-2r), i.e. N = 2 e^(-2r) I (the independent `gaussian.wire_channel` recursion
agrees to 1.3e-33); squares/crosses: sampled
ensembles of 1500 trajectories on `gaussian` and `native-gaussian` (Tr V - 2, clipped at 1e-4 where shot
noise dominates; both engines draw the same random numbers for equal seeds, so their markers coincide).
The logical map is the identity to 1.7e-16 at every r; the added
noise is reported separately and is the physical cost of finite squeezing, not a compilation error. All
sampled moments lie within 2.28 shot-noise standard errors of the analytic channel (acceptance
<= 5); the native engine reproduces the NumPy trajectories seed by seed (max record difference
0.0e+00). Model: hbar = 2, vacuum covariance
I. Source: `paper_experiments/01_core_mbqc/02_identity_finite_squeezing.py`.
