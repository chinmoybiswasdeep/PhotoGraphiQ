**R5: random single-mode symplectic compilation.** Histogram of the Frobenius error between the
unconditional channel matrix S of the compiled MBQC pattern (`Circuit(1).gaussian(0, S).compile`,
resource squeezing r = 1.0) and the target S = R(phi1) S(r) R(phi2) for 400 random
targets (seed 20260913, phi uniform in (-pi, pi), logical r uniform in (-1.6, 1.6)); 0
decomposition failures. Maximum map error 7.8e-12 (acceptance
<= 1e-07); median log10 error -15.14. The added finite-squeezing noise is
reported separately (Tr N between 0.542 and 3.31e+03) and is not part of this error.
All 400 compiled patterns pass the graph-metadata checks (edges/weights equal to the
Entangle commands, node squeezing equal to the Prepare commands, each non-output node measured once,
feed-forward after the measurement it reads, `OpenGraph` round trip). The first 8 patterns
were executed on `gaussian` and `native-gaussian` (600 shots each): ensemble moments within
1.70 shot-noise standard errors of the analytic channel (acceptance <= 5); the
planner selects `native-gaussian`. Source:
`paper_experiments/02_compiler/05_random_symplectic_compile.py`.
