# Numerical policy

Algebraic identities use tight float64 tolerances; Monte Carlo checks use fixed
seeds with tolerances appropriate to sample size. Hilbert-space truncation,
quadrature integration, finite physical resources and product-formula synthesis
have separate refinement controls. Never relax a tolerance merely to make CI pass.

Mixed-state checks include trace, Hermiticity, positivity, conditional normalization
and pure-limit agreement. GKP checks refine grid, peak count and Fock cutoff.
Differentiation checks compare analytic derivatives and refined finite differences;
the forward observable and its gradient both need cutoff studies. Synthesis checks
compare independent direct evolution and decreasing approximation error.

Retain failing regimes and document their error mechanism. A same-backend round
trip is useful for API integrity but insufficient evidence of physical correctness.

| Test class | Error budget and acceptance strategy |
| --- | --- |
| Exact finite algebra | Typically 1e-13 to 2e-12 absolute; preserve complete number sectors and phases |
| Finite-Fock approximation | Resolve reference and production cutoffs separately; retain failed coarse rows |
| Integration | Refine grid/window independently; analytic Gaussian integrals supply an additional oracle |
| Stochastic estimator | Fixed seeds, measured standard errors and a five-standard-error check; report variance |
| Product formula | Fit phase-aligned amplitude error over several slice counts above the numerical floor |

No tolerance is loosened to hide a scientific failure. A resolved infinite-space
reference and an exponential of a projected generator can differ at finite cutoff;
tests identify that distinction explicitly. Coverage is a software execution
metric, not a numerical-accuracy certificate.
