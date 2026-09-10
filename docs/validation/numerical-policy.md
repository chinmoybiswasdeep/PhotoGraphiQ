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
