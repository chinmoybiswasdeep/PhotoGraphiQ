# Decoding and leakage semantics

Nearest-cell decoding chooses m=floor(x/L+1/2), with comparisons performed in
the original coordinate to preserve floating-point half ties. It returns m mod 2
and residual x-mL in [-L/2,L/2). Negative cells follow the same rule. Residual is
analog distance to a lattice point, not a logical error probability.

`SoftDecisionDecoder` uses a specified preparation ensemble with strictly positive
priors. For Z the hypotheses are the actual normalized finite zero/one Fock states;
for X they are plus/minus. If f_b(x) is their homodyne density, the posterior is

    P(b | x) = prior_b f_b(x) / Σ_c prior_c f_c(x).

This normalization is Bayesian conditioning on an ensemble, not normalization of
nonorthogonal codeword overlaps with an unknown state. The finite superposition's
interference is retained in each f_b. Tests compare these posteriors against
independent continuous-comb likelihoods and synthetic mixture draws.
The model does not describe arbitrary entangled-state hypotheses without a new
joint likelihood model. Nearest-cell is not called optimal; Bayesian optimality
would itself be conditional on the supplied ensemble and decision cost.

Three different diagnostics must not be conflated:

| Quantity | Definition |
| --- | --- |
| Code-subspace leakage | 1-Tr(Pρ), P=BG^-1B† for the actual finite basis |
| Inconclusive probability | Tr(E_inc ρ) for the specified dual-basis POVM |
| Resource projection loss | 1 minus pre-normalization captured Fock mass |

The first vanishes for every normalized state prepared within span(B), even though
the basis states overlap and discrimination remains ambiguous. Stabilizer
expectations are additional diagnostics, not alternate names for leakage.
An analog record alone does not determine any of these state-level quantities.

Joint modular readout sums the full correlated density matrix against local
effects. It returns marginals by summing the resulting joint distribution.
Independent marginal errors cannot be assumed after CZ. Logical Pauli frames
track binary ideal byproducts modulo global phase and transform measurement labels;
they do not represent arbitrary analog displacement noise or repair finite envelopes.
