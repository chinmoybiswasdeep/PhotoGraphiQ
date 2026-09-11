# Encoded GKP validation

Run `python -m pytest tests/gkp -q` and
`python -m experiments.gkp_logical_evidence`. The latter writes
`validation/gkp-logical/evidence.json`, including finite target fidelity, subspace
leakage, stabilizers, joint probabilities and independent-reference error.

## Independent references

Readout tests normalize the continuous finite comb using closed Gaussian integrals.
For p readout they Fourier-transform each peak analytically using hbar=2. Cell
probabilities are integrated independently from the production Fock POVM. Sweeps
use widths/envelopes (0.4,0.4), (0.55,0.5), (0.7,0.35), cutoffs 24,48,112,
both computational codewords and both X eigenstate preparations. The final
probability error must be below 3e-7. This is a numerical agreement tolerance,
not a bound on physical logical error. Broad peaks still misclassify.

Grid 2049/4097 and peak count 4/6 are compared separately. Residual moments are
integrated per cell. Width/envelope studies change the physical model and are
reported as resource sweeps rather than proofs of numerical convergence.

H and S are compared to independent Hermite/quadrature constructions on zero,
one, plus and minus. S comparisons remove global phase only. Existing independent
X/Z displacement and stabilizer tests remain in place. Two-mode CZ tests use
the five requested input preparations and compare against direct multiplication
by exp(iq₁q₂/2) on an independent quadrature grid. Total cutoffs 24,40,64 must
reduce reference infidelity below 1e-7. Finite decomposition error at low cutoff
is retained and measured, not assumed zero.

An encoded X measurement after CZ is checked against an independent quadrature
bra contraction for both its density and surviving state. Instrument tests cover
complex rank-one outcomes, multi-Kraus retained updates, entangled destructive
updates, completeness, positivity, impossible outcomes, and early pure-backend
rejection of potentially mixed survivors. Soft decoding is compared with independent
likelihoods and synthetic ensemble draws. Ideal XY tests cover six qubit states.

## Scope of evidence

These are finite primitive and small two-mode validations. They do not establish
fault tolerance, a universal cutoff, physical Y, magic-state injection, arbitrary
XY, or an executable finite-energy MuTA model. The downstream bridge remains
blocked. See the [hardening report](../development/gkp-logical-interface-hardening.md)
for measured release-gate results and remaining work.
