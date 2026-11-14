**R43 - native Gaussian backend vs analytic phase-space evolution.** (a) Relative covariance
error of `native-gaussian` against a dense first-principles reference (gate symplectics
S = expm(2 Omega M) from each quadratic generator; loss X = sqrt(eta) I, Y = (1-eta)(2 n_th+1) I)
for seeded random circuits of rotations, squeezers (|r| ~ 0.15 min(1, sqrt(40 n/D)), keeping the accumulated squeezing O(1)), beam splitters, CZ, shears,
displacements and thermal loss (eta in [0.6, 1], n_th <= 0.2) on n = 1..128
modes and depth D = 10..2560 (worst shown 3.4e-14; worst over the
unplotted unitary family, dominated by the expm reference itself at n = 1, 1.6e-12; worst physicality margin
min eig(V + i Omega)/||V|| = -1.3e-15). (b) Symplectic residual of the composed native
map S, extracted column-by-column from the native mean update (worst 2.0e-15). Dashed:
acceptance bound 64 D eps (heuristic forward-error model, not a proof). (c) Homodyne (ideal and
eta = 0.7, noise 0.15), and heterodyne conditioning on random mixed states (n = 2, 4, 8) against
a physical-dilation, precision-matrix reference: worst conditional-moment error 3.6e-16,
worst relative outcome-density error 1.6e-15 (exact zeros drawn at 1e-17). Not shown: closed-form coherent/squeezed/
thermal states (worst 0.0e+00),
and adaptive feed-forward: the two-step wire with outcome-dependent angle agrees with hand propagation
to 4.0e-15; native and NumPy trajectories with the same seed agree to 1.8e-15 in
outcomes and final moments. hbar = 2, float64, seed 20261143. Source: R43 (14_native_gaussian/43_native_gaussian_validation.py).
