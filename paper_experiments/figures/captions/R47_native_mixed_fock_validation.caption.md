**R47 - native mixed-state Fock validation.** (a) Density matrices of `native-mixed-fock` vs exact
density matrices of the same truncated model (total-photon cutoff c = 8..48): thermal inputs
(nbar = (0.3, 1.0)), a displaced squeezed thermal input (r = 0.4, nbar = 0.3, from dense expm), and the reduced state of
a two-mode squeezed vacuum (r = 0.6, partial trace); worst 4.4e-16 (acceptance 1e-12).
(b) Convergence of physical invariants with the cutoff: pure loss (eta = (0.3, 0.7)) on |alpha = 1.5> must give the
pure coherent state |sqrt(eta) alpha> (1 - fidelity, 1 - purity and <n> error at c = 48: 2.2e-16); thermal loss
(n_th = 0.4) moments; lossy (eta = 0.8) TMSV photon-count probabilities against the analytic thermal marginal;
homodyne conditioning (x = 0.7) of the lossy TMSV against Gaussian conditioning on the NumPy Gaussian backend
(conditional mean, covariance and outcome density; 3.1e-12 at c = 48). (c) Physicality after each of
21 operations (incl. Kerr, cubic phase, CZ, thermal loss) on 3 modes: worst Hermiticity/trace residual 1.3e-15,
smallest eigenvalue -8.8e-16. Without loss, the mixed engine on a pure input equals |psi><psi| of `native-fock`
to 5.0e-16. Source: R47 (15_native_fock/47_native_mixed_fock_validation.py).
Limitations: <= 3 modes; float64.
