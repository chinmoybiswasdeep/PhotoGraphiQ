**R45 - native Fock (pure-state) accuracy and cutoff convergence.** Total-photon truncation
n_1 + n_2 < cutoff, cutoffs 6..96, gates applied as P U P and renormalised (retained norm
recorded). (a) Distance (global phase aligned) between the engine's state and the exact state of
the SAME truncated model, P U psi / ||P U psi||, from 30-digit closed forms (coherent |alpha| = 1.30
and 4, squeezed r = 0.8), exact phases (Kerr kappa = 0.35), enlarged-space expm (50:50 beam splitter on
|1,1>, |2,1>, |3,3>, |4,2>), Hermite quadrature of the projected cubic phase (gamma = (0.1, 0.3)), and the exact
two-mode-squeezed-vacuum photon-count probability (r = 0.6): worst 1.7e-15 over all cutoffs.
(b) Convergence of untruncated observables (<n> = |alpha|^2, sinh^2 r; <a> under Kerr; cubic
<p> = gamma, <p^2> = 1 + 3 gamma^2; P(k) = (1 - l^2) l^(2k)); at cutoff 96 the projected cubic
(gamma = 0.3) <p^2> error is 9.2e-12. (c) Leaked norm 1 - ||P U psi||^2
(number-conserving gates leak nothing; the truncated-generator cubic phase is unitary on the truncated
space and reports no leakage by construction). (d) Same sequences on Piquasso 8.0.1: for |alpha| = 4 the
worst error of D(4)|cutoff/2> (interior columns of the displacement matrix) is 3.9e-06 (Piquasso) vs
7.2e-16 (native) over cutoffs <= 48; on D(4)|0> both reach rounding. HOM coincidence P(1,1) =
0.0e+00. Source: R45 (15_native_fock/45_native_fock_accuracy.py).
Limitations: one- and two-mode systems; float64.
