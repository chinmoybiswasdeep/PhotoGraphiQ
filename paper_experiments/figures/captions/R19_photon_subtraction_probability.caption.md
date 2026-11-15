**R19: heralded photon subtraction, analytic binomial law.** Probability of heralding k photons when
|n> (n = 1, 2, 3) passes a beamsplitter of angle theta whose other port
starts in vacuum (`non_gaussian.photon_subtraction`), for 15 angles in
[0.05, 1.52]. Lines: C(n, k) sin^(2k)(theta) cos^(2(n-k))(theta); circles: the
postselected branch probability reported by `native-fock` (total-photon cutoff 8). Over
135 branches the maximum absolute error is 4.4e-16
(native-fock), 4.4e-16 (piquasso-fock) and 2.2e-16
(native-mixed-fock); the conditional signal photon number equals n - k to
0.0e+00. Minimum retained Fock norm over all runs
0.999999999999999 (the input never leaves the truncated space).
Each value is a postselected (conditional) probability, not a deterministic operation. Source:
`paper_experiments/07_non_gaussian/19_photon_subtraction_probability.py`.
