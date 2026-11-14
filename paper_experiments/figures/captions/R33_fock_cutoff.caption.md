**R33: Fock cutoff convergence.** Top row and bottom left: infidelity of each state to the state at the highest
tested cutoff for a cat state (alpha = 1.5), heralded photon subtraction from |2>, the cubic phase (gamma = 0.15) on a
coherent state and a finite-energy GKP resource, on `native-fock` (circles) and `piquasso-fock` (crosses); the
highest cutoff is a reference, not the exact state, so convergence is judged from the trend. The two engines agree at
every cutoff to fidelity >= 1.000000000000. Bottom middle/right:
demanding Gaussian workloads with an exact reference -- vacuum squeezed (r = 0.5) then displaced to alpha = 4
(<n> = 16.272, cutoffs 40-120) and a thermal
(0.2) displaced (alpha = 3) squeezed (r = 0.3) mixed input on the density-matrix engines (cutoffs 40-80). Relative error of
<n> at the highest cutoff: 2.2e-16 (native-fock) vs 6.8e-15 (piquasso-fock) and
7.6e-16 (native-mixed-fock) vs 0.0e+00 (piquasso-mixed-fock). Truncation
is never hidden: retained norms and boundary populations are in the result files. Model: hbar = 2, total-photon cutoff.
Source: `paper_experiments/10_convergence/33_fock_cutoff.py`.
