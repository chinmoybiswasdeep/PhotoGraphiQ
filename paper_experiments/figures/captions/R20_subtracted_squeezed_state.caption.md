**R20: heralded photon subtraction on squeezed vacuum.** Left: photon-number distribution of
q-squeezed vacuum (r = 0.6) before and after one heralded subtraction (beamsplitter tap
theta = 0.15, herald count 1; herald probability 8.6171e-03); centre/right:
Wigner functions (hbar = 2) before and after. Parity changes from +1.0000 to
-1.0000; the mean photon number from 0.4053 to 2.1418; the subtracted
state has Wigner minimum -0.1592 (negative: non-Gaussian). Main engine
`native-fock` at total-photon cutoff 24 (minimum retained norm 0.999999937626);
`piquasso-fock` and `native-mixed-fock` reproduce herald probability, parity, mean photon number and the
distribution (total-variation distance <= 2.5e-16). Cutoff
study on `native-fock` (cutoffs [12, 16, 20, 28, 36]): fidelity between the two highest cutoffs
0.9999998094976486. The branch is postselected (conditional on count = 1) and
the tap is a physical, lossy subtraction, not the ideal annihilation operator. Source:
`paper_experiments/07_non_gaussian/20_subtracted_squeezed_state.py`.
