**R34: cubic-phase injection over cutoff and resource squeezing.** Fidelity of the m = 0 postselected output of
`non_gaussian.cubic_injection` (gamma = 0.15, coherent input alpha = (0.2+0j)) to the ideal CubicPhase(gamma) built
independently with NumPy/SciPy, on a grid of total-photon cutoffs [24, 32, 48, 64, 96] and resource squeezing [0.1, 0.3, 0.6, 1.0, 1.5, 2.0];
left `native-fock`, right `piquasso-fock`. Hatched cells stopped at the retained-norm guard (15 of
30 native, 16 Piquasso): the finite-squeezing cubic-phase
ancilla spreads to high photon numbers, so larger r needs a larger cutoff. Successful native cells range from
0.9469 to 0.9974; along each row the fidelity is
stable in the cutoff (converged), and it rises with r (the conditional filter approaches the ideal gate only as
r -> infinity). Where both engines succeed their outputs agree to fidelity >= 0.999992. Postselected branch,
finite energy: not a deterministic cubic gate. Source: `paper_experiments/10_convergence/34_cubic_convergence.py`.
