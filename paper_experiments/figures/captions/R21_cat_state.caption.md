**R21: finite-Fock cat states.** Wigner functions (hbar = 2) of the even and odd cat states
|alpha> +/- |-alpha> with alpha = 1.5, prepared by `CatResource` on `native-fock` at
total-photon cutoff 32; Wigner minima -0.0937 (even) and -0.1592
(odd), showing the interference fringes between the two coherent components. Across alpha in [0.5, 1.0, 1.5, 2.0] and
both parities (8 states) the fidelity with an independent NumPy coherent-state expansion is at
least 1.000000000000 on `native-fock` and
1.000000000000 on
the reference engines `piquasso-fock` and `native-mixed-fock`; parities have the expected sign, and the
engines agree on parity and mean photon number to
8.9e-16.
Maximum boundary population 1.6e-16: the cutoff is not limiting
for these amplitudes. Source: `paper_experiments/07_non_gaussian/21_cat_state.py`.
