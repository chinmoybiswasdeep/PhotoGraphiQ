**R26: finite-energy GKP resources.** Left: q-quadrature probability density of the finite-energy square-lattice
GKP |0>_L (peak width = envelope = 0.4): the target wavefunction of `GKPResource` versus its
projection onto Fock states below 48 photons, reconstructed from Hermite functions (L1 distance
2.60e-04).
Right: discarded weight 1 - captured mass of the projection (circles) and the q-translation stabilizer
|<exp(-i sqrt(2 pi) p)>| on `native-fock` (squares) versus cutoff [24, 32, 48, 64, 96]. Across widths 0.3/0.4/0.6
and both codewords, |<S_q>| ranges 0.2040 to
0.7532 and the captured mass is >=
0.999808; `native-fock` and `piquasso-fock` agree on stabilizers and photon
number to 8.9e-16. Finite-energy resources only: no
fault-tolerance, threshold or logical-error-rate claim follows from these numbers. Model: hbar = 2, lattice spacing
sqrt(2 pi). Source: `paper_experiments/08_gkp/26_gkp_resource.py`.
