**GKP-encoded MBQC step (finite energy).** (a) `gkp.correction_pattern(quadrature="q")`: the
data mode `in` is coupled by CZ (between Fourier rotations, i.e. a SUM gate) to a
finite-energy logical-plus GKP ancilla that is measured in q; the modular feed-forward
q -= residual(m) is an opaque nearest-cell decoder (`decode_shift`), so it is an annotation,
not a drawn linear correction. (b) Wigner function of the finite-energy |0_L> (peak width =
envelope = 0.45, total-photon cutoff 60) after a known shift q = 0.25. (c) Decoded
residuals of 4 `native-fock` trajectories: 0.459, 0.060, 0.003, -1.043
(lattice spacing 2.5066); single shots scatter because the finite ancilla has its own
peak width. This is a finite-ancilla demonstration, not a fault-tolerance claim; GKP logical
flows are not analysed: `flows.analyze` on this cv-nongaussian graph returns `unsupported`.

Source: `scripts/generate_demo_figures.py` (`09_gkp_encoded_mbqc`).
