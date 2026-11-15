**R59 — PhotoGraphiQML restricted physical MuTA on `native-fock` vs `piquasso-fock`.**
Model: finite-energy GKP modes (hbar = 2; resource levels wide: peak width 0.9, envelope 0.9, 4 peaks, moderate: peak width 0.6, envelope 0.6, 6 peaks, narrow: peak width 0.45, envelope 0.45, 8 peaks),
total-photon Fock cutoff, unit CZ, p-homodyne nearest-cell decoding (spacing sqrt(2π)), classical
logical Pauli frame applied to the final joint Z/X modular readout; angles restricted to 0/π.
(a) Decoded-probability difference and final-state infidelity between the engines for
34 fixed-analog-outcome (postselected, conditional) cases: max |ΔP| =
8.99e-05 and max 1 − F = 2.49e-07 for the wide resource
(filled markers; tolerances 2e-04 and 2e-06, asserted only there). Open markers, moderate
resource: max |ΔP| by cutoff 32: 4.6e-04, 40: 3.3e-04, 48: 1.0e-04 (required to
decrease with cutoff; Piquasso's per-factor CZ truncation dominates). Decoded bits identical in every case.
(b) Distance to the largest-cutoff native result: 1 wire native-fock c=32: 6.7e-05; 1 wire native-fock c=40: 6.7e-06; 1 wire native-fock c=48: 1.6e-05; 1 wire native-fock c=56: 5.2e-06; 1 wire piquasso-fock c=32: 2.7e-04; 1 wire piquasso-fock c=40: 7.8e-05; 1 wire piquasso-fock c=48: 3.3e-05; 2 wire native-fock c=32: 1.1e-04; 2 wire native-fock c=40: 3.1e-05.
(c) TV distance to the logical target for all 16 lattice-point branches of a 1-wire MuTA,
with and without the decoded Pauli frame: wide (Δ = 0.9, cutoff 32): mean TV 0.191 corrected vs 0.277 uncorrected, worst corrected 0.362; moderate (Δ = 0.6, cutoff 40): mean TV 0.099 corrected vs 0.287 uncorrected, worst corrected 0.275; narrow (Δ = 0.45, cutoff 56): mean TV 0.038 corrected vs 0.328 uncorrected, worst corrected 0.110. Branches are postselected
conditional trajectories; the remaining corrected gap is the finite-GKP resource error, not a
numerical error. (d) physical-shots, 32 shots, seed 160:
decoded bits and sampled output bits identical on both engines (the sampled analog homodyne values
differ by up to 8.9e-03, never across a decoding cell), Rao-Blackwell probabilities differ by
5.0e-05; the logical target is shown for reference only.
(e) Runtime per run vs peak Hilbert dimension; provisional (shared machine), total script time
1229 s. (f) The logical-physical TV gap agrees between engines to
4.6e-04. Source: paper_experiments/20_photographiqml/59_photographiqml_physical.py.
Limitations: no loss, no detector inefficiency, no arbitrary-angle physical measurement, no
decoded-state fidelity and no fault-tolerance claim; finite-GKP codewords overlap (|⟨0|1⟩| =
wide 0.534, moderate 0.174, narrow 0.035).
