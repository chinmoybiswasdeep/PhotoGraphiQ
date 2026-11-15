**R27: GKP logical shifts, decoding and readout errors.** Nearest-cell decoding of raw q-homodyne outcomes over
[-3 sqrt(2 pi), 3 sqrt(2 pi)] (61 points): the decoded bit and residual equal an independent
floor-based formula at every point (dotted lines: cell boundaries (k + 1/2) sqrt(2 pi)). Finite-energy codewords
(peak width = envelope = 0.35, cutoff 48, `native-fock`): exact single-shot readout error
probabilities (integral of |psi(q)|^2 over wrong cells, ideal homodyne) are
3.52e-04, 3.55e-04 for |0>_L and |1>_L and
X|0>: 3.74e-04, Z|0>: 3.96e-04, X|1>: 3.66e-04, Z|1>: 4.04e-04 after the
logical displacements; `piquasso-fock` agrees to
3.6e-16.
These describe finite-energy codewords under ideal detection, not a fault-tolerant logical error rate. Source:
`paper_experiments/08_gkp/27_gkp_stabilizers.py`.
