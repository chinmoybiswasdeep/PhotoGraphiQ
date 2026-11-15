# PhotoGraphiQ capability audit

Source: sibling checkout of [PhotoGraphiQ](https://github.com/chinmoybiswasdeep/PhotoGraphiQ)
at `f15957d297f65f1e4761007e189f11119282dfdd`, version 0.3.0.

Inspected GKP, measurements, expressions, compiler, Gaussian channels and
simulation interfaces. Pattern commands support finite squeezed preparation,
weighted CZ, homodyne, displacement feed-forward and physical Fock resources.
`Parameter` returns a shared immutable Expr. Use it directly.
`simulate` returns a conditional trajectory; `run_shots` estimates ensembles.
Gaussian and Piquasso Fock execution remain PhotoGraphiQ responsibilities.
Covariance convention is [q,p]=2i with vacuum V=I, interleaved q,p ordering.

GKPResource provides finite-comb projection and captured weight; superposition
retains finite-state overlap. `decode_shift` is nearest-cell parity, not a
complete decoder of arbitrary logical density matrices. Logical displacements
move the finite envelope and are not exact finite-energy Paulis.

Missing for general physical MuTA: a validated arbitrary logical XY
measurement instrument, magic-state injection for the discrete non-Clifford
measurement, and a complete logical decoding/error-correction policy for
multi-mode outputs. Fock cutoff is a total-photon cutoff. One-mode resource
projection checks do not certify multi-mode gate convergence.

Consequently the GKP bridge can prepare resources, report overlaps/projection
weights, and expose ideal logical targets, but must refuse to advertise a
general finite-energy MuTA execution until these primitives are supplied.
No CV simulator or symbolic engine is duplicated in PhotoGraphiQML.
