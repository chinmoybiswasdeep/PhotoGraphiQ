# Measurements

`pattern.measure(node, measurement, key=...)` consumes the mode and writes a
classical result key, defaulting to its label. Keys must be unique and cannot be
used before their producer. Use `Outcome(key)` for later corrections.

| Description | Outcome | Backends |
|---|---|---|
| Homodyne(angle, efficiency=1, noise=0) | real quadrature | Gaussian; ideal pure Fock; mixed Fock with noise |
| Heterodyne() | (q,p) pair | Gaussian |
| Generaldyne(covariance) | (q,p) pair | Gaussian |
| PhotonNumber() | nonnegative integer | pure/mixed Fock |

Homodyne q and p constructors correspond to angles 0 and pi/2. General-dyne
covariance is a physical statistical covariance of the Gaussian measurement
seed. Vacuum heterodyne outcomes have covariance 2I in these units. Select pair
components explicitly, e.g. `Outcome("heterodyne", component=0)`.

Fock execution accepts `measurement_outcomes={key: value}` for postselection.
Counting reports probabilities; homodyne reports densities. Gaussian simulation
currently samples outcomes and does not report likelihood diagnostics. See the
[output guide](outputs.md) for the distinction between a reading and a branch.
