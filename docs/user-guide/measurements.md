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
| MeasurementInstrument({label: [Kraus, ...]}) | outcome label | rank-one effects: pure/mixed Fock; general Kraus/POVM: mixed Fock only |
| PhysicalGKPReadout(basis="Z" or "X", decoder, frame, flip) | `LogicalDecodeResult` (decoded bit plus raw analog value) | pure/mixed Fock (`encoded_gkp_xz`) |

"Gaussian" means `gaussian`, `piquasso` and `native-gaussian`; "pure Fock" means
`piquasso-fock` and `native-fock`; "mixed Fock" means `piquasso-mixed-fock` and
`native-mixed-fock`. A backend that lacks a measurement's capability rejects the
pattern before any state is allocated.

Homodyne q and p constructors correspond to angles 0 and pi/2. General-dyne
covariance is a physical statistical covariance of the Gaussian measurement
seed. Vacuum heterodyne outcomes have covariance 2I in these units. Select pair
components explicitly, e.g. `Outcome("heterodyne", component=0)`.

`MeasurementInstrument` is a complete finite-outcome instrument: its effects must
sum to the identity (include failure outcomes), and its input dimension must
equal the backend cutoff. Instruments whose effects are all rank one keep pure
states pure and run on pure-Fock backends (`rank_one_instrument`); any other
Kraus set needs `custom_kraus_instrument`, which only the mixed-Fock backends
provide. Inside a pattern, `Measure` discards the measured mode and keeps the
conditional state of the other modes.

`PhysicalGKPReadout` is an ideal analog homodyne (angle 0 for Z, pi/2 for X)
followed by modular decoding and a Pauli-frame correction. Only X and Z bases are
implemented; Y and general XY raise `NotImplementedError`. Postselection always
fixes the raw analog value, never the decoded bit, and code-subspace leakage is
reported as `None` because one analog outcome cannot determine it.

Every backend that executes patterns accepts `measurement_outcomes={key: value}`
for postselection, Gaussian backends included. Each measurement records a diagnostic in
`result.measurement_statistics[key]`: photon counting and instruments report a
probability, homodyne, heterodyne and general-dyne report a probability density,
and the record says whether the outcome was postselected or sampled. On Gaussian
backends the density is the exact Gaussian marginal of the conditioned outcome
(tests/protocols/test_gaussian_postselection.py checks it against closed-form
conditioning and converged Fock postselection). `result.log_likelihood` sums the
logarithms of these values. See the [output guide](outputs.md) for the
distinction between a reading and a branch.
