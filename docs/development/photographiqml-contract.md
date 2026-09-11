# PhotoGraphiQML encoded contract (0.3.1)

PhotoGraphiQ provides physical encoded primitives; PhotoGraphiQML decides which
logical operations its model requests. No ML, MuTA, optimizer or MentPy dependency
is added to PhotoGraphiQ. The sibling PhotoGraphiQML checkout was audited, including
its bridge, logical executor, MuTA model, mapping, audit and release assessment.
Its `GKPBridge.run()` remains unsupported and was not modified.

## Stable interfaces

Root exports include `GKPCode`, `GKPResource`, `PhysicalGKPReadout`,
`IdealLogicalXYMeasurement`, `MeasurementInstrument`, `NearestCellDecoder`,
`SoftDecisionDecoder`, `BaseGKPDecoder`, `LogicalDecodeResult`,
`LogicalPauliFrame`, `LogicalMeasurementSynthesis`, `modular_effects`,
`multimode_readout`, `measurement_convergence`, and `Parameter`.
Existing measurement constructors and destructive Pattern semantics are preserved.
Version 0.3.1 is additive and fits `photographiq>=0.3.0,<0.4`.

`GKPCode(peak_width, envelope, cutoff, peaks, grid_points)` orchestrates existing
finite-comb projection. Width is the isolated probability peak standard deviation;
envelope is the inverse envelope width on centers. Cutoff is exclusive total photon
number in execution, not a tensor-product per-mode cutoff. Encoding normalizes a
linear combination of the actual nonorthogonal codewords. It is a preparation
recipe, not an isometric quantum channel. No orthogonalization changes resources.

## Supported physical subset

| Request | Lowering | Status |
| --- | --- | --- |
| Z | q homodyne, upper-tie nearest-cell parity | Implemented and independently tested |
| X | p homodyne, same modular decoder | Implemented and independently tested |
| XY at 0 modulo 2π | X | Implemented |
| XY at π modulo 2π | X with outcome labels reversed | Implemented |
| Y, other Clifford XY | No lowering | Unsupported |
| XY at π/4, arbitrary XY | No injection/synthesis | Unsupported |

`logical_measurement("XY", alpha=...)` accepts finite numerical angles. It uses
absolute angle tolerance 1e-14; a nearby trainable angle is not snapped with a
relative tolerance. Unsupported requests fail during construction, before Fock
allocation. Backend preflight checks measurement capabilities and instrument
dimensions before resetting or preparing simulator state.

`logical_gate(node, "X"/"Z"/"H"/"S")` returns the ideal-lattice displacement,
rotation or shear realization. `logical_cz(u,v)` returns unit-weight physical CZ.
These gates distort finite peaks/envelopes; their encoded target fidelity can be
well below one even after numerical convergence. They are not exact finite-code gates.

## Results and classical control

Physical readout returns `LogicalDecodeResult` in the ordinary `Result` records.
It contains bit, raw analog outcome, nearest cell, residual, decoder name,
optional posterior/confidence, optional subspace leakage and frame correction.
Nearest-cell confidence and probability fields are `None`, not invented certainty.
Leakage from a single analog sample is unknown and is `None`. Evaluate
`code.leakage(rho)` separately before destructive readout when the state is available.
`measurement_statistics` retains the raw homodyne **density**, not a bit probability.
`measurement_outcomes` specifies analog postselection, never a requested bit.

For feed-forward use `CallableExpression(lambda r: r["m"].bit, frozenset({"m"}))`
in `Signal` or an existing command expression. A frame is classical ideal logical
bookkeeping. It does not undo finite-envelope deformation as a physical displacement
would. Propagate frames with `hadamard`, `phase`, `cz`, `compose`, and `xy_angle`;
resolve X/Z labels by attaching the frame to a readout.

`multimode_readout` contracts the complete density matrix with tensor-product local
effects and returns joint and marginal probabilities. It does not multiply
marginals or assume independent errors. Its residuals and leakage are unknown;
analog trajectory records and separate state diagnostics supply those quantities.

## Instrument and extension semantics

`MeasurementInstrument` takes labels mapped to nonempty Kraus sequences. All
matrices share input/output dimensions, may be rectangular, and must sum to a
complete effect resolution. Include failure outcomes explicitly. `conditional`
retains the output system. Pattern `Measure` traces that output out, as all Pattern
measurements are destructive. A supplied finite instrument is a mathematical CP
model; PhotoGraphiQ does not claim optical synthesis for arbitrary matrices.

Mixed Fock supports general destructive instruments. Pure Fock supports rank-one
effects, which guarantee pure survivors; all other effects fail preflight.
`MeasurementProtocol` allows backend-specific extensions without changing Pattern.
JSON serialization and JAX execution of these new measurement descriptions are
not implemented and reject them rather than mislabeling them as homodyne.

`LogicalMeasurementSynthesis.lower` is an explicit unsupported extension point.
It does not install an unvalidated magic-state protocol. Full physical MuTA still
needs downstream adaptive lowering, a multi-mode output/error-correction policy,
and circuit-level convergence. Pauli measurements alone do not certify Pauli MuTA.

## Repeated execution

Resource codewords are cached per `GKPCode`. Existing `run_shots` remains the
trajectory batch interface. No compiled executor is added: preflight and backend
reset/preparation remain necessary, and no measured compilation bottleneck justifies
changing their lifecycle. `autodiff.parameter_batch_expectation` binds ordered
parameter rows with JAX `vmap` for existing fixed-outcome supported operations only.
It does not differentiate sampled bits, decoders or encoded instruments.
