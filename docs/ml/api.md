# Input/output contract

All public APIs are experimental. Logical calculations use wire 0 as the most
significant tensor factor. Optical quantities follow PhotoGraphiQ's hbar=2.

| API | Input | Output |
|---|---|---|
| MuTA.run | normalized complex (2**n,), dict or active vector (P,) | LogicalResult with state (2**n,), probabilities (2**n,), density_matrix (2**n,2**n) |
| MuTA.run_batch | normalized complex rows (N,2**n), shared parameters | complex (N,2**n); empty batch supported |
| MuTA.parameters | none | all names mapped to PhotoGraphiQ Expr |
| MuTA.trainable_parameters | none | active names mapped to the same expressions |
| MuTA.initialize | seed, Gaussian scale | active parameter dictionary; does not mutate stored values |
| MuTA.freeze/unfreeze | one name or sequence, optional fixed scalar | model; fixes/removes constraints explicitly |
| MuTA.save/load | local JSON path | logical topology, values and frozen metadata round trip |
| Trainer.fit | deterministic callable R**P -> scalar, vector (P,) | final vector and History; initial point included |
| MuTAClassifier.fit | real (N,n), binary labels (N,) | fitted classifier with weights/history |
| predict_proba | real (N,n) | (N,2), labels [0,1] |
| MuTARegressor.predict | real (N,n) | real (N,) |
| MuTAKernel | real X1=(N,2), X2=(M,2) | fidelity matrix (N,M) |
| QuantumInstrumentModel.run | logical input and parameters | two branches: bit, probability, conditional state (2**(n-1),) or None for a negligible branch |
| GKPBridge.encode | normalized logical one-qubit vector (2,) | PhotoGraphiQ FockInput with cutoff amplitudes |
| GKPBridge.diagnostics | cutoff, width, envelope, peak/grid settings | captured weights, overlap, Gram eigenvalues; no gate validation claim |
| GKPBridge.logical_target | MuTA and logical input | ideal LogicalResult explicitly labeled logical |

Model parameter order is paper block, wire, local measurement column. Names
`alpha.wW.cC` use global column C. Dict bindings can be partial; vector bindings
must contain exactly the active values in `trainable_parameters()` order.
Freeze values cannot be overridden by a call. Stored values are exposed through
`state_dict`; current training wrappers retain learned weights separately.
Wrapper persistence, tied parameter groups and bounds are not implemented.

Classical encoders apply product Ry rotations directly to feature values in
radians; no standardization or clipping occurs. Regressors use an affine head
on first-wire Z, classifiers use a logistic head, and kernels use the paper's
two-feature measurement map. Quantum inputs bypass the classical encoder.

Raw Gaussian/Fock states can be passed directly to PhotoGraphiQ. They cannot
be passed as logical MuTA states. The bridge's finite GKP superposition is a
preparation prescription with nonorthogonal codewords, not an exact isometry.
No general decoded density-matrix API is claimed. Physical supervised wrappers
use decoded output probabilities and explicit shot settings.

## Physical v0.2 APIs

| API | Input | Output |
| --- | --- | --- |
| PhysicalMuTA.run | normalized product vector (2**n,) or factors (n,2), signed-X parameters, explicit mode/shots/seed | PhysicalMuTAResult; raw physical state(s), decoded joint/marginal probabilities, separate ideal target, frames and diagnostics |
| GKPBridge.run | fixed logical model or PhysicalMuTA; same physical options | Same capability-audited physical result; legacy resource models still reject execution |
| MuTA.physical_capabilities | bound parameters and optional GKPPhysicalConfig | Allocation-free angle audit and resource estimates |
| lower_muta_to_gkp | model, bound parameters, configuration | Inspectable Pattern and logical/physical/signal mappings; allocates codewords only after audit |
| PhysicalMuTA.initialize | seed | Categorical 0/pi parameter dictionary |
| DiscreteSearch.fit | PhysicalMuTA, scalar physical objective, optional categorical initial vector | DiscreteSearchResult with chosen parameters, loss and evaluations |
| PhysicalMuTA.physical_convergence | product input, increasing values, one axis and execution settings | Whole-pattern study; no automatic certification |
| GKPBridge.resource_readout | X/Z and nearest/soft | Public physical GKP readout; soft ensemble assumptions apply |
| compare_logical_physical | PhysicalMuTAResult | Decoded TV distance, observable differences and leakage; no invented state fidelity |

See [execution modes](physical/execution-modes.md) for conditional versus shot
semantics. `MuTAKernel` and `QuantumInstrumentModel` remain logical only.

Trainer accepts a derivative callable; default central differences apply to
deterministic scalar objectives only. `History` stores loss, optional validation
loss, gradient norm, parameter norm and cumulative seconds. The optimizer
initialization is supplied explicitly; dataset/split/measurement seeds belong
to the caller and are recorded in experiment configurations. Full-batch only.
Stochastic score-function gradients and automatic differentiation are pending.
