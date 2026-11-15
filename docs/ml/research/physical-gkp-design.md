# PhotoGraphiQ 0.3.1 downstream design

Audit target: `db07f9f9bf47da841bfa6b206562c5a3ffb121d3`, version 0.3.1.
Read both repositories' bridge, model, logical execution, parameter, training,
validation, tests, CI and research contracts, and upstream encoded/instrument,
GKP studies, pattern, simulation and pure/mixed Fock interfaces.

## Public contract

Use root exports GKPCode, PhysicalGKPReadout, LogicalDecodeResult,
LogicalPauliFrame, BaseGKPDecoder, NearestCellDecoder, SoftDecisionDecoder,
Pattern, Prepare, Measure, Signal, CallableExpression, Output, simulate,
run_shots, multimode_readout, measurement_convergence and Parameter.
GKPCode.logical_measurement performs authoritative absolute-1e-14 angle
checking without constructing codewords. Its logical_cz supplies unit CZ.
No backend state, private comb basis or decoder math is manipulated.

## Subset and migration

Keep logical MuTA unchanged. Historical `gkp` stays a refusing resource bridge,
with a deprecation warning recommending `gkp-resource` or explicit PhysicalMuTA.
`PhysicalMuTA` has representation `gkp-physical`, measurement_family `X`, and
categorical 0/pi parameters. GKPBridge can validate a fixed logical model without
rounding it. New physical schema 2 stores resource and decoder configuration;
schema 1 logical/resource models retain their old semantics. CV-native remains
a distinct proposal. Y, pi/4, generic XY, general entangled input preparation
and continuous physical-angle optimization remain unsupported.

## Lowering and frame convention

Construct the open-graph resource with finite GKP plus states. Prepare each
future neighbor just before its first required CZ, preserving all original
edges and causal measurement order while limiting the live frontier. Every
measured node is physically X (or signed X). A Signal interprets its decoded
bit using the Z component of the accumulated logical Pauli frame. Propagate
the interpreted bit through the original flow correction sets. These are
virtual corrections in the pre-entangled open-graph convention; do not also
propagate them through delayed CZs, which would double-count graph corrections.
Classical frames correct final X/Z labels, not finite-envelope deformation.
No analog displacement or finite-energy error-correction claim is made.

## Preflight and structured outputs

Before any codeword projection or backend creation, audit all bound angles;
validate code, decoder, execution mode, outcomes, product input, backend and
peak-live-mode Hilbert/matrix bounds. Lowering returns Pattern, code, mode/key
maps, frame dependencies, angle audit, config and resource accounting.

Physical-conditional requires a complete explicit analog branch. Its readout
is conditional, not an ensemble. Physical-shots calls PhotoGraphiQ run_shots;
each survivor state is read with the full joint modular POVM. Average these
conditional distributions and report Monte Carlo standard errors; additionally
sample final bits for empirical frequencies and binomial standard errors.
The result keeps physical Result/ShotResult, ideal LogicalResult, decoded joint
and marginal probabilities, original/adjusted records, frames, leakage and
backend norm/boundary diagnostics separate. No decoded qubit statevector.

SoftDecisionDecoder posterior refers to its declared one-mode preparation
ensemble; expose it for calibrated resource readout, but reject its use as
MuTA feed-forward without a validated conditional ensemble model.

## ML, validation and files

Add lowering.py, physical.py, physical_training.py; adapt gkp.py and root
Parameter imports; add explicit physical supervised evaluation/search paths.
Logical kernels remain Eq. 5 only. Use bounded coordinate search over 0/pi,
never Adam plus rounding. Classification uses decoded first-wire Z statistics.

Validate preallocation refusal with poisoned resource/backend calls, signed-X
and all logical branch frame equivalence, direct small PhotoGraphiQ patterns,
one/two-wire physical runs, output correlations, reproducible shots, and
independent cutoff/grid/peak sweeps. Width/envelope scans change physics.
Use upstream measurement_convergence only as a resource study, alongside
whole-model fixed-branch or shot studies. Preserve MentPy tests as logical-only.

Update dependency/CI pins, schema/version, docs/physical, tutorials 31–44,
physical demo/evidence scripts, visualization, release report and environment
metadata. Report current hosted runs separately from local unpushed checks.
