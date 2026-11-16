# Execution modes and results

## Ideal logical execution

`MuTA.run` retains exact qubit branch semantics. MentPy comparisons concern this
layer only. `GKPBridge.logical_target` explicitly requests the same ideal target.

## Physical conditional execution

```python
from photographiqml import PhysicalMuTA
model = PhysicalMuTA(1)
result = model.run([1, 0], mode="physical-conditional",
                  analog_outcomes=dict.fromkeys(model.measurement_order, 0.0))
```

This conditions on one complete set of real homodyne outcomes. Analog zero is a
quadrature value; it is not a requested all-zero logical branch. Such continuous
outcomes describe a conditional density, not an event with finite point probability.
Exactly one outcome is required for every measured node. `shots` must be one.

### Analog records for XY nodes

With `measurement_family="XY"` a measured node may run several homodynes, so its analog
outcome is a dictionary `{"ancilla": [q_1, ...], "data": p}`: one ancilla q value per
injection round, then the final data p value. A Clifford-angle node takes `{"data": p}`.
Use the dictionary form for every node: the request check accepts a scalar, but the XY
instrument itself rejects it with `ValueError`. The list must match
the rounds the protocol actually takes for those values (the decoded ancilla bits decide
whether another round follows); too few or too many raise `ValueError`.

```python
import numpy as np
from photographiqml import GKPPhysicalConfig, PhysicalMuTA
config = GKPPhysicalConfig(cutoff=56, backend="native-fock", xy_protocol="auto",
                           max_dimension=200_000)
model = PhysicalMuTA(1, 1, one_column=True, measurement_family="XY", physical_config=config)
names = list(model.parameters())
angles = dict(zip(names, (0.0, 0.0, np.pi / 4, 0.0)))
outcomes = {node: {"data": 0.0} for node in model.measurement_order}
outcomes[model.measurement_order[2]] = {"ancilla": [0.1], "data": 0.2}  # one round, bit 0
result = model.run([1, 0], angles, mode="physical-conditional", analog_outcomes=outcomes)
```

The node's record is a `pg.LogicalXYMeasurementResult` with `rounds`,
`heralded_success`, `applied_operations`, `joint_density` (likelihood density of the
complete analog record, not a probability) and `readout_probabilities`.

## Physical shots

`model.run([1,0], mode="physical-shots", shots=16, seed=2026)` samples 16 complete
homodyne trajectories with upstream independent child seed streams. A separate
child stream samples final bits from each trajectory's **joint** output POVM.
Identical settings/seeds reproduce the result. One-shot standard error is unknown.

`decoded_joint_probabilities` averages the conditional POVM probabilities over
trajectories (a Rao-Blackwell estimator). Its `standard_errors` are sample standard
deviations divided by sqrt(shots), for shots greater than one.
`empirical_probabilities` and `empirical_standard_errors` separately report sampled
bit frequencies and plug-in binomial errors. Small-sample plug-in errors can be
zero and are not confidence intervals. Neither estimator certifies resource accuracy.
`diagnostics["per_shot_joint_probabilities"]` keeps each trajectory's conditional joint
POVM row (label order of `decoded_joint_probabilities`); `physical_loss` uses these rows
as per-shot values.

For a single logical-XY instrument, `record.readout_probabilities` gives the exact decoded
X probabilities of the data state just before its final p homodyne. Averaging them over
trajectories is the same Rao-Blackwell idea at the level of one measurement: it removes
the final homodyne's sampling noise but keeps the spread of the ancilla records.

### Heralding

Every XY record reports `heralded_success`. It is `False` only when the
`max_injection_rounds` budget ran out before the remaining angle became Clifford; that
trajectory measured the wrong angle and is kept in the averages. Filter or count such
trajectories explicitly, for example
`[r.heralded_success for shot in result.measurement_records for r in shot.values()]`.
Decoded joint probabilities are unconditional unless you filter them yourself.

## Result contract

`PhysicalMuTAResult.physical_result` is the raw PhotoGraphiQ `Result` or
`ShotResult`, including the physical Fock state(s). `logical_target` is a separate
ideal result. No decoded qubit statevector or logical fidelity is invented.
Other fields retain corrected measurement records, output Pauli frames, decoded
joint/marginal distributions, sampled bits, resource configuration, lowering and
diagnostics. `representation` is always `gkp-physical`.

`compare_logical_physical(result)` returns decoded total variation distance,
per-wire observable differences, marginal code-subspace leakage and mode labels.
It leaves `finite_state_fidelity` unknown. Conditional branch comparisons cannot be
interpreted as unconditional physical gate error; shot averages also have sampling error.
