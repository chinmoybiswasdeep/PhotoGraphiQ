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
