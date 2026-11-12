# 39. Physical shots and statistical error

## Goal

Sample reproducible physical trajectories.

## Theory

Averaged conditional probabilities and sampled bit frequencies are different estimators.

## Code

Run this standalone example after installing the pinned dependency and package.

```python
from photographiqml import GKPPhysicalConfig, PhysicalMuTA

config = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
model = PhysicalMuTA(1, physical_config=config)
result = model.run([1, 0], shots=3, seed=14)
assert len(result.sampled_output_bits) == 3
assert result.standard_errors is not None
print("mean conditional POVM:", result.decoded_joint_probabilities)
print("standard errors:", result.standard_errors)
print("sampled bit frequencies:", result.empirical_probabilities)
```

## Output

Captured from execution; upstream truncation warnings go to stderr and remain active.

```text
mean conditional POVM: {(0,): 0.7130826296347751, (1,): 0.2869173703652246}
standard errors: {(0,): 0.060778554260392, (1,): 0.06077855426039197}
sampled bit frequencies: {(0,): 0.3333333333333333, (1,): 0.6666666666666666}
```

## Validation

The assertions above execute during generation and in `scripts/ml/execute_tutorials.py`.
See [physical architecture](../physical/architecture.md) and the
[release report](../release-report.md) for independent logical and physical checks.

## Physical interpretation

Each trajectory samples homodyne outcomes; final bits use the joint output POVM.

## Limitations

Three shots illustrate the API, not a statistically precise estimate.
