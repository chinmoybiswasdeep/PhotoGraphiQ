# 35. Zero-angle physical MuTA

## Goal

Run a complete finite-energy conditional chain.

## Theory

A zero-angle logical identity can still have finite-resource decoded errors.

## Code

Run this standalone example after installing the pinned dependency and package.

```python
from photographiqml import GKPPhysicalConfig, PhysicalMuTA

config = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
model = PhysicalMuTA(1, physical_config=config)
result = model.run(
    [1, 0], mode="physical-conditional", analog_outcomes=dict.fromkeys(model.measurement_order, 0.0)
)
assert result.representation == "gkp-physical"
assert not result.convergence["certified"]
print(result.decoded_joint_probabilities)
print(result.diagnostics["marginal_code_subspace_leakage"])
```

## Output

Captured from execution; upstream truncation warnings go to stderr and remain active.

```text
{(0,): 0.8271510787751981, (1,): 0.17284892122480164}
[{(0, 4): 0.00021750138882559167}]
```

## Validation

The assertions above execute during generation and in `scripts/ml/execute_tutorials.py`.
See [physical architecture](../physical/architecture.md) and the
[release report](../release-report.md) for independent logical and physical checks.

## Physical interpretation

All zero analog outcomes specify one physical conditional trajectory.

## Limitations

These statistics are not the unconditional identity-channel error.
