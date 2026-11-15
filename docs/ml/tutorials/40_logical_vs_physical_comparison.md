# 40. Logical vs physical comparison

## Goal

Compare decoded statistics without a false state overlap.

## Theory

Logical qubit and physical Fock vectors occupy different Hilbert spaces.

## Code

Run this standalone example after installing the pinned dependency and package.

```python
from photographiqml import GKPPhysicalConfig, PhysicalMuTA, compare_logical_physical

config = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
model = PhysicalMuTA(1, physical_config=config)
result = model.run(
    [1, 0], mode="physical-conditional", analog_outcomes=dict.fromkeys(model.measurement_order, 0.0)
)
comparison = compare_logical_physical(result)
assert comparison["finite_state_fidelity"] is None
assert comparison["total_variation_distance"] >= 0
print(comparison)
```

## Output

Captured from execution; upstream truncation warnings go to stderr and remain active.

```text
{'total_variation_distance': 0.17284892122480144, 'observable_differences': [-0.3456978424496029], 'comparison_mode': 'physical-conditional', 'finite_state_fidelity': None, 'marginal_code_subspace_leakage': [{(0, 4): 0.00021750138882559167}], 'convergence': {'certified': False, 'reason': 'A single resource setting is not a convergence study'}}
```

## Validation

The assertions above execute during generation and in `scripts/ml/execute_tutorials.py`.
See [physical architecture](../physical/architecture.md) and the
[release report](../release-report.md) for independent logical and physical checks.

## Physical interpretation

Total variation compares decoded probabilities to an ideal logical target.

## Limitations

The comparison remains conditional and does not certify a quantum channel.
