# 41. Physical convergence

## Goal

Vary numerical grid resolution at fixed resources.

## Theory

Numerical convergence and changes in physical squeezing are distinct studies.

## Code

Run this standalone example after installing the pinned dependency and package.

```python
from photographiqml import GKPPhysicalConfig, PhysicalMuTA

config = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
model = PhysicalMuTA(1, physical_config=config)
study = model.physical_convergence(
    [1, 0],
    [513, 1025],
    axis="grid_points",
    mode="physical-conditional",
    analog_outcomes=dict.fromkeys(model.measurement_order, 0.0),
)
assert not study["certified"]
print(study["axis_type"])
print([row["max_probability_delta"] for row in study["rows"]])
```

## Output

Captured from execution; upstream truncation warnings go to stderr and remain active.

```text
numerical refinement
[None, 2.220446049250313e-16]
```

## Validation

The assertions above execute during generation and in `scripts/ml/execute_tutorials.py`.
See [physical architecture](../physical/architecture.md) and the
[release report](../release-report.md) for independent logical and physical checks.

## Physical interpretation

Only the integration grid changes in this whole-pattern calculation.

## Limitations

A small grid delta does not establish cutoff, peak-sum or finite-squeezing accuracy.
