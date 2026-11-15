# 34. Restricted physical MuTA

## Goal

Inspect signed-X lowering and categorical parameters.

## Theory

Every bound angle is audited before finite-resource construction.

## Code

Run this standalone example after installing the pinned dependency and package.

```python
from photographiqml import GKPPhysicalConfig, PhysicalMuTA, lower_muta_to_gkp

config = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
model = PhysicalMuTA(1, physical_config=config)
angles = model.initialize(seed=4)
lowering = lower_muta_to_gkp(model, angles, config=config)
lowering.pattern.validate()
assert lowering.audit["supported"]
print(angles)
print({k: lowering.audit[k] for k in ("peak_live_modes", "hilbert_dimension", "cz_operations")})
```

## Output

Captured from execution; upstream truncation warnings go to stderr and remain active.

```text
{'alpha.w0.c0': np.float64(3.141592653589793), 'alpha.w0.c1': np.float64(3.141592653589793), 'alpha.w0.c2': np.float64(3.141592653589793), 'alpha.w0.c3': np.float64(3.141592653589793)}
{'peak_live_modes': 2, 'hilbert_dimension': 300, 'cz_operations': 4}
```

## Validation

The assertions above execute during generation and in `scripts/ml/execute_tutorials.py`.
See [physical architecture](../physical/architecture.md) and the
[release report](../release-report.md) for independent logical and physical checks.

## Physical interpretation

The lowering is an executable Pattern with explicit mode and signal mappings.

## Limitations

This categorical family does not implement arbitrary-angle MuTA.
