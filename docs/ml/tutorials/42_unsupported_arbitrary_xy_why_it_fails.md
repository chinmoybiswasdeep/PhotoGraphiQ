# 42. Unsupported arbitrary XY why it fails

## Goal

Exercise the preserved safety boundary.

## Theory

Rotating a homodyne angle does not synthesize arbitrary logical XY measurement.

## Code

Run this standalone example after installing the pinned dependency and package.

```python
from photographiqml import GKPPhysicalConfig, PhysicalMuTA

config = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
model = PhysicalMuTA(1, physical_config=config)
try:
    model.run([1, 0], {"alpha.w0.c1": 0.37})
except NotImplementedError as error:
    print(error)
else:
    raise AssertionError("Unsupported physical measurement executed")
```

## Output

Captured from execution; upstream truncation warnings go to stderr and remain active.

```text
Physical GKP MuTA cannot lower alpha.w0.c1=0.37 rad: PhotoGraphiQ 0.3.1 supports physical XY only at 0 and pi modulo 2pi (absolute tolerance 1e-14). Use representation='logical' or an explicit discrete X family; other angles require a validated LogicalMeasurementSynthesis protocol.
```

## Validation

The assertions above execute during generation and in `scripts/ml/execute_tutorials.py`.
See [physical architecture](../physical/architecture.md) and the
[release report](../release-report.md) for independent logical and physical checks.

## Physical interpretation

The exception is raised before finite GKP projection and Fock simulation.

## Limitations

Arbitrary angles require a separately validated measurement-synthesis instrument.
