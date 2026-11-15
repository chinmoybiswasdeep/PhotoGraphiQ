# 36. Pauli-frame propagation

## Goal

Inspect virtual corrections from flow records.

## Theory

An X readout anticommutes with the Z component of its logical frame.

## Code

Run this standalone example after installing the pinned dependency and package.

```python
from photographiqml import GKPPhysicalConfig, PhysicalMuTA
from photographiqml.lowering import node_frame

config = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
model = PhysicalMuTA(1, physical_config=config)
records = {("bit", v): int(i % 2) for i, v in enumerate(model.measurement_order)}
frame = node_frame(model, model.output_nodes[0], records)
assert frame.correction("X") == frame.z_bit
assert frame.correction("Z") == frame.x_bit
print(frame)
```

## Output

Captured from execution; upstream truncation warnings go to stderr and remain active.

```text
LogicalPauliFrame(x_bit=1, z_bit=0)
```

## Validation

The assertions above execute during generation and in `scripts/ml/execute_tutorials.py`.
See [physical architecture](../physical/architecture.md) and the
[release report](../release-report.md) for independent logical and physical checks.

## Physical interpretation

Frames alter interpretation without a physical analog displacement.

## Limitations

This algebra example supplies interpreted bits; real runs obtain them from the decoder.
