# 37. Hard vs soft decoding

## Goal

Compare unknown hard confidence to a calibrated posterior.

## Theory

The soft model discriminates a specified preparation ensemble.

## Code

Run this standalone example after installing the pinned dependency and package.

```python
from photographiqml import GKPBridge

bridge = GKPBridge(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
hard = bridge.resource_readout().decoder.decode(0.2)
soft = bridge.resource_readout(decoder="soft").decoder.decode(0.2)
assert hard.confidence is None and soft.confidence is not None
assert abs(sum(soft.probabilities) - 1) < 1e-12
print("hard confidence:", hard.confidence)
print("preparation posterior:", soft.probabilities)
```

## Output

Captured from execution; upstream truncation warnings go to stderr and remain active.

```text
hard confidence: None
preparation posterior: (np.float64(0.9573438096254954), np.float64(0.042656190374504646))
```

## Validation

The assertions above execute during generation and in `scripts/ml/execute_tutorials.py`.
See [physical architecture](../physical/architecture.md) and the
[release report](../release-report.md) for independent logical and physical checks.

## Physical interpretation

Soft values concern equal-prior zero/one preparation labels.

## Limitations

They are not calibrated for arbitrary adaptive MuTA node states.
