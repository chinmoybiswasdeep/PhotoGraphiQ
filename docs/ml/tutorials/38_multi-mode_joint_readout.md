# 38. Multi-mode joint readout

## Goal

Retain correlations of two physical outputs.

## Theory

A tensor POVM acts on the full density matrix before marginalization.

## Code

Run this standalone example after installing the pinned dependency and package.

```python
from photographiqml import GKPPhysicalConfig, PhysicalMuTA

config = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
model = PhysicalMuTA(1, physical_config=config)
model = PhysicalMuTA(2, physical_config=config)
result = model.run(
    [1, 0, 0, 0],
    mode="physical-conditional",
    analog_outcomes=dict.fromkeys(model.measurement_order, 0.0),
)
p = result.decoded_joint_probabilities
assert len(p) == 4 and abs(sum(p.values()) - 1) < 1e-9
marginals = list(result.decoded_marginals.values())
connected = p[(0, 0)] - marginals[0][0] * marginals[1][0]
assert abs(connected) > 1e-5
print(p)
print("connected probability:", connected)
```

## Output

Captured from execution; upstream truncation warnings go to stderr and remain active.

```text
{(0, 0): 0.6578184991070793, (0, 1): 0.15903906568906884, (1, 0): 0.14565397323522333, (1, 1): 0.03748846196862815}
connected probability: 0.0014959319688053752
```

## Validation

The assertions above execute during generation and in `scripts/ml/execute_tutorials.py`.
See [physical architecture](../physical/architecture.md) and the
[release report](../release-report.md) for independent logical and physical checks.

## Physical interpretation

The joint distribution is not reconstructed from independent marginals.

## Limitations

Broad finite codewords and a fixed analog branch limit the physical interpretation.
