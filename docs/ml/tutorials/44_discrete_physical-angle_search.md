# 44. Discrete physical-angle search

## Goal

Evaluate categorical candidates with physical shots.

## Theory

Coordinate flips preserve exact 0/pi measurements; the physical loss has sampling noise.

## Code

Run this standalone example after installing the pinned dependency and package.

```python
import numpy as np

from photographiqml import GKPPhysicalConfig, PhysicalMuTA
from photographiqml.physical_training import DiscreteSearch

config = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
model = PhysicalMuTA(1, physical_config=config)


def objective(angles):
    result = model.run([1, 0], angles, shots=1, seed=14)
    return result.decoded_joint_probabilities[(1,)]


search = DiscreteSearch(sweeps=1, seed=14).fit(model, objective, initial=np.zeros(4))
assert set(search.parameters) <= {0, np.pi}
assert len(search.evaluations) == 5
print("selected:", search.parameters)
print("estimated error:", search.loss)
```

## Output

Captured from execution; upstream truncation warnings go to stderr and remain active.

```text
selected: [0. 0. 0. 0.]
estimated error: 0.18161338187027842
```

## Validation

The assertions above execute during generation and in `scripts/ml/execute_tutorials.py`.
See [physical architecture](../physical/architecture.md) and the
[release report](../release-report.md) for independent logical and physical checks.

## Physical interpretation

Every objective evaluation executes a supported finite GKP pattern.

## Limitations

One shot and a relabeling-only family cannot establish expressive trainability.
