# 43. Logical training to physical validation

## Goal

Audit a continuously trained logical candidate.

## Theory

Logical optimizer success says nothing about physical measurement support.

## Code

Run this standalone example after installing the pinned dependency and package.

```python
import numpy as np

from photographiqml import GKPBridge, MuTA, Trainer

model = MuTA(1)
parameters, history = Trainer(epochs=2).fit(
    lambda p: float((model.run([1, 0], p).probabilities[1] - 0.3) ** 2), np.full(4, 0.5)
)
audit = model.physical_capabilities(parameters)
assert not audit["supported"]
try:
    GKPBridge().run(model, [1, 0], parameters)
except NotImplementedError as error:
    print(error)
else:
    raise AssertionError("Continuous candidate was silently altered")
```

## Output

Captured from execution; upstream truncation warnings go to stderr and remain active.

```text
Physical GKP MuTA cannot lower alpha.w0.c0=0.49995317302484121 rad: PhotoGraphiQ 0.3.1 supports physical XY only at 0 and pi modulo 2pi (absolute tolerance 1e-14). Use representation='logical' or an explicit discrete X family; other angles require a validated LogicalMeasurementSynthesis protocol.
```

## Validation

The assertions above execute during generation and in `scripts/ml/execute_tutorials.py`.
See [physical architecture](../physical/architecture.md) and the
[release report](../release-report.md) for independent logical and physical checks.

## Physical interpretation

The candidate is rejected unchanged; it is not rounded into the discrete family.

## Limitations

Two logical optimizer steps toward a target output probability demonstrate the workflow, not converged task-learning performance.
