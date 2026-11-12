# 31. PhotoGraphiQ 0.3.1 encoded interface

## Goal

Check public encoded primitives before building a physical model.

## Theory

Finite codewords overlap; normalized encoding is not an exact isometry.

## Code

Run this standalone example after installing the pinned dependency and package.

```python
import numpy as np

from photographiqml import GKPBridge
from photographiqml.lowering import check_photographiq_contract

print(check_photographiq_contract())
bridge = GKPBridge(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
assert np.isclose(np.linalg.norm(bridge.encode([1, 0]).amplitudes), 1)
print(bridge.diagnostics())
```

## Output

Captured from execution; upstream truncation warnings go to stderr and remain active.

```text
{'version': '0.3.1', 'audited_commit': 'db07f9f9bf47da841bfa6b206562c5a3ffb121d3'}
{'cutoff': 24, 'captured_weights': [1.0, 0.999999999991365], 'codeword_overlap': 0.5339586741671603, 'gram_eigenvalues': [0.46604132583283975, 1.5339586741671603], 'physical_muta_validated': False}
```

## Validation

The assertions above execute during generation and in `scripts/ml/execute_tutorials.py`.
See [physical architecture](../physical/architecture.md) and the
[release report](../release-report.md) for independent logical and physical checks.

## Physical interpretation

The normalized vector is a one-mode finite Fock resource.

## Limitations

Preparation alone does not validate an MBQC computation.
