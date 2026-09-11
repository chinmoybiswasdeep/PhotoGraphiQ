# The pi/4 measurement boundary

## Goal

Verify that ideal pi/4 probabilities do not enable physical injection.

## Theory

An arbitrary logical Rz rotation requires a validated encoded protocol. Rotated optical homodyne is not a substitute for a logical XY measurement.

## Code

Run `python examples/tutorials/47-gkp-magic-boundary.py` from the repository root.

```python
"""The pi/4 measurement boundary. See the matching documentation for physical limitations."""

import numpy as np
import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

alpha = np.pi/4
print("ideal qubit probabilities:", pg.IdealLogicalXYMeasurement(alpha).probabilities([1,0]))
try:
    code.logical_measurement("XY", alpha=alpha)
except NotImplementedError as error:
    print("physical request refused:", error)
else:
    raise AssertionError("Unvalidated physical pi/4 readout must be refused")
```

## Output

```text
--8<-- "docs/tutorials/outputs/47-gkp-magic-boundary.txt"
```

## Validation

The executable assertions check the stated identities or refusal behavior. Physical
readout uses the existing conditional Fock backend; independent reference tests are
described in the [validation guide](../validation/gkp-logical-validation.md).
Printed finite fidelities and error probabilities are observations, not universal bounds.

## Convergence

There is no physical convergence claim for this unsupported operation. Injection needs a literature-backed complete branch protocol before validation can begin.

## Limitations

All examples use finite, nonorthogonal codewords and exclusive total-photon cutoff.
They do not establish fault tolerance or full physical MuTA. Physical Y, pi/4 and
arbitrary XY remain unsupported. The [downstream contract](../development/photographiqml-contract.md)
states the precise supported interfaces and result meanings.
