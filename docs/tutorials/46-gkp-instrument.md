# Encoded finite measurement instrument

## Goal

Distinguish ambiguity from outside-code leakage.

## Theory

Reciprocal code vectors produce unambiguous conclusive effects. A separate inconclusive effect completes the resolution within the code span.

## Code

Run `python examples/tutorials/46-gkp-instrument.py` from the repository root.

```python
"""Encoded finite measurement instrument. See the matching documentation for physical limitations."""

import numpy as np
import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

instrument = code.discrimination_instrument()
probabilities = instrument.probabilities(code.zero().amplitudes)
print(probabilities)
assert abs(probabilities[1]) < 1e-10
assert probabilities["inconclusive"] > 0
assert abs(probabilities["outside-code"]) < 1e-10
p, rho = instrument.conditional(code.zero().amplitudes, 0)
print("branch probability:", p, "trace:", np.trace(rho))
assert abs(np.trace(rho)-1) < 1e-10
```

## Output

```text
--8<-- "docs/tutorials/outputs/46-gkp-instrument.txt"
```

## Validation

The executable assertions check the stated identities or refusal behavior. Physical
readout uses the existing conditional Fock backend; independent reference tests are
described in the [validation guide](../validation/gkp-logical-validation.md).
Printed finite fidelities and error probabilities are observations, not universal bounds.

## Convergence

Compare the Gram eigenvalues with refined resource parameters. This is a mathematical instrument and has no supplied optical synthesis.

## Limitations

All examples use finite, nonorthogonal codewords and exclusive total-photon cutoff.
They do not establish fault tolerance or full physical MuTA. Physical Y, pi/4 and
arbitrary XY remain unsupported. The [downstream contract](../development/photographiqml-contract.md)
states the precise supported interfaces and result meanings.
