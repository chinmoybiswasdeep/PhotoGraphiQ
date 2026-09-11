# Finite-energy logical zero and one

## Goal

Compare actual finite codewords without replacing them by an orthogonal basis.

## Theory

The finite Gaussian comb has nonzero codeword overlap. Its Gram matrix controls normalization and the subspace projector.

## Code

Run `python examples/tutorials/37-gkp-finite-codewords.py` from the repository root.

```python
"""Finite-energy logical zero and one. See the matching documentation for physical limitations."""

import numpy as np
import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

zero, one = code.zero(), code.one()
print("Gram matrix:", code.gram)
print("zero leakage:", code.leakage(zero.amplitudes))
print("one leakage:", code.leakage(one.amplitudes))
assert abs(code.gram[0, 1]) > 0
assert code.leakage(zero.amplitudes) < 1e-10
print("captured mass:", code.resource(0).project(code.cutoff)[1])
```

## Output

```text
--8<-- "docs/tutorials/outputs/37-gkp-finite-codewords.txt"
```

## Validation

The executable assertions check the stated identities or refusal behavior. Physical
readout uses the existing conditional Fock backend; independent reference tests are
described in the [validation guide](../validation/gkp-logical-validation.md).
Printed finite fidelities and error probabilities are observations, not universal bounds.

## Convergence

Repeat with larger cutoff, grid_points and peaks independently. Overlap need not tend to zero when only numerical resolution changes.

## Limitations

All examples use finite, nonorthogonal codewords and exclusive total-photon cutoff.
They do not establish fault tolerance or full physical MuTA. Physical Y, pi/4 and
arbitrary XY remain unsupported. The [downstream contract](../development/photographiqml-contract.md)
states the precise supported interfaces and result meanings.
