# Logical X from p homodyne

## Goal

Read the finite plus state using momentum homodyne.

## Theory

The hbar=2 Fourier transform maps ideal plus/minus to even/odd momentum cells. Logical X is p homodyne, not q homodyne.

## Code

Run `python examples/tutorials/39-gkp-x-readout.py` from the repository root.

```python
"""Logical X from p homodyne. See the matching documentation for physical limitations."""

import numpy as np
import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

p = pg.Pattern(inputs=(0,)).measure(0, code.logical_measurement("X"), key="x")
r = pg.simulate(p, inputs={0: code.plus()}, backend="piquasso-fock", cutoff=code.cutoff,
                measurement_outcomes={"x": -0.2})
print(r.outcomes["x"])
assert r.outcomes["x"].bit == 0
for state in (code.plus(), code.minus()):
    v = np.array(state.amplitudes)
    print(np.einsum("i,kij,j->k", v.conj(), pg.modular_effects(code.cutoff, "X"), v).real)
```

## Output

```text
--8<-- "docs/tutorials/outputs/39-gkp-x-readout.txt"
```

## Validation

The executable assertions check the stated identities or refusal behavior. Physical
readout uses the existing conditional Fock backend; independent reference tests are
described in the [validation guide](../validation/gkp-logical-validation.md).
Printed finite fidelities and error probabilities are observations, not universal bounds.

## Convergence

Sweep cutoff and resource envelope separately. The continuous-comb Fourier oracle in the tests establishes finite readout probabilities.

## Limitations

All examples use finite, nonorthogonal codewords and exclusive total-photon cutoff.
They do not establish fault tolerance or full physical MuTA. Physical Y, pi/4 and
arbitrary XY remain unsupported. The [downstream contract](../development/photographiqml-contract.md)
states the precise supported interfaces and result meanings.
