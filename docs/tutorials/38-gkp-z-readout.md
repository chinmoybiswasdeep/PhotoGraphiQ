# Logical Z from q homodyne

## Goal

Obtain an analog Z record and its modular parity.

## Theory

Ideal q peaks lie at mL; cell parity is the logical Z label. Finite peaks have a nonzero tail across cell boundaries.

## Code

Run `python examples/tutorials/38-gkp-z-readout.py` from the repository root.

```python
"""Logical Z from q homodyne. See the matching documentation for physical limitations."""

import numpy as np
import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

p = pg.Pattern(inputs=(0,)).measure(0, code.logical_measurement("Z"), key="z")
r = pg.simulate(p, inputs={0: code.zero()}, backend="piquasso-fock", cutoff=code.cutoff,
                measurement_outcomes={"z": 0.25})
print(r.outcomes["z"])
print(r.measurement_statistics)
assert r.outcomes["z"].bit == 0
assert r.outcomes["z"].raw_outcome == 0.25
v = np.array(code.zero().amplitudes)
print("integrated Z probabilities:", np.einsum("i,kij,j->k", v.conj(), pg.modular_effects(code.cutoff), v).real)
```

## Output

```text
--8<-- "docs/tutorials/outputs/38-gkp-z-readout.txt"
```

## Validation

The executable assertions check the stated identities or refusal behavior. Physical
readout uses the existing conditional Fock backend; independent reference tests are
described in the [validation guide](../validation/gkp-logical-validation.md).
Printed finite fidelities and error probabilities are observations, not universal bounds.

## Convergence

Compare integrated probabilities with measurement_convergence at cutoffs 24,48,80. A fixed analog branch tests conditioning, not statistical accuracy.

## Limitations

All examples use finite, nonorthogonal codewords and exclusive total-photon cutoff.
They do not establish fault tolerance or full physical MuTA. Physical Y, pi/4 and
arbitrary XY remain unsupported. The [downstream contract](../development/photographiqml-contract.md)
states the precise supported interfaces and result meanings.
