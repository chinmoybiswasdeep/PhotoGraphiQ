# Logical Pauli frames

## Goal

Propagate ideal binary byproducts and resolve readout labels.

## Theory

Conjugation by X^x Z^z changes an XY angle to (-1)^x alpha+pi*z. H swaps frame bits and S maps z to z xor x.

## Code

Run `python examples/tutorials/45-gkp-frames.py` from the repository root.

```python
"""Logical Pauli frames. See the matching documentation for physical limitations."""

import numpy as np
import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

frame = pg.LogicalPauliFrame(1, 0)
print("H:", frame.hadamard(), "S:", frame.phase())
print("CZ:", frame.cz(pg.LogicalPauliFrame(0, 1)))
print("adapted angle:", frame.xy_angle(0.3))
p = pg.Pattern(inputs=(0,)).measure(0, pg.PhysicalGKPReadout("Z", frame=frame), key="m")
p.append(pg.Signal("bit", pg.CallableExpression(lambda r:r["m"].bit, frozenset({"m"}))))
r = pg.simulate(p, inputs={0:code.zero()}, backend="piquasso-fock", cutoff=code.cutoff,
                measurement_outcomes={"m":0.1})
print(r.records)
assert r.records["bit"] == 1
```

## Output

```text
--8<-- "docs/tutorials/outputs/45-gkp-frames.txt"
```

## Validation

The executable assertions check the stated identities or refusal behavior. Physical
readout uses the existing conditional Fock backend; independent reference tests are
described in the [validation guide](../validation/gkp-logical-validation.md).
Printed finite fidelities and error probabilities are observations, not universal bounds.

## Convergence

Frame algebra is exact on ideal qubits and tested independently. It is not a claim that finite envelope corrections can always be deferred exactly.

## Limitations

All examples use finite, nonorthogonal codewords and exclusive total-photon cutoff.
They do not establish fault tolerance or full physical MuTA. Physical Y, pi/4 and
arbitrary XY remain unsupported. The [downstream contract](../development/photographiqml-contract.md)
states the precise supported interfaces and result meanings.
