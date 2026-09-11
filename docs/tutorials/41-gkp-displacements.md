# Logical X and Z displacements

## Goal

Measure finite target distortion under lattice displacements.

## Theory

X shifts q by L and Z shifts p by L. Their actions include translating the finite envelope; ideal Pauli identities do not imply unit finite-code fidelity.

## Code

Run `python examples/tutorials/41-gkp-displacements.py` from the repository root.

```python
"""Logical X and Z displacements. See the matching documentation for physical limitations."""

import numpy as np
import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

for gate in ("X", "Z"):
    source = code.zero() if gate == "X" else code.plus()
    target = code.one() if gate == "X" else code.minus()
    p = pg.Pattern(inputs=(0,)).append(code.logical_gate(0, gate))
    r = pg.simulate(p, inputs={0: source}, backend="piquasso-fock", cutoff=code.cutoff)
    fidelity = abs(np.vdot(target.amplitudes, r.state.state_vector)) ** 2
    print(gate, "target fidelity:", fidelity, "leakage:", code.leakage(r.state.state_vector))
    assert 0 < fidelity < 1
```

## Output

```text
--8<-- "docs/tutorials/outputs/41-gkp-displacements.txt"
```

## Validation

The executable assertions check the stated identities or refusal behavior. Physical
readout uses the existing conditional Fock backend; independent reference tests are
described in the [validation guide](../validation/gkp-logical-validation.md).
Printed finite fidelities and error probabilities are observations, not universal bounds.

## Convergence

Compare cutoffs 48,80,112. Physical envelope error can remain even when numerical fidelity changes stop.

## Limitations

All examples use finite, nonorthogonal codewords and exclusive total-photon cutoff.
They do not establish fault tolerance or full physical MuTA. Physical Y, pi/4 and
arbitrary XY remain unsupported. The [downstream contract](../development/photographiqml-contract.md)
states the precise supported interfaces and result meanings.
