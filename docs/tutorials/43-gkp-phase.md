# GKP phase gate

## Goal

Inspect the physical shear that realizes ideal-lattice S.

## Theory

At q=mL, exp(iq^2/4) is 1 on even m and i on odd m. Within a finite peak this phase varies continuously and distorts the wavefunction.

## Code

Run `python examples/tutorials/43-gkp-phase.py` from the repository root.

```python
"""GKP phase gate. See the matching documentation for physical limitations."""

import numpy as np
import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

p = pg.Pattern(inputs=(0,)).append(code.logical_gate(0, "S"))
r = pg.simulate(p, inputs={0: code.plus()}, backend="piquasso-fock", cutoff=code.cutoff)
target = code.encode(1, 1j)
fidelity = abs(np.vdot(target.amplitudes, r.state.state_vector)) ** 2
print("S target fidelity:", fidelity)
print(code.diagnostics(r.state))
assert 0 < fidelity < 1
```

## Output

```text
--8<-- "docs/tutorials/outputs/43-gkp-phase.txt"
```

## Validation

The executable assertions check the stated identities or refusal behavior. Physical
readout uses the existing conditional Fock backend; independent reference tests are
described in the [validation guide](../validation/gkp-logical-validation.md).
Printed finite fidelities and error probabilities are observations, not universal bounds.

## Convergence

The shear populates higher Fock levels. Compare output fidelity and stabilizers at increasing cutoff; norm conservation is insufficient.

## Limitations

All examples use finite, nonorthogonal codewords and exclusive total-photon cutoff.
They do not establish fault tolerance or full physical MuTA. Physical Y, pi/4 and
arbitrary XY remain unsupported. The [downstream contract](../development/photographiqml-contract.md)
states the precise supported interfaces and result meanings.
