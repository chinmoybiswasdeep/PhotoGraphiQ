# Two-mode logical CZ

## Goal

Execute a finite two-mode encoded CZ and inspect correlated readout.

## Theory

At ideal sites exp(i q1 q2/2)=(-1)^(mn). Total photon truncation couples the represented support of both modes.

## Code

Run `python examples/tutorials/44-gkp-cz.py` from the repository root.

```python
"""Two-mode logical CZ. See the matching documentation for physical limitations."""

import numpy as np
import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

v = np.array(code.plus().amplitudes)
data = {(i,j): v[i]*v[j] for i in range(code.cutoff) for j in range(code.cutoff-i)}
mass = sum(abs(a)**2 for a in data.values())
source = pg.FockSuperposition.from_mapping({k:a/np.sqrt(mass) for k,a in data.items()})
p = pg.Pattern(inputs=(0,1)).append(code.logical_cz(0,1))
r = pg.simulate(p, initial_state=source, backend="piquasso-fock", cutoff=code.cutoff)
print("total preparation mass:", mass)
probabilities = pg.multimode_readout(r.state, {0:"X", 1:"Z"})
print(probabilities)
assert abs(sum(probabilities["joint_probabilities"].values())-1) < 1e-9
```

## Output

```text
--8<-- "docs/tutorials/outputs/44-gkp-cz.txt"
```

## Validation

The executable assertions check the stated identities or refusal behavior. Physical
readout uses the existing conditional Fock backend; independent reference tests are
described in the [validation guide](../validation/gkp-logical-validation.md).
Printed finite fidelities and error probabilities are observations, not universal bounds.

## Convergence

Run the independent CZ evidence at total cutoffs 24,40,64. Joint probabilities must be compared directly; multiplying marginals loses correlations.

## Limitations

All examples use finite, nonorthogonal codewords and exclusive total-photon cutoff.
They do not establish fault tolerance or full physical MuTA. Physical Y, pi/4 and
arbitrary XY remain unsupported. The [downstream contract](../development/photographiqml-contract.md)
states the precise supported interfaces and result meanings.
