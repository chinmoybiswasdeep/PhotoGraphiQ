# Preparing a downstream physical measurement step

## Goal

Use encoded primitives through the public Pattern API.

## Theory

The downstream package requests a logical basis. PhotoGraphiQ supplies its bosonic realization, analog record and explicit capability boundary.

## Code

Run `python examples/tutorials/49-gkp-downstream-step.py` from the repository root.

```python
"""Preparing a downstream physical measurement step. See the matching documentation for physical limitations."""

import numpy as np
import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

measurement = code.logical_measurement("XY", alpha=np.pi)
p = pg.Pattern(inputs=(0,)).measure(0, measurement, key="logical")
r = pg.simulate(p, inputs={0:code.plus()}, backend="piquasso-fock", cutoff=code.cutoff,
                measurement_outcomes={"logical":0.1})
print(r.outcomes["logical"])
assert r.outcomes["logical"].bit == 1
print("One signed-X step only. Full physical MuTA remains unsupported.")
```

## Output

```text
--8<-- "docs/tutorials/outputs/49-gkp-downstream-step.txt"
```

## Validation

The executable assertions check the stated identities or refusal behavior. Physical
readout uses the existing conditional Fock backend; independent reference tests are
described in the [validation guide](../validation/gkp-logical-validation.md).
Printed finite fidelities and error probabilities are observations, not universal bounds.

## Convergence

The cross-mode regression validates an X step after physical CZ. Downstream adaptive scheduling, output policy and full-circuit convergence are still required.

## Limitations

All examples use finite, nonorthogonal codewords and exclusive total-photon cutoff.
They do not establish fault tolerance or full physical MuTA. Physical Y, pi/4 and
arbitrary XY remain unsupported. The [downstream contract](../development/photographiqml-contract.md)
states the precise supported interfaces and result meanings.
