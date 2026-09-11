# Independent GKP convergence axes

## Goal

Inspect probability and residual convergence rather than captured mass alone.

## Theory

Cutoff changes the represented Hilbert space; grid and peaks change resource quadrature accuracy. Width and envelope change the physical resource.

## Code

Run `python examples/tutorials/48-gkp-convergence.py` from the repository root.

```python
"""Independent GKP convergence axes. See the matching documentation for physical limitations."""

import numpy as np
import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

study = pg.measurement_convergence(code, (24,48), basis="X", coefficients=(1,-1))
for row in study.rows:
    print(row)
assert study.rows[-1]["fidelity_to_previous"] > 0.99
print("Two-mode study: python -m experiments.gkp_logical_evidence")
```

## Output

```text
--8<-- "docs/tutorials/outputs/48-gkp-convergence.txt"
```

## Validation

The executable assertions check the stated identities or refusal behavior. Physical
readout uses the existing conditional Fock backend; independent reference tests are
described in the [validation guide](../validation/gkp-logical-validation.md).
Printed finite fidelities and error probabilities are observations, not universal bounds.

## Convergence

The two-mode evidence includes joint readout, finite target fidelity, subspace leakage and stabilizers. A small state-preparation sweep cannot certify a large graph.

## Limitations

All examples use finite, nonorthogonal codewords and exclusive total-photon cutoff.
They do not establish fault tolerance or full physical MuTA. Physical Y, pi/4 and
arbitrary XY remain unsupported. The [downstream contract](../development/photographiqml-contract.md)
states the precise supported interfaces and result meanings.
