# GKP Hadamard

## Goal

Compare the Fourier-rotated zero with the finite plus target.

## Theory

A pi/2 oscillator rotation realizes logical H on the ideal square lattice. A finite comb need not be self-dual under the Fourier transform.

## Code

Run `python examples/tutorials/42-gkp-hadamard.py` from the repository root.

```python
"""GKP Hadamard. See the matching documentation for physical limitations."""

import numpy as np
import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

p = pg.Pattern(inputs=(0,)).append(code.logical_gate(0, "H"))
r = pg.simulate(p, inputs={0: code.zero()}, backend="piquasso-fock", cutoff=code.cutoff)
fidelity = abs(np.vdot(code.plus().amplitudes, r.state.state_vector)) ** 2
print("H target fidelity:", fidelity)
print(code.diagnostics(r.state))
assert 0 < fidelity <= 1 + 1e-12
```

## Output

```text
--8<-- "docs/tutorials/outputs/42-gkp-hadamard.txt"
```

## Validation

The executable assertions check the stated identities or refusal behavior. Physical
readout uses the existing conditional Fock backend; independent reference tests are
described in the [validation guide](../validation/gkp-logical-validation.md).
Printed finite fidelities and error probabilities are observations, not universal bounds.

## Convergence

H itself preserves Fock occupation. Resource convergence and finite target mismatch still require checks.

## Limitations

All examples use finite, nonorthogonal codewords and exclusive total-photon cutoff.
They do not establish fault tolerance or full physical MuTA. Physical Y, pi/4 and
arbitrary XY remain unsupported. The [downstream contract](../development/photographiqml-contract.md)
states the precise supported interfaces and result meanings.
