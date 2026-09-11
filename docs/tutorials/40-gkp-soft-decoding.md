# Soft GKP decoding

## Goal

Compare parity decisions with ensemble posteriors.

## Theory

Bayes conditioning uses the two specified finite preparation densities. It is not normalization of overlaps with an unknown logical state.

## Code

Run `python examples/tutorials/40-gkp-soft-decoding.py` from the repository root.

```python
"""Soft GKP decoding. See the matching documentation for physical limitations."""

import numpy as np
import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

decoder = pg.SoftDecisionDecoder(code, "Z")
for raw in (-2.6, 0.1, 1.2, 2.6):
    result = decoder.decode(raw)
    print(result)
    assert abs(sum(result.probabilities) - 1) < 1e-12
    assert result.confidence == max(result.probabilities)
print("nearest-cell:", code.decode(1.2))
```

## Output

```text
--8<-- "docs/tutorials/outputs/40-gkp-soft-decoding.txt"
```

## Validation

The executable assertions check the stated identities or refusal behavior. Physical
readout uses the existing conditional Fock backend; independent reference tests are
described in the [validation guide](../validation/gkp-logical-validation.md).
Printed finite fidelities and error probabilities are observations, not universal bounds.

## Convergence

Recompute likelihoods at larger cutoff. Calibration tests use synthetic draws from an independent continuous-comb ensemble.

## Limitations

All examples use finite, nonorthogonal codewords and exclusive total-photon cutoff.
They do not establish fault tolerance or full physical MuTA. Physical Y, pi/4 and
arbitrary XY remain unsupported. The [downstream contract](../development/photographiqml-contract.md)
states the precise supported interfaces and result meanings.
