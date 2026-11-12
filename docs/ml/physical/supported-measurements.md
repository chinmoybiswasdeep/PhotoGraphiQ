# Supported measurements

The physical MuTA family measures every non-output vertex in signed X:
`alpha = 0` or `pi` modulo `2*pi`, with PhotoGraphiQ's absolute tolerance `1e-14`.
The pi setting changes outcome labeling. It is not a continuously tunable physical
homodyne approximation to a logical equatorial measurement.

| Measurement | Ideal logical model | Physical GKP primitive | Physical MuTA use |
| --- | --- | --- | --- |
| X / XY(0) | Yes | Yes | Intermediate and output |
| XY(pi) | Yes | X with flipped bit | Intermediate |
| Z | Computational output/instrument | Yes | Final output only |
| Y / XY(pi/2) | Yes | Unsupported | Rejected |
| XY(pi/4) | Yes | Unsupported | Rejected |
| Arbitrary XY | Yes | Unsupported | Rejected |

Output basis is explicitly `"X"` or `"Z"` for all output modes. Intermediate Z
measurements are not part of this MuTA lowering. NaNs, infinities and finite angles
outside the supported tolerance fail. Angles are never rounded to the family.

```python
from photographiqml import MuTA
model = MuTA(1)
audit = model.physical_capabilities({"alpha.w0.c1": 0.37})
assert not audit["supported"]
print(audit["unsupported_nodes"])
```

Tests poison codeword preparation, resource projection and simulator entry points
to prove unsupported requests cannot reach Fock allocation. The check covers all
nodes, including a late unsupported measurement in a multilayer model.

## Boundary to arbitrary physical MuTA

A validated logical equatorial measurement synthesis/injection instrument is
required, together with its non-Gaussian resources, feed-forward protocol,
finite-energy error model and convergence tests. An `alpha -> homodyne angle`
substitution does not supply this instrument. The current signed-X family does
not demonstrate arbitrary MuTA, universality, continuous trainability, finite-resource
expressivity monotonicity or quantum advantage. Loss channels and detector-noise
settings are not exposed by this lowering; adding them requires validated physical
instrument support, not post hoc noise on decoded bits.
