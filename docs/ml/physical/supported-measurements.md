# Supported measurements

`PhysicalMuTA` has two measurement families. The default is unchanged.

| Family | Config | Angles | Physical instrument |
| --- | --- | --- | --- |
| `measurement_family="X"` (default) | `GKPPhysicalConfig(xy_protocol="signed-x")` (default) | 0 or pi | p homodyne, nearest-cell decoding, bit flip at pi (`PhysicalGKPReadout`) |
| `measurement_family="XY"` | `xy_protocol="clifford"` | k pi/2 | `pg.GKPLogicalXYMeasurement`: Gaussian shear S/S^dagger, frame flip for Z, then X readout |
| `measurement_family="XY"` | `xy_protocol="auto"` (XY default) | any finite angle | as `"clifford"` for k pi/2; repeat-until-success magic-state injection otherwise |
| `measurement_family="XY"` | `xy_protocol="injection"` | any finite angle | magic-state injection for every angle, including k pi/2 |

The family and the protocol must agree: `measurement_family="X"` requires
`xy_protocol="signed-x"`, and `"XY"` requires one of the other three (`ValueError`
otherwise). `PhysicalMuTA(measurement_family="XY")` without a config uses
`GKPPhysicalConfig(xy_protocol="auto")`.

## Signed-X family (default)

The signed-X family measures every non-output vertex in signed X:
`alpha = 0` or `pi` modulo `2*pi`, with PhotoGraphiQ's absolute tolerance `1e-14`.
The pi setting changes outcome labeling. It is not a continuously tunable physical
homodyne approximation to a logical equatorial measurement.

| Measurement | Ideal logical model | Physical GKP primitive | signed-X family | XY family |
| --- | --- | --- | --- | --- |
| X / XY(0) | Yes | Yes | Intermediate and output | Intermediate (no operation before the X readout; one ancilla round with `injection`) |
| XY(pi) | Yes | X with flipped bit | Intermediate | Intermediate (Z frame flip; with `injection`, one ancilla round applies Z instead) |
| Z | Computational output/instrument | Yes | Final output only | Final output only |
| Y / XY(pi/2) | Yes | Gaussian shear S^dagger + X readout, or Y-state injection | Rejected | Intermediate |
| XY(pi/4) | Yes | Magic-state injection (repeat-until-success) | Rejected | Intermediate (`auto`, `injection`) |
| Arbitrary XY | Yes | Magic-state injection (repeat-until-success) | Rejected | Intermediate (`auto`, `injection`) |

Output basis is explicitly `"X"` or `"Z"` for all output modes. Intermediate Z
measurements are not part of this MuTA lowering. NaNs, infinities and finite angles
outside the supported tolerance fail. Angles are never rounded to the family. With the
XY family only non-finite angles (and, for `xy_protocol="clifford"`, angles other than
k pi/2) are rejected.

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

The signed-X family alone does not demonstrate arbitrary MuTA, universality,
continuous trainability, finite-resource expressivity monotonicity or quantum advantage.

## XY family: Clifford angles and magic-state injection

`measurement_family="XY"` lowers every measured vertex to `pg.GKPLogicalXYMeasurement`, which
applies D(-a) = diag(1, e^{-ia}) to the data mode and then reads out X_L (see the
[GKP user guide](../../user-guide/gkp.md#physical-logical-y-and-xy-measurements)):

- **Clifford angles k pi/2** (absolute tolerance 1e-12): S or S^dagger as the Gaussian shear
  exp(+-i q^2/4); Z as an exact outcome relabelling. Deterministic, no ancilla.
- **Other angles**: repeat-until-success gate teleportation. Each round prepares a
  finite-energy encoded magic state (|0~> + e^{i phi}|1~>)/norm, applies a logical CNOT
  (Gaussian SUM), measures the ancilla q quadrature and decodes it; bit 1 doubles the
  remaining angle. A Clifford remainder is finished with a frame flip or a Y-state ancilla.

```python
import numpy as np
from photographiqml import GKPPhysicalConfig, PhysicalMuTA
config = GKPPhysicalConfig(cutoff=56, backend="native-fock", xy_protocol="clifford",
                           max_dimension=200_000)
model = PhysicalMuTA(1, 1, one_column=True, measurement_family="XY", physical_config=config)
names = list(model.parameters())
audit = model.physical_capabilities(dict.fromkeys(names, 0.0) | {names[1]: 0.3})
print([row["parameter"] for row in audit["unsupported_nodes"]])  # the 0.3 node
```

What is supported and what is not:

- **No rotated-homodyne substitution.** A quadrature at angle a is not a measurement of
  M_L(a) for the square GKP code; no `alpha -> homodyne angle` mapping exists anywhere.
- **Magic-state preparation is not simulated.** The encoded magic state is a supplied
  normalized finite-Fock vector built from the same finite codewords. Its cost and errors
  are outside the model.
- **Round budget and heralded failure.** `GKPPhysicalConfig.max_injection_rounds`
  (default 4) bounds the non-Clifford rounds. An exhausted budget is reported in the record
  (`heralded_success=False`); it is not hidden or retried. For a generic angle the
  ideal-code success probability is 1 - 2^-K.
- **Noise options need mixed backends.** The standalone instrument accepts ancilla loss,
  detector efficiency and homodyne noise, which require `native-mixed-fock` or
  `piquasso-mixed-fock`. The PhotoGraphiQML lowering validates `piquasso-fock` and
  `native-fock` only and does not expose these options.
- **Finite energy.** The Gaussian SUM and shear distort the finite envelope and every
  failed round adds energy to the data mode, so injected nodes need a larger cutoff. The
  audit accounts for the ancilla mode, not for this energy growth; the backend's
  retained-norm guard stops underresolved runs.

The XY family does not demonstrate fault tolerance, universality of the finite-energy
instrument or quantum advantage. Evidence and limits: experiments R60-R67
(`paper_experiments_ml/14_gkp_xy/`) and the
[final report](../../development/GKP_ARBITRARY_ANGLE_FINAL_REPORT.md).
