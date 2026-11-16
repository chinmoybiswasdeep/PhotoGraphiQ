# GKP diagrams

GKP diagrams **execute** existing PhotoGraphiQ GKP primitives inside a typed diagram. No GKP
logical rewrites exist, and no new GKP operation is introduced. For the code itself see
[GKP code](../gkp-code.md), [GKP measurements](../gkp-measurements.md) and
[GKP decoding](../gkp-decoding.md).

## Generators

| Box | Signature | Semantics |
|---|---|---|
| `GKPPrepare` | `GKPPrepare(mode, code, c0=1.0, c1=0.0)` | The finite-energy codeword c₀\|0~⟩ + c₁\|1~⟩ of a `photographiq.GKPCode`, from `code.encode(c0, c1)`. Fock semantics only (`gaussian = False`). |
| `ModularDecode` | `ModularDecode(source, key, spacing=sqrt(2π))` | A classical box. It reads the real record `source` and writes the bit round(m / spacing) mod 2, the nearest-cell parity. |

With ħ = 2, the q-peaks of the square-code \|0~⟩ sit at even multiples of √(2π), so the
default `spacing` decodes Z from a q-homodyne record. `GKPPrepare` holds a code object and
cannot be read back with `CVDiagram.from_dict`.

## Z readout as a diagram

```python
import numpy as np
import photographiq as pg
from photographiq.calculus.cv_diagrams import (
    CZ, CVDiagram, GKPPrepare, Homodyne, ModularDecode, Rotation, run_fock,
)

code = pg.GKPCode(cutoff=40, peak_width=0.6, envelope=0.6, peaks=6, grid_points=2049)
readout = CVDiagram(
    (),
    (GKPPrepare("g", code, np.cos(0.4), np.sin(0.4)), Homodyne("g", 0.0, "m"), ModularDecode("m", "bit")),
    (),
)
run = run_fock(readout, 40, outcomes={"m": 0.3})
print(run.records, round(run.density, 6))
# {'m': 0.3, 'bit': 0} 0.443702
```

`run.density` is the record density at m = 0.3. To get P(bit = 0), integrate the density
over outcomes. Integrate with a midpoint rule **aligned to the decoding cells**
[(k − ½)s, (k + ½)s]. A grid that is not aligned misclassifies the points near cell
boundaries. For the state above (cutoff 40), a plain Riemann sum over 561 points on [−14, 14]
gave 0.7906. The cell-aligned rule of [tutorial 9](examples.md#9-gkp-syndrome-diagram) gives
0.7869, which matches `multimode_readout`.

## Recorded results (R68, status `pass`)

Width = envelope = 0.6, 6 peaks, logical state cos 0.4\|0~⟩ + sin 0.4\|1~⟩:

| Quantity | P(Z = 0) |
|---|---|
| Readout diagram (cutoff 40 and 48) | 0.78686 |
| Existing `pg.multimode_readout` | 0.78684 |
| Untruncated grid model | 0.78675 |
| Ideal cos² 0.4 | 0.84835 (the gap is the finite-energy error) |
| Z-syndrome diagram (SUM onto \|0~⟩, cutoff 64) | 0.723102 |
| Independent grid computation of the same syndrome protocol | 0.723102 |

## Z-syndrome extraction

`Rotation(a, π/2)`, `CZ(d, a, 1)`, `Rotation(a, −π/2)` is the SUM gate q_a → q_a + q_d
(and p_d → p_d − p_a). Tutorial 9 checks this on its Gaussian symplectic matrix. A SUM from
the data onto a \|0~⟩ ancilla, followed by a q-homodyne and parity of the ancilla, decodes
the data's Z value:

```python
syndrome = CVDiagram(
    (),
    (
        GKPPrepare("d", code, np.cos(0.4), np.sin(0.4)),
        GKPPrepare("a", code, 1.0, 0.0),
        Rotation("a", np.pi / 2), CZ("d", "a", 1.0), Rotation("a", -np.pi / 2),
        Homodyne("a", 0.0, "m"),
        ModularDecode("m", "bit"),
    ),
    ("d",),
)
```

The decoded statistics differ from the direct readout: 0.7231 in R68 (cutoff 64), and 0.7234
in tutorial 9 (cutoff 40, coarser grid), against 0.7868 for the direct readout. This gap is
the finite-energy error of the SUM. The ancilla's own peak width adds to the record. It is a
property of finite-energy codewords at width 0.6, not a numerical artefact: the untruncated
grid model of the same protocol gives the same value.

## Scope

- These diagrams are finite-energy, finite-cutoff numerical executions.
- They make no fault-tolerance claim and implement no decoder beyond nearest-cell parity.
- Arbitrary-angle GKP measurements and logical gates are available in PhotoGraphiQ's GKP
  modules, not as diagram rewrites.
