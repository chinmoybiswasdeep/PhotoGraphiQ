# Finite-energy GKP infrastructure

At hbar=2, this square lattice uses logical shift spacing `sqrt(2*pi)` and
stabilizer translations twice that spacing. `GKPResource` builds a normalized
Gaussian comb with separate peak width and envelope controls, then projects it
into Fock space at execution cutoff.

```python
import photographiq as pg
resource = pg.GKPResource(logical=0, peak_width=0.5, envelope=0.5)
input_state, captured_weight = resource.project(48)
print(captured_weight)
pattern = pg.Pattern(inputs=(0,))
result = pg.simulate(pattern, inputs={0: input_state},
                     backend="piquasso-fock", cutoff=48)
print(pg.gkp.stabilizers(result.state))
```

`pg.gkp.superposition(alpha,beta,cutoff=...)` combines the two finite basis states
and normalizes with their actual overlap. They are not assumed exactly orthogonal.
`decode_shift(value)` returns nearest-cell residual and cell parity, with residual
in `[-spacing/2,spacing/2)`. `correction_pattern(resource=logical_plus)` constructs
one finite-ancilla SUM syndrome extraction and modular correction. Its callable
correction is runtime-only and cannot be serialized as JSON or differentiated.

Increase `peaks` to test lattice-sum truncation, `grid_points` for quadrature
resolution, and cutoff for Fock projection. Captured weight measures only the
last of these. Stabilizer expectations need not equal one for finite resources.
This is experimental preparation, diagnostics and elementary correction
infrastructure, not a demonstrated fault-tolerant architecture or threshold study.
