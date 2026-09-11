# Finite-energy GKP code

```python
import photographiq as pg
code = pg.GKPCode(peak_width=0.4, envelope=0.4, cutoff=48,
                  peaks=8, grid_points=4097)
zero, one = code.zero(), code.one()
plus, minus = code.plus(), code.minus()
plus_i = code.encode(1, 1j)
print(code.gram)
```

The Gram matrix is explicitly nonidentity. Resources use the existing comb
implementation and codewords are cached per code instance. `encode` preserves
their actual overlap before normalizing. It is not an isometric encoder.

Use `code.resource(0).project(code.cutoff)` for captured projection mass,
`code.leakage(vector_or_density)` for subspace leakage, and `code.diagnostics(state)`
for Gram, leakage and stabilizers on a single-mode Fock snapshot. Resource width
and envelope are physical parameters; cutoff, peaks and grid require separate
numerical checks. See [the downstream contract](../development/photographiqml-contract.md)
and [measurement derivation](../theory/gkp-measurement.md).
