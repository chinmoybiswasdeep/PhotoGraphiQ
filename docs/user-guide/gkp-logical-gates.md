# Ideal-lattice gate realizations

```python
import photographiq as pg
code = pg.GKPCode(cutoff=80)
p = pg.Pattern(inputs=(0,)).append(code.logical_gate(0, "S"))
r = pg.simulate(p, inputs={0: code.plus()}, backend="piquasso-fock", cutoff=80)
print(code.diagnostics(r.state))
```

X and Z displace by L=√(2π) in q and p. H rotates by π/2. S applies
`QuadraticPhase(s=1)`. CZ uses `logical_cz(u,v)` at unit weight. These are
Gaussian realizations of logical gates on ideal lattice sites; they are approximate
relative to the finite basis. Compare target fidelity, Gram-aware subspace leakage,
stabilizers and independent cutoff sweeps. A unit norm alone is insufficient.

The same caution applies to composing these commands with readout. This release
does not enable Y measurement merely because S exists. See
[the derivations](../theory/gkp-measurement.md) and
[validation](../validation/gkp-logical-validation.md).
