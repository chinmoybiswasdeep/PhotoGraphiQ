# Non-Gaussian simulation


```python
import photographiq as pg

resource = pg.FockInput.cat(alpha=0.4, cutoff=12).photon_added()
pattern = pg.Pattern().append(pg.Prepare("a", state=resource))
pattern.append(pg.CubicPhase("a", gamma=0.005))
result = pg.simulate(pattern, backend="native-fock", cutoff=16, seed=4)
print(result.state.photon_number("a"), result.state.retained_norms)
```

Photon-number measurement is supported via `PhotonNumber()`. Use a cutoff scan
to check observables. A low retained norm raises an exception. Gaussian MBQC uses
no Fock cutoff. Noisy (inefficient) Fock homodyne, loss and mixed (density-matrix) inputs
need a mixed-Fock backend (`native-mixed-fock`, `piquasso-mixed-fock`; see
[Mixed Fock](mixed-fock.md)). Finite-energy GKP resources are covered in [GKP](gkp.md).
Compilation covers the Gaussian-plus-cubic gate set with experimental Kerr/quartic synthesis
([universal compilation](universal-compilation.md)); arbitrary unitaries are not compiled.
Fock heterodyne and general-dyne are unsupported. Derivations:
[Fock and non-Gaussian theory](../theory/non-gaussian.md).


See the [API reference](../api/index.md) and [tutorials](../tutorials/index.md) for executable examples.
