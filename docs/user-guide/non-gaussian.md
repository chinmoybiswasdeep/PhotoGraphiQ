# Non-Gaussian simulation


```python
resource = pg.FockInput.cat(alpha=0.4, cutoff=12).photon_added()
pattern = pg.Pattern().append(pg.Prepare("a", state=resource))
pattern.append(pg.CubicPhase("a", gamma=0.005))
result = pg.simulate(pattern, backend="piquasso-fock", cutoff=16, seed=4)
print(result.state.photon_number("a"), result.state.retained_norms)
```

Photon-number measurement is supported via `PhotonNumber()`. Use a cutoff scan
to check observables. A low retained norm raises an exception. Gaussian MBQC uses
no Fock cutoff. Noisy Fock homodyne, mixed Fock inputs, GKP resources and universal
non-Gaussian compilation are explicitly unavailable in this release.


See the [API reference](../api/index.md) and [tutorials](../tutorials/index.md) for executable examples.
