# Gaussian simulation


```python
pattern = pg.protocols.identity(squeezing=0.8)
shots = pg.run_shots(pattern, 1000, backend="gaussian", seed=123)
ensemble = shots.ensemble_state()
print(ensemble.covariance)

pattern = pg.Pattern().append(pg.Prepare(0, 0.8))
pattern.append(pg.Loss(0, transmissivity=0.9, thermal_photons=0.05))
pattern.measure(0, pg.Homodyne(angle=0.2, efficiency=0.85, noise=0.02))
```

Each shot receives a distinct child seed and executes a complete adaptive
trajectory. `shots.values(key)` extracts a record array. Seeds including zero
are reproducible. Finite squeezing, environmental photon loss and detector
inefficiency are separate mechanisms. `ensemble_state` includes covariance of
the trajectory means, not just the average conditional covariance. For nonlinear
adaptation the ensemble is generally not a Gaussian state; the container represents
only its moments, so Gaussian fidelity/parity formulas for that container need not
equal the mixture's actual observables. Compute per-trajectory observables and
average them instead when necessary.


See the [API reference](../api/index.md) and [tutorials](../tutorials/index.md) for executable examples.
