# Finite-resource convergence studies

A successfully simulated pattern is not automatically converged. Every physical
result starts with `convergence.certified = False`. Backend retained-norm and
boundary diagnostics are retained; upstream truncation warnings and strict errors
are not suppressed or relaxed by the bridge.

| Axis | Interpretation | What to hold fixed |
| --- | --- | --- |
| cutoff | Exclusive total-photon numerical truncation | Physical widths, envelope, peaks, grid |
| grid_points | Wavefunction quadrature resolution | Cutoff and physical resource |
| peaks | Finite lattice-sum extent | Width/envelope and numerical resolution |
| peak_width | Physical squeezing/resource change | Numerical settings |
| envelope | Physical envelope/resource change | Numerical settings |

`GKPBridge.measurement_convergence` delegates the single-resource study to
PhotoGraphiQ. `PhysicalMuTA.physical_convergence` reruns the **whole pattern** for
one independently varied axis. It reports probabilities, changes between successive
points, ideal decoded-statistics discrepancy, leakage, standard errors, predicted
most-likely bit string and backend diagnostics. Every proposed setting is audited
before the first simulation. Values must increase strictly. A decreasing-width
physical improvement study can be inspected in reverse row order; no monotonicity
claim follows from the API's ordering convention.

```python
from photographiqml import PhysicalMuTA, GKPPhysicalConfig
model = PhysicalMuTA(physical_config=GKPPhysicalConfig(
    cutoff=24, peak_width=.9, envelope=.9, peaks=4, grid_points=1025))
study = model.physical_convergence([1, 0], [20, 24, 28],
    mode="physical-conditional",
    analog_outcomes=dict.fromkeys(model.measurement_order, 0.0))
assert not study["certified"]
```

Fixed analog branch comparisons remain conditional. Shot studies should compare
changes to statistical uncertainty and use independent validation seeds; the
workflow does not call small sampling fluctuations convergence. Finite codeword
overlap and finite squeezing may leave sizable logical errors after cutoff/grid
stability. Increasing cutoff alone does not repair the physical code.

The reproducible [evidence](evidence.json) varies all five axes separately and
includes a one-resource cutoff study. Run `python experiments/restricted_physical.py`
to regenerate it. The broad width/envelope .9 resource keeps this demonstration
small and is intentionally not presented as high-fidelity GKP computation.
