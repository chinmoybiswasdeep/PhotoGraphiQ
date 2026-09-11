# Physical and ideal GKP measurements

```python
import photographiq as pg
code = pg.GKPCode(cutoff=48)
pattern = pg.Pattern(inputs=(0,)).measure(0, code.logical_measurement("X"), key="x")
result = pg.simulate(pattern, inputs={0: code.plus()}, backend="piquasso-fock",
                     cutoff=48, measurement_outcomes={"x": 0.2})
print(result.outcomes["x"])
```

X means p homodyne; Z means q homodyne. The postselection above is an analog
value with a probability density, not a logical bit. For sampled trajectories
omit `measurement_outcomes`. Results include the raw value and modular residual.
Physical Y and arbitrary XY raise `NotImplementedError`. Signed X is supported
through `logical_measurement("XY", alpha=0)` or `alpha=π`.

For an ideal qubit reference use `IdealLogicalXYMeasurement(alpha).probabilities(v)`.
For finite-state probabilities use `modular_effects(cutoff, basis)` contracted
with the density matrix, or `multimode_readout(state, bases)`. The latter preserves
joint correlations. `code.discrimination_instrument()` is a different mathematical
POVM with explicit inconclusive/outside-code outcomes; it is not modular homodyne.

`MeasurementInstrument({label: (M1, M2, ...)})` validates completeness and defines
updates. Its `conditional` method retains the output system; Pattern measurement
traces it out. General destructive instruments require mixed Fock. Pure Fock accepts
rank-one effects only. See [the theory](../theory/gkp-measurement.md).
