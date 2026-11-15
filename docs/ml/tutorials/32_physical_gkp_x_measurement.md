# 32. Physical GKP X measurement

## Goal

Execute p readout on a finite GKP plus state.

## Theory

Logical X is a physical p-homodyne instrument with modular decoding.

## Code

Run this standalone example after installing the pinned dependency and package.

```python
import photographiq as pg

code = pg.GKPCode(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
pattern = pg.Pattern(inputs=(0,))
pattern.append(pg.Measure(0, code.logical_measurement("X"), "readout"))
result = pg.simulate(
    pattern,
    inputs={0: code.plus()},
    cutoff=24,
    backend="piquasso-fock",
    measurement_outcomes={"readout": 0.2},
)
record = result.records["readout"]
assert record.raw_outcome == 0.2 and record.bit in (0, 1)
assert record.confidence is None
print(record)
```

## Output

Captured from execution; upstream truncation warnings go to stderr and remain active.

```text
LogicalDecodeResult(bit=0, raw_outcome=0.2, cell=0, residual=0.2, decoder='nearest-cell', confidence=None, probabilities=None, code_subspace_leakage=None, frame_correction=0)
```

## Validation

The assertions above execute during generation and in `scripts/ml/execute_tutorials.py`.
See [physical architecture](../physical/architecture.md) and the
[release report](../release-report.md) for independent logical and physical checks.

## Physical interpretation

The .2 value is a conditional analog outcome, not a logical bit.

## Limitations

One analog branch has no shot standard error or convergence certificate.
