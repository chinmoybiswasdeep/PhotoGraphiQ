import photographiq as pg

code = pg.GKPCode(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
pattern = pg.Pattern(inputs=(0,))
pattern.append(pg.Measure(0, code.logical_measurement("Z"), "readout"))
result = pg.simulate(
    pattern,
    inputs={0: code.zero()},
    cutoff=24,
    backend="piquasso-fock",
    measurement_outcomes={"readout": 0.2},
)
record = result.records["readout"]
assert record.raw_outcome == 0.2 and record.bit in (0, 1)
assert record.confidence is None
print(record)
