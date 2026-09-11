"""Logical Pauli frames. See the matching documentation for physical limitations."""

import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

frame = pg.LogicalPauliFrame(1, 0)
print("H:", frame.hadamard(), "S:", frame.phase())
print("CZ:", frame.cz(pg.LogicalPauliFrame(0, 1)))
print("adapted angle:", frame.xy_angle(0.3))
p = pg.Pattern(inputs=(0,)).measure(0, pg.PhysicalGKPReadout("Z", frame=frame), key="m")
p.append(pg.Signal("bit", pg.CallableExpression(lambda r: r["m"].bit, frozenset({"m"}))))
r = pg.simulate(
    p,
    inputs={0: code.zero()},
    backend="piquasso-fock",
    cutoff=code.cutoff,
    measurement_outcomes={"m": 0.1},
)
print(r.records)
assert r.records["bit"] == 1
