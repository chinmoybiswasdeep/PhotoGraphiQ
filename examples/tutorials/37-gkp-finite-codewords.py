"""Finite-energy logical zero and one. See the matching documentation for physical limitations."""

import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

zero, one = code.zero(), code.one()
print("Gram matrix:", code.gram)
print("zero leakage:", code.leakage(zero.amplitudes))
print("one leakage:", code.leakage(one.amplitudes))
assert abs(code.gram[0, 1]) > 0
assert code.leakage(zero.amplitudes) < 1e-10
print("captured mass:", code.resource(0).project(code.cutoff)[1])
