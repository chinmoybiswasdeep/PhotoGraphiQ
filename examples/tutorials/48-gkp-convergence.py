"""Independent GKP convergence axes. See the matching documentation for physical limitations."""

import photographiq as pg

code = pg.GKPCode(peak_width=0.55, envelope=0.5, cutoff=80, peaks=6, grid_points=2049)

study = pg.measurement_convergence(code, (24, 48), basis="X", coefficients=(1, -1))
for row in study.rows:
    print(row)
assert study.rows[-1]["fidelity_to_previous"] > 0.99
print("Two-mode study: python -m experiments.gkp_logical_evidence")
