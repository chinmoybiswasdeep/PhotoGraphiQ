"""Run with: python examples/non_gaussian.py"""

import numpy as np

import photographiq as pg

pattern = pg.non_gaussian.photon_subtraction(theta=0.2)
result = pg.simulate(
    pattern,
    backend="piquasso-fock",
    cutoff=24,
    seed=42,
    inputs={"in": pg.CatResource(1.0)},
    measurement_outcomes={"count": 1},
)
print("Herald probability:", np.exp(result.log_likelihood))
print("Conditional parity:", result.state.parity())

pattern = pg.non_gaussian.cubic_injection(gamma=0.3, squeezing=0.2)
study = pg.cutoff_convergence(pattern, [36, 48, 64], measurement_outcomes={"m": 0.4})
for row in study.rows:
    print(row["cutoff"], row["fidelity_to_previous"], row["minimum_retained_norm"])
# This is an exact-outcome density, not the probability of a finite detector bin.
print("Homodyne density:", study.results[-1].measurement_statistics["m"]["value"])
