"""Compile a Gaussian circuit and separate its ideal map from finite-resource noise.

Run from the repository root. The printed conditional covariance describes one
trajectory; channel.noise describes the unconditional finite-squeezing channel.
See docs/tutorials/07-gaussian-compilation.md for figures and exercises.
"""

import photographiq as pg

circuit = pg.Circuit(1).rotate(0, 0.4).squeeze(0, 0.2)
pattern = circuit.compile(squeezing=1.0)
channel = pg.gaussian_channel(pattern)
result = pg.simulate(pattern, inputs={0: pg.GaussianInput.coherent(0.3)}, seed=17, frame=True)
print("Ideal symplectic matrix:\n", channel.matrix)
print("Finite-squeezing added noise:\n", channel.noise)
print("One conditional output:\n", result.state.covariance)
