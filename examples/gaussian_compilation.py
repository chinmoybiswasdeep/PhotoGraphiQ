import photographiq as pg

circuit = pg.Circuit(1).rotate(0, 0.4).squeeze(0, 0.2)
pattern = circuit.compile(squeezing=1.0)
channel = pg.gaussian_channel(pattern)
result = pg.simulate(pattern, inputs={0: pg.GaussianInput.coherent(0.3)}, seed=17, frame=True)
print("Ideal symplectic matrix:\n", channel.matrix)
print("Finite-squeezing added noise:\n", channel.noise)
print("One conditional output:\n", result.state.covariance)
