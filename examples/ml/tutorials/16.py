import numpy as np

from photographiqml.training import finite_difference

p = np.array([0.3, -0.2])
g = finite_difference(lambda x: np.sum(x**2), p)
assert np.allclose(g, 2 * p, atol=1e-8)
print(g.round(6))
