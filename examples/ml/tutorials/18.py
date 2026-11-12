import numpy as np

from photographiq.gaussian import wire_channel

S, N = wire_channel([0] * 4, 1.0)
assert np.allclose(S, np.eye(2))
assert np.trace(N) > 0
print(N.round(6))
