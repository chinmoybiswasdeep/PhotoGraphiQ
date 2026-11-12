import numpy as np

from photographiqml.models import angle_encode

s = angle_encode([0, np.pi])
assert np.allclose(abs(s) ** 2, [0, 1, 0, 0])
print(abs(s).round(6))
