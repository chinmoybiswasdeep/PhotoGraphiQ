import numpy as np

from photographiq.gaussian import teleportation_matrix

T = teleportation_matrix(0)
assert np.allclose(np.linalg.matrix_power(T, 4), np.eye(2))
print("CV cell ingredient verified; CVMuTA is a proposal")
