import numpy as np

from photographiqml.diagnostics import pure_qfi
from photographiqml.logical import Z

H = (np.kron(Z, np.eye(2)) + np.kron(np.eye(2), Z)) / 2
s = np.array([1, 0, 0, 1]) / np.sqrt(2)
qfi = pure_qfi(s, H)
assert np.isclose(qfi, 4)
print(qfi)
