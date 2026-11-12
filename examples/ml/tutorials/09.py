import numpy as np

import photographiqml as pqml

s = np.array([1, 0, 0, 1]) / np.sqrt(2)
r = pqml.MuTA(2).run(s)
assert abs(np.vdot(s, r.state)) ** 2 > 1 - 1e-12
print(r.probabilities.round(6))
