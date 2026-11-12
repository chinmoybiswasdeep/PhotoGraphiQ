import numpy as np

import photographiqml as pqml

m = pqml.MuTA(2, one_column=True)
r = m.run([1, 0, 0, 0], {"alpha.w1.c1": np.pi / 2})
assert np.allclose(r.probabilities, [0.5, 0, 0, 0.5])
print(r.probabilities.round(6))
