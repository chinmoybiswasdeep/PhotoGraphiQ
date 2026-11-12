import numpy as np

import photographiqml as pqml

a, b = pqml.MuTA(2, 1), pqml.MuTA(2, 2)
p = a.initialize(3)
assert np.allclose(a.unitary(p), b.unitary(p))
print("Identity extension verified")
