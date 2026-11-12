import numpy as np

import photographiqml as pqml

m = pqml.MuTA(1)
r = m.run([1, 0])
assert np.allclose(r.probabilities, [1, 0])
print(r.probabilities.round(6))
