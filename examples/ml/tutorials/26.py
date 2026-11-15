import numpy as np

import photographiqml as pqml

m = pqml.MuTA(1)
r = m.run([1, 0])
value = r.expectation(np.diag([1, -1]))
assert np.isclose(value, 1)
print(value)
