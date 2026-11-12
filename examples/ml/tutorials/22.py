import numpy as np

import photographiqml as pqml
from photographiqml.diagnostics import concurrence

m = pqml.MuTA(2, one_column=True)
phi = 0.7
c = concurrence(m.run([1, 0, 0, 0], {"alpha.w1.c1": phi}).state)
assert np.isclose(c, abs(np.sin(phi)))
print(round(c, 6))
