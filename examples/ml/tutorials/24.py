import numpy as np

import photographiqml as pqml
from photographiqml.expressivity import state_fisher

m = pqml.MuTA(1)
F = state_fisher(m, [1, 0], list(m.initialize(1).values()))
assert np.linalg.eigvalsh(F).min() > -1e-9
print("Local rank:", np.linalg.matrix_rank(F, tol=1e-8))
