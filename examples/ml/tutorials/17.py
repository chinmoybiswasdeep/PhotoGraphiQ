import numpy as np

import photographiqml as pqml

p, h = pqml.Trainer("lbfgs", epochs=20).fit(
    lambda x: np.sum(x * x), [1, 2], gradient=lambda x: 2 * x
)
assert np.linalg.norm(p) < 1e-8
print(p.round(6))
