import numpy as np

import photographiqml as pqml

p, h = pqml.Trainer(epochs=180).fit(lambda p: np.sum((p - 0.3) ** 2), [1, -1])
assert np.linalg.norm(p - 0.3) < 0.001
print(p.round(4))
