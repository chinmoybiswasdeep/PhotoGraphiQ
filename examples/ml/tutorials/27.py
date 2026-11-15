import numpy as np

import photographiqml as pqml

target = np.array([0.5, -0.4])


def loss(p):
    return float(np.mean((p - target) ** 2))


p, h = pqml.Trainer("lbfgs", epochs=20).fit(loss, [0, 0])
assert loss(p) < 1e-10
print(round(loss(p), 8))
