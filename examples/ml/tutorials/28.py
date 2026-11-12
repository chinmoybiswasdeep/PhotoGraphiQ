import numpy as np

import photographiqml as pqml

seen = []


def callback(p, h):
    seen.append((len(h.losses) - 1, float(np.linalg.norm(p))))


p, h = pqml.Trainer(epochs=3).fit(lambda p: np.sum(p * p), [1], callback=callback)
assert len(seen) == 4
print([step for step, norm in seen])
