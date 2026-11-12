import numpy as np
from scipy.linalg import expm

import photographiqml as pqml
from photographiqml.logical import X
from photographiqml.models import haar_states, infidelity

m = pqml.MuTA(1)
s = haar_states(1, 7, 42)
t = s @ expm(0.4j * X).T


def loss(p):
    return infidelity(m.run_batch(s, p), t)


p, h = pqml.Trainer(epochs=100).fit(loss, np.zeros(4))
assert h.losses[-1] < 1e-4
print(round(h.losses[-1], 6))
