import numpy as np

from photographiqml import GKPBridge, MuTA, Trainer

model = MuTA(1)
parameters, history = Trainer(epochs=2).fit(
    lambda p: float((model.run([1, 0], p).probabilities[1] - 0.3) ** 2), np.full(4, 0.5)
)
audit = model.physical_capabilities(parameters)
assert not audit["supported"]
try:
    GKPBridge().run(model, [1, 0], parameters)
except NotImplementedError as error:
    print(error)
else:
    raise AssertionError("Continuous candidate was silently altered")
