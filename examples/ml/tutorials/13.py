import numpy as np

import photographiqml as pqml

m = pqml.QuantumInstrumentModel(pqml.MuTA(2, one_column=True))
b = m.run(np.array([1, 0, 0, 1]) / np.sqrt(2))
assert np.isclose(sum(x.probability for x in b), 1)
print([(x.outcome, round(x.probability, 6)) for x in b])
