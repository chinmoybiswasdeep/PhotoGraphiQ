import tempfile
from pathlib import Path

import numpy as np

import photographiqml as pqml

m = pqml.MuTA(1)
m.freeze("alpha.w0.c1", 0.3)
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "model.json"
    m.save(p)
    restored = pqml.MuTA.load(p)
    assert np.allclose(m.unitary(), restored.unitary())
print("Logical model round trip verified")
