import numpy as np

import photographiqml as pqml

r = pqml.GKPBridge(cutoff=24, grid_points=2049).diagnostics()
assert not r["physical_muta_validated"]
assert all(0 < w <= 1.000001 for w in r["captured_weights"])
print("GKP captured weights:", np.round(r["captured_weights"], 5))
