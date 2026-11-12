import numpy as np

from photographiqml import GKPBridge
from photographiqml.lowering import check_photographiq_contract

print(check_photographiq_contract())
bridge = GKPBridge(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
assert np.isclose(np.linalg.norm(bridge.encode([1, 0]).amplitudes), 1)
print(bridge.diagnostics())
