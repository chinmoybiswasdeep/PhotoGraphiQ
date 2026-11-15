from photographiqml import GKPBridge

bridge = GKPBridge(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
hard = bridge.resource_readout().decoder.decode(0.2)
soft = bridge.resource_readout(decoder="soft").decoder.decode(0.2)
assert hard.confidence is None and soft.confidence is not None
assert abs(sum(soft.probabilities) - 1) < 1e-12
print("hard confidence:", hard.confidence)
print("preparation posterior:", soft.probabilities)
