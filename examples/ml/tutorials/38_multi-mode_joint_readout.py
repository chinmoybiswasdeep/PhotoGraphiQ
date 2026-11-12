from photographiqml import GKPPhysicalConfig, PhysicalMuTA

config = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
model = PhysicalMuTA(1, physical_config=config)
model = PhysicalMuTA(2, physical_config=config)
result = model.run(
    [1, 0, 0, 0],
    mode="physical-conditional",
    analog_outcomes=dict.fromkeys(model.measurement_order, 0.0),
)
p = result.decoded_joint_probabilities
assert len(p) == 4 and abs(sum(p.values()) - 1) < 1e-9
marginals = list(result.decoded_marginals.values())
connected = p[(0, 0)] - marginals[0][0] * marginals[1][0]
assert abs(connected) > 1e-5
print(p)
print("connected probability:", connected)
