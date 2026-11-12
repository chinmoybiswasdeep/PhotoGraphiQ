from photographiqml import GKPPhysicalConfig, PhysicalMuTA

config = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
model = PhysicalMuTA(1, physical_config=config)
result = model.run([1, 0], shots=3, seed=14)
assert len(result.sampled_output_bits) == 3
assert result.standard_errors is not None
print("mean conditional POVM:", result.decoded_joint_probabilities)
print("standard errors:", result.standard_errors)
print("sampled bit frequencies:", result.empirical_probabilities)
