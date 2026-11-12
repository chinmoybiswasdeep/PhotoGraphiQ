from photographiqml import GKPPhysicalConfig, PhysicalMuTA

config = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
model = PhysicalMuTA(1, physical_config=config)
result = model.run(
    [1, 0], mode="physical-conditional", analog_outcomes=dict.fromkeys(model.measurement_order, 0.0)
)
assert result.representation == "gkp-physical"
assert not result.convergence["certified"]
print(result.decoded_joint_probabilities)
print(result.diagnostics["marginal_code_subspace_leakage"])
