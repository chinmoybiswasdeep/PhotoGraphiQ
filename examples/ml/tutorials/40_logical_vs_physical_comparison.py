from photographiqml import GKPPhysicalConfig, PhysicalMuTA, compare_logical_physical

config = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
model = PhysicalMuTA(1, physical_config=config)
result = model.run(
    [1, 0], mode="physical-conditional", analog_outcomes=dict.fromkeys(model.measurement_order, 0.0)
)
comparison = compare_logical_physical(result)
assert comparison["finite_state_fidelity"] is None
assert comparison["total_variation_distance"] >= 0
print(comparison)
