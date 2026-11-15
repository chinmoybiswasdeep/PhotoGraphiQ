from photographiqml import GKPPhysicalConfig, PhysicalMuTA

config = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
model = PhysicalMuTA(1, physical_config=config)
study = model.physical_convergence(
    [1, 0],
    [513, 1025],
    axis="grid_points",
    mode="physical-conditional",
    analog_outcomes=dict.fromkeys(model.measurement_order, 0.0),
)
assert not study["certified"]
print(study["axis_type"])
print([row["max_probability_delta"] for row in study["rows"]])
