from photographiqml import GKPPhysicalConfig, PhysicalMuTA, lower_muta_to_gkp

config = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
model = PhysicalMuTA(1, physical_config=config)
angles = model.initialize(seed=4)
lowering = lower_muta_to_gkp(model, angles, config=config)
lowering.pattern.validate()
assert lowering.audit["supported"]
print(angles)
print({k: lowering.audit[k] for k in ("peak_live_modes", "hilbert_dimension", "cz_operations")})
