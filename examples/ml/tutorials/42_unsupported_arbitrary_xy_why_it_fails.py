from photographiqml import GKPPhysicalConfig, PhysicalMuTA

config = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
model = PhysicalMuTA(1, physical_config=config)
try:
    model.run([1, 0], {"alpha.w0.c1": 0.37})
except NotImplementedError as error:
    print(error)
else:
    raise AssertionError("Unsupported physical measurement executed")
