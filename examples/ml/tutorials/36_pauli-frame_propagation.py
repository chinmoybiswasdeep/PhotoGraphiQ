from photographiqml import GKPPhysicalConfig, PhysicalMuTA
from photographiqml.lowering import node_frame

config = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
model = PhysicalMuTA(1, physical_config=config)
records = {("bit", v): int(i % 2) for i, v in enumerate(model.measurement_order)}
frame = node_frame(model, model.output_nodes[0], records)
assert frame.correction("X") == frame.z_bit
assert frame.correction("Z") == frame.x_bit
print(frame)
