import numpy as np

from photographiqml import GKPPhysicalConfig, PhysicalMuTA
from photographiqml.physical_training import DiscreteSearch

config = GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
model = PhysicalMuTA(1, physical_config=config)


def objective(angles):
    result = model.run([1, 0], angles, shots=1, seed=14)
    return result.decoded_joint_probabilities[(1,)]


search = DiscreteSearch(sweeps=1, seed=14).fit(model, objective, initial=np.zeros(4))
assert set(search.parameters) <= {0, np.pi}
assert len(search.evaluations) == 5
print("selected:", search.parameters)
print("estimated error:", search.loss)
