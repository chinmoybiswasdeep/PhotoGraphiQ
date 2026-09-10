"""Photon-number measurement. Run from the repository after installing dev/visualization extras."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import photographiq as pg

resource = pg.FockSuperposition.from_mapping({(0, 0): 2**-0.5, (1, 1): 2**-0.5})
pattern = (
    pg.Pattern().append(pg.PrepareResource(("a", "b"), resource)).measure("a", pg.PhotonNumber())
)
result = pg.simulate(pattern, backend="piquasso-fock", cutoff=6, measurement_outcomes={"a": 1})
print("Branch probability:", np.exp(result.log_likelihood))
print("Remaining photon number:", result.state.photon_number("b"))
assert np.isclose(np.exp(result.log_likelihood), 0.5)
assert np.isclose(result.state.photon_number("b"), 1)
pattern.draw()
plt.close("all")
