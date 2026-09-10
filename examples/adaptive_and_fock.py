"""Compare real adaptive homodyne readings with integer photon counts.

The first dictionary contains quadrature outcomes. The second contains a
deterministic count for a prepared number state; neither dictionary is amplitudes.
See tutorials 04 and 09 for interpretation and validation.
"""

import photographiq as pg

pattern = pg.protocols.adaptive(squeezing=0.8)
print(pg.simulate(pattern, parameters={"k": 0.3}, seed=0).outcomes)

pattern = pg.Pattern().append(pg.Prepare("photon", state=pg.FockInput.number(1)))
pattern.measure("photon", pg.PhotonNumber())
print(pg.simulate(pattern, backend="piquasso-fock", cutoff=5, seed=0).outcomes)
