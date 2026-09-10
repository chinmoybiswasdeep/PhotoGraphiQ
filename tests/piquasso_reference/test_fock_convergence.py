import numpy as np
import piquasso as pq

import photographiq as pg


def test_fock_cz_raw_reference_and_cutoff_convergence():
    estimates = []
    for cutoff in (8, 12):
        p = pg.Pattern().extend([pg.Prepare(0, 0.1), pg.Prepare(1, 0.15), pg.Entangle(0, 1, 0.1)])
        actual = pg.simulate(p, backend="piquasso-fock", cutoff=cutoff).state
        passive = np.array([[1, 0.05j], [0.05j, 1]])
        active = np.array([[0, 0.05j], [0.05j, 0]])
        with pq.Program() as program:
            pq.Q() | pq.Vacuum()
            pq.Q(0) | pq.Squeezing(r=-0.1)
            pq.Q(1) | pq.Squeezing(r=-0.15)
            pq.Q(0, 1) | pq.GaussianTransform(passive=passive, active=active)
        raw = (
            pq.PureFockSimulator(d=2, config=pq.Config(cutoff=cutoff, hbar=2))
            .execute(program)
            .state
        )
        raw.normalize()
        np.testing.assert_allclose(actual.density_matrix, raw.density_matrix, atol=1e-12)
        estimates.append(actual.photon_number(0))
    assert abs(estimates[0] - estimates[1]) < 1e-5


def test_offline_photon_resources():
    assert pg.FockInput.number(2).photon_added() == pg.FockInput.number(3)
    assert pg.FockInput.number(2).photon_subtracted() == pg.FockInput.number(1)
