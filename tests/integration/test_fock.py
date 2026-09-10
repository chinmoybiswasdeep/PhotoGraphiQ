import numpy as np
import pytest

import photographiq as pg


def test_fock_number_and_counting():
    p = pg.Pattern().append(pg.Prepare("a", state=pg.FockInput.number(2)))
    result = pg.simulate(p, backend="piquasso-fock", cutoff=5, seed=2)
    assert result.state.photon_number("a") == 2
    assert result.state.parity() == 1
    p.measure("a", pg.PhotonNumber())
    result = pg.simulate(p, backend="piquasso-fock", cutoff=5, seed=2)
    assert result.outcomes["a"] == 2
    assert result.state.nodes == ()


def test_fock_conditional_beamsplitter():
    p = pg.Pattern().extend(
        [
            pg.Prepare(0, state=pg.FockInput.number(1)),
            pg.Prepare(1, 0),
            pg.BeamSplitter(0, 1, np.pi / 4),
        ]
    )
    p.measure(0, pg.PhotonNumber())
    for seed in range(4):
        result = pg.simulate(p, backend="piquasso-fock", cutoff=4, seed=seed)
        assert np.isclose(result.state.photon_number(1), 1 - result.outcomes[0])


def test_fock_capability_errors():
    p = pg.Pattern().append(pg.Prepare(0, 0)).measure(0)
    with pytest.raises(ValueError, match="cutoff"):
        pg.simulate(p, backend="piquasso-fock")
    assert np.isfinite(pg.simulate(p, backend="piquasso-fock", cutoff=4).outcomes[0])
    p = pg.Pattern().append(pg.Prepare(0, 2.0))
    with pytest.raises(ValueError, match="truncation"):
        pg.simulate(p, backend="piquasso-fock", cutoff=4)


def test_cat_and_cubic_resource():
    p = pg.Pattern().append(pg.Prepare(0, state=pg.FockInput.cat(0.3, 10)))
    p.append(pg.CubicPhase(0, 0.005))
    result = pg.simulate(p, backend="piquasso-fock", cutoff=12)
    assert np.isclose(np.trace(result.state.density_matrix), 1)
