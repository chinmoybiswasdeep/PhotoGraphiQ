import numpy as np
import pytest

import photographiq as pg
from photographiq.backends.fock import PiquassoFockBackend


def test_strong_cubic_convergence_not_just_unit_norm():
    p = pg.Pattern().extend([pg.Prepare(0, 0), pg.CubicPhase(0, 0.65)])
    study = pg.cutoff_convergence(p, [12, 24, 48, 80])
    assert all(abs(row["norm"] - 1) < 1e-12 for row in study.rows)
    assert study.rows[1]["fidelity_to_previous"] < 0.9999
    assert study.rows[-1]["fidelity_to_previous"] > 0.9999
    assert (
        study.rows[-1]["trace_distance_to_previous"] < study.rows[1]["trace_distance_to_previous"]
    )
    assert abs(study.results[-1].state.photon_number(0) - 3 * 0.65**2 / 4) < 1e-5
    assert abs(study.results[-1].state.quadrature(0, np.pi / 2)[0] - 0.65) < 1e-4


def test_finite_cubic_injection_cutoff_convergence():
    p = pg.non_gaussian.cubic_injection(0.3, 0.2)
    study = pg.cutoff_convergence(p, [36, 48, 64], measurement_outcomes={"m": 0.4})
    assert study.rows[-1]["fidelity_to_previous"] > 1 - 4e-7
    assert study.rows[-1]["minimum_retained_norm"] > study.rows[0]["minimum_retained_norm"]
    assert study.rows[-1]["comparable_to_previous"]


def test_cat_factory_convergence_and_branch_comparability():
    def factory(c):
        return pg.Pattern().append(pg.Prepare(0, state=pg.FockInput.cat(1.5, c)))

    study = pg.cutoff_convergence(factory, [8, 16, 24])
    assert study.rows[-1]["fidelity_to_previous"] > 1 - 1e-8
    p = pg.Pattern().append(pg.Prepare(0, 0)).append(pg.CubicPhase(0, 0.3)).measure(0)
    study = pg.cutoff_convergence(p, [8, 16])
    assert not study.rows[-1]["comparable_to_previous"]
    assert study.rows[-1]["fidelity_to_previous"] is None
    fixed = pg.cutoff_convergence(p, [8, 16], measurement_outcomes={0: 0.2})
    assert fixed.rows[-1]["comparable_to_previous"]
    with pytest.raises(ValueError):
        pg.cutoff_convergence(p, [12, 8])


def test_truncation_warnings_and_rejection():
    p = pg.Pattern().append(pg.Prepare(0, 0.4))
    with pytest.warns(RuntimeWarning, match="retained Fock norm"):
        pg.simulate(p, backend="piquasso-fock", cutoff=10)
    with pytest.raises(ValueError, match="truncation"):
        pg.simulate(p, backend="piquasso-fock", cutoff=3)
    backend = PiquassoFockBackend(6)
    with pytest.warns(RuntimeWarning, match="boundary population"):
        backend.prepare(0, state=pg.FockInput.number(5))
    with pytest.raises(ValueError):
        backend.prepare_resource((1, 2), pg.FockSuperposition.number((3, 3)))
    assert np.isclose(backend.get_state().norm, 1)
