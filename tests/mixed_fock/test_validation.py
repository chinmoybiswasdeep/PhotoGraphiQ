import numpy as np
import pytest

import photographiq as pg
from photographiq.backends.mixed_fock import MixedFockBackend


def test_thermal_attenuation_and_correlated_density():
    p = pg.Pattern(inputs=(0,)).append(pg.Loss(0, 0.7, 0.2))
    result = pg.simulate(p, backend="piquasso-mixed-fock", cutoff=12)
    assert result.state.photon_number(0) == pytest.approx(0.06, abs=1e-9)
    rho = pg.FockDensityMatrix([[0.5, 0.2j], [-0.2j, 0.5]], ((0, 0), (1, 1)))
    p = pg.Pattern(inputs=("a", "b")).append(pg.Output(("b",)))
    result = pg.simulate(p, initial_state=rho, backend="piquasso-mixed-fock", cutoff=6)
    np.testing.assert_allclose(result.state.density_matrix[:2, :2], np.eye(2) / 2)
    assert np.trace(
        result.state.density_matrix @ result.state.density_matrix
    ).real == pytest.approx(0.5)


@pytest.mark.parametrize("addition", [False, True])
def test_mixed_ladder_weighting(addition):
    rho = pg.FockDensityMatrix([[0.5, 0], [0, 0.5]], ((1,), (2,)))
    command = pg.PhotonAdd(0) if addition else pg.PhotonSubtract(0)
    result = pg.simulate(
        pg.Pattern(inputs=(0,)).append(command),
        initial_state=rho,
        backend="piquasso-mixed-fock",
        cutoff=8,
    )
    expected = {(2,): 0.4, (3,): 0.6} if addition else {(0,): 1 / 3, (1,): 2 / 3}
    for occupation, probability in expected.items():
        assert result.state.probabilities[occupation] == pytest.approx(probability)


def test_mixed_homodyne_sampling_is_reproducible_and_gaussian():
    p = pg.Pattern(inputs=(0,)).measure(0, pg.Homodyne())
    kwargs = dict(backend="piquasso-mixed-fock", cutoff=6, seed=12)
    left, right = pg.simulate(p, **kwargs), pg.simulate(p, **kwargs)
    assert left.outcomes == right.outcomes
    value = left.outcomes[0]
    assert np.exp(left.log_likelihood) == pytest.approx(
        np.exp(-(value**2) / 2) / np.sqrt(2 * np.pi), rel=1e-10
    )


def test_mixed_resource_preflight_and_impossible_branches():
    backend = MixedFockBackend(6)
    with pytest.raises(ValueError, match="match"):
        backend.validate_preparation(pg.FockDensityMatrix.thermal(0.1, 8))
    backend.prepare(0, state=pg.FockInput.number(0))
    with pytest.raises(ValueError, match="new and unique"):
        backend.prepare(0)
    with pytest.raises(ValueError, match="zero"):
        backend.measure(0, pg.PhotonNumber(), outcome=1)
    with pytest.raises(ValueError, match="integer"):
        backend.measure(0, pg.PhotonNumber(), outcome=0.2)
    with pytest.raises(NotImplementedError, match="PhotonNumber"):
        backend.measure(0, pg.Heterodyne())
    with pytest.raises(ValueError, match="transmissivity"):
        backend.loss(0, 1.1)


def test_mixed_convergence_and_nonlinear_pure_limit():
    p = pg.Pattern(inputs=(0,)).extend([pg.CubicPhase(0, 0.02), pg.Kerr(0, 0.1), pg.Loss(0, 0.9)])
    study = pg.cutoff_convergence(
        p, [10, 16], backend="piquasso-mixed-fock", high_order_moments=True
    )
    assert study.rows[-1]["fidelity_to_previous"] > 0.99999
    p = pg.Pattern(inputs=(0,)).extend(
        [pg.Squeeze(0, 0.1), pg.CubicPhase(0, 0.02), pg.Kerr(0, 0.1)]
    )
    pure = pg.simulate(p, backend="piquasso-fock", cutoff=16)
    mixed = pg.simulate(p, backend="piquasso-mixed-fock", cutoff=16)
    assert pg.fidelity(pure.state, mixed.state) > 1 - 1e-10
