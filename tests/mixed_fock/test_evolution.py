import numpy as np
import pytest

import photographiq as pg
from photographiq.backends.mixed_fock import MixedFockBackend


def test_binomial_loss_and_mixed_input():
    p = pg.Pattern(inputs=("a",)).append(pg.Loss("a", 0.6))
    result = pg.simulate(
        p, inputs={"a": pg.FockInput.number(2)}, backend="piquasso-mixed-fock", cutoff=5
    )
    np.testing.assert_allclose(
        list(result.state.probabilities.values())[:3], [0.16, 0.48, 0.36], atol=1e-12
    )
    rho = pg.FockDensityMatrix([[0.3, 0], [0, 0.7]], ((0,), (1,)))
    result = pg.simulate(
        pg.Pattern(inputs=(0,)), initial_state=rho, backend="piquasso-mixed-fock", cutoff=5
    )
    assert result.state.photon_number(0) == pytest.approx(0.7)


def test_entangled_pnr_and_homodyne():
    resource = pg.FockSuperposition.from_mapping({(0, 0): 2**-0.5, (1, 1): 2**-0.5})
    p = pg.Pattern().append(pg.PrepareResource(("a", "b"), resource))
    for measurement, value in [(pg.PhotonNumber(), 1), (pg.Homodyne(0.3), 0.4)]:
        pattern = p.copy().measure("a", measurement)
        kwargs = dict(cutoff=5, measurement_outcomes={"a": value})
        pure = pg.simulate(pattern, backend="piquasso-fock", **kwargs)
        mixed = pg.simulate(pattern, backend="piquasso-mixed-fock", **kwargs)
        np.testing.assert_allclose(
            pure.state.density_matrix, mixed.state.density_matrix, atol=1e-12
        )
        assert mixed.log_likelihood == pytest.approx(pure.log_likelihood)


def test_gaussian_thermal_and_noisy_vacuum():
    p = pg.Pattern(inputs=(0,)).measure(0, pg.Homodyne(noise=0.2, efficiency=0.7))
    result = pg.simulate(p, backend="piquasso-mixed-fock", cutoff=8, measurement_outcomes={0: 0.3})
    variance = 1 / 0.7 + 0.2
    assert np.exp(result.log_likelihood) == pytest.approx(
        np.exp(-(0.3**2) / (2 * variance)) / np.sqrt(2 * np.pi * variance), rel=1e-8
    )
    p = pg.Pattern(inputs=(0,))
    result = pg.simulate(
        p,
        inputs={0: pg.GaussianInput(covariance=((1.2, 0), (0, 1.2)))},
        backend="piquasso-mixed-fock",
        cutoff=12,
    )
    assert result.state.photon_number(0) == pytest.approx(0.1, abs=1e-9)


def test_guards_and_serialization():
    with pytest.raises(ValueError, match="positive"):
        pg.FockDensityMatrix([[1.1, 0], [0, -0.1]], ((0,), (1,)))
    backend = MixedFockBackend(12, max_matrix_bytes=100)
    with pytest.raises(MemoryError):
        backend.prepare(0)
    p = pg.Pattern().append(pg.Prepare(0, state=pg.FockDensityMatrix.thermal(0.1, 5)))
    assert p.to_json() == pg.Pattern.from_json(p.to_json()).to_json()
