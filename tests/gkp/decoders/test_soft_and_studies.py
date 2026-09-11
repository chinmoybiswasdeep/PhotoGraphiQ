import numpy as np
import pytest
from scipy.integrate import simpson

import photographiq as pg
from tests.gkp.convergence.test_modular_reference import reference_density, reference_metrics


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_soft_decoder_matches_continuous_likelihoods_and_calibrated_draws(basis):
    code = pg.GKPCode(0.55, 0.5, 80, 6, 4097)
    decoder = pg.SoftDecisionDecoder(code, basis)
    coefficients = ((1, 0), (0, 1)) if basis == "Z" else ((1, 1), (1, -1))
    densities = [reference_density(code, c, basis) for c in coefficients]
    grid = np.linspace(-12, 12, 2401)
    likelihood = np.array([[density(x) for x in grid] for density in densities])
    # Synthetic mixture draws on a fine grid; test posterior calibration in aggregate.
    weights = likelihood.sum(axis=0)
    rng = np.random.default_rng(421)
    samples = rng.choice(len(grid), size=1500, p=weights / weights.sum())
    posteriors = np.array([decoder.decode(grid[i]).probabilities for i in samples])
    expected = (likelihood / likelihood.sum(axis=0))[:, samples].T
    np.testing.assert_allclose(posteriors, expected, atol=3e-5)
    assert abs(posteriors[:, 0].mean() - 0.5) < 0.05
    assert simpson(likelihood[0], x=grid) == pytest.approx(1, abs=1e-8)


def test_measurement_study_tracks_more_than_projection_mass():
    code = pg.GKPCode(0.55, 0.5, 48, 6, 2049)
    study = pg.measurement_convergence(code, (24, 48), basis="X", coefficients=(1, -1))
    expected, residual_second = reference_metrics(reference_density(code, (1, -1), "X"))
    assert study.rows[-1]["probabilities"] == pytest.approx(expected, abs=1e-5)
    assert study.rows[-1]["residual_second_moment"] == pytest.approx(residual_second, abs=1e-5)
    assert study.rows[-1]["fidelity_to_previous"] > 0.99
    assert study.rows[-1]["code_subspace_leakage"] < 1e-12
    for axis, values in (
        ("grid_points", (2049, 4097)),
        ("peaks", (4, 6)),
        ("peak_width", (0.5, 0.6)),
        ("envelope", (0.4, 0.5)),
    ):
        result = pg.measurement_convergence(code, values, axis=axis)
        assert len(result.rows) == 2


def test_invalid_decoder_and_study_requests():
    code = pg.GKPCode()
    for prior in ((0, 1), (0.3, 0.4), (-1, 2), (np.nan, 1), (1,)):
        with pytest.raises(ValueError):
            pg.SoftDecisionDecoder(code, prior=prior)
    with pytest.raises(NotImplementedError):
        pg.SoftDecisionDecoder(code, "Y")
    with pytest.raises(ValueError):
        pg.NearestCellDecoder().decode(np.inf)
    for axis, values in (("unknown", (1, 2)), ("cutoff", (24,)), ("cutoff", (48, 24))):
        with pytest.raises(ValueError):
            pg.measurement_convergence(code, values, axis=axis)
