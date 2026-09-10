import numpy as np
import pytest
from scipy.integrate import simpson

import photographiq as pg
from tests.v03_reference import hermites


def comb_inner(left, right, dq=0.0, dp=0.0):
    """Closed Gaussian integrals for equal-width finite combs and Weyl shifts."""
    assert left.peak_width == right.peak_width
    width = left.peak_width
    x = pg.gkp.SPACING * (2 * np.arange(-left.peaks, left.peaks + 1) + left.logical)
    y = pg.gkp.SPACING * (2 * np.arange(-right.peaks, right.peaks + 1) + right.logical)
    w = (
        np.exp(-(left.envelope**2) * x * x / 4)[:, None]
        * np.exp(-(right.envelope**2) * y * y / 4)[None, :]
    )
    return (
        np.sqrt(2 * np.pi)
        * width
        * np.sum(
            w
            * np.exp(
                -((x[:, None] - y[None, :] - dq) ** 2) / (8 * width**2)
                - width**2 * dp**2 / 8
                + 1j * dp * (x[:, None] + y[None, :]) / 4
            )
        )
    )


def overlap(left, right, dq=0.0, dp=0.0):
    return comb_inner(left, right, dq, dp) / np.sqrt(
        comb_inner(left, left) * comb_inner(right, right)
    )


@pytest.mark.parametrize("width,envelope", [(0.4, 0.4), (0.55, 0.5), (0.7, 0.35)])
def test_grid_peak_and_cutoff_are_independent(width, envelope):
    options = dict(peak_width=width, envelope=envelope)
    grid_states = [
        pg.GKPResource(0, **options, peaks=5, grid_points=n).project(64) for n in (2049, 4097)
    ]
    np.testing.assert_allclose(
        grid_states[0][0].amplitudes, grid_states[1][0].amplitudes, atol=2e-10
    )
    assert abs(grid_states[0][1] - grid_states[1][1]) < 2e-11
    # Peak count varied with a fixed, sufficiently resolved grid and cutoff.
    peaks = [pg.GKPResource(0, **options, peaks=n, grid_points=4097) for n in (2, 4, 6)]
    errors = [1 - abs(overlap(r, peaks[-1])) ** 2 for r in peaks[:-1]]
    assert errors[-1] <= errors[0] + 1e-13
    assert errors[-1] < 1e-10
    resource = peaks[-1]
    masses = [resource.project(c)[1] for c in (16, 32, 64)]
    assert masses[0] <= masses[1] + 1e-13 <= masses[2] + 2e-13
    assert masses[-1] > 0.9999
    q, psi = resource.wavefunction()
    assert simpson(abs(psi) ** 2, x=q) == pytest.approx(1, abs=2e-13)
    independent = simpson(hermites(q, 64) * psi[:, None], x=q, axis=0)
    state, mass = resource.project(64)
    np.testing.assert_allclose(
        np.asarray(state.amplitudes) * np.sqrt(mass), independent, atol=2e-12
    )


@pytest.mark.parametrize("width,envelope", [(0.4, 0.4), (0.6, 0.5), (0.8, 0.4)])
def test_finite_logical_overlap_matches_closed_gaussian_integral(width, envelope):
    zero = pg.GKPResource(0, width, envelope, 6, 4097)
    one = pg.GKPResource(1, width, envelope, 6, 4097)
    expected = abs(overlap(zero, one))
    errors = []
    for cutoff in (24, 48, 80):
        a, b = zero.fock(cutoff), one.fock(cutoff)
        errors.append(abs(abs(np.vdot(a.amplitudes, b.amplitudes)) - expected))
    assert errors[-1] < 2e-7
    assert errors[-1] <= errors[0] + 1e-12
    assert expected > 0  # finite codewords are not orthogonal by assumption
