import numpy as np
import pytest
from scipy.special import gammaln

import photographiq as pg
from tests.fock.test_states_and_metrics import state


def cat_wigner(q, p, alpha, parity):
    x, y = np.meshgrid(q, p)
    displaced = np.exp(-((x - 2 * alpha.real) ** 2 + (y - 2 * alpha.imag) ** 2) / 2)
    displaced += np.exp(-((x + 2 * alpha.real) ** 2 + (y + 2 * alpha.imag) ** 2) / 2)
    interference = (
        2 * parity * np.exp(-(x * x + y * y) / 2) * np.cos(2 * (alpha.imag * x - alpha.real * y))
    )
    return (displaced + interference) / (4 * np.pi * (1 + parity * np.exp(-2 * abs(alpha) ** 2)))


@pytest.mark.parametrize("alpha", [0.3, 0.8, 1.5, 2.0, 0.8 + 0.4j])
@pytest.mark.parametrize("parity", [-1, 1])
def test_cat_phase_normalization_and_cutoff_convergence(alpha, parity):
    # Independently form coherent coefficients including exp(-|alpha|^2/2).
    n = np.arange(40)
    coefficients = np.exp(-(abs(alpha) ** 2) / 2 + n * np.log(complex(alpha)) - gammaln(n + 1) / 2)
    coefficients *= 1 + parity * (-1) ** n
    coefficients /= np.sqrt(2 * (1 + parity * np.exp(-2 * abs(alpha) ** 2)))
    finite = pg.FockInput.cat(alpha, 40, parity)
    np.testing.assert_allclose(finite.amplitudes, coefficients, atol=1e-13, rtol=1e-12)
    outputs = [state(pg.FockInput.cat(alpha, c, parity), c) for c in (8, 16, 32)]
    errors = [1 - outputs[i].fidelity(outputs[-1]) for i in (0, 1)]
    # At small alpha the tail is already below roundoff; do not demand strict
    # monotonicity of numbers indistinguishable from zero.
    assert errors[1] <= errors[0] + 1e-14
    assert errors[1] < 1e-5  # projected cat tail at c=16, worst case |alpha|=2
    assert np.isclose(outputs[-1].parity(), parity)
    assert abs(outputs[-1].norm - 1) < 1e-13


@pytest.mark.parametrize(
    "alpha,parity", [(0, 1), (0.8, 1), (0.8, -1), (0.8 + 0.4j, 1), (0.8 + 0.4j, -1)]
)
def test_wigner_analytic_cats_parity_and_grid_convergence(alpha, parity):
    output = state(pg.CatResource(alpha, parity), 32)
    for points in (81, 161):
        axis = np.linspace(-7, 7, points)
        grid = output.wigner(axis, axis)
        expected = cat_wigner(axis, axis, complex(alpha), parity)
        # Analytic interference independent of the Laguerre implementation;
        # c=32 resolves these cats so finite-Fock error is below 1e-10.
        np.testing.assert_allclose(grid.values, expected, atol=1e-10, rtol=1e-9)
        assert abs(grid.captured_mass - 1) < 1e-6
        assert abs(grid.values[points // 2, points // 2] - output.parity() / (2 * np.pi)) < 1e-13
        if alpha == 0:
            assert grid.values.min() >= 0
    if parity == -1:
        assert grid.values[80, 80] < 0


def test_wigner_window_and_negative_volume_refinement():
    output = state(pg.FockInput.number(1), 6)
    exact = 2 * np.exp(-0.5) - 1
    errors = []
    for points in (61, 121, 241):
        axis = np.linspace(-6, 6, points)
        errors.append(abs(output.wigner(axis, axis).negative_volume - exact))
    assert errors[-1] < errors[0]
    assert errors[-1] < 1e-4  # trapezoidal error at the circular Wigner zero contour
    masses = []
    for bound in (2, 4, 7):
        axis = np.linspace(-bound, bound, 161)
        masses.append(abs(1 - output.wigner(axis, axis).captured_mass))
    assert masses[-1] < masses[0]
    assert masses[-1] < 1e-8


def test_zero_limits_and_large_cat_truncation():
    for alpha in (1e-12, 1e-200):
        assert abs(pg.FockInput.cat(alpha, 8, 1).amplitudes[0]) > 1 - 1e-13
        assert abs(pg.FockInput.cat(alpha, 8, -1).amplitudes[1]) > 1 - 1e-13
    with pytest.raises(ValueError):
        pg.FockInput.cat(0, 8, -1)
    with pytest.raises(ValueError, match="truncation"):
        state(pg.CatResource(4), 8)
    assert np.isclose(state(pg.CatResource(4), 64).parity(), 1)
