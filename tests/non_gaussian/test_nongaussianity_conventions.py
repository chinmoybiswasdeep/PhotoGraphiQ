"""Pin hbar=2 non-Gaussianity conventions of fock_analysis against independent references.

WignerGrid.negative_volume is int max(-W, 0) dq dp, which is one half of the
Kenfack-Zyczkowski negativity delta_KZ = int |W| dq dp - 1 for a normalized W.
The relative-entropy non-Gaussianity S(rho_G) - S(rho) is computed here from
PhotoGraphiQ moments only (vacuum covariance = I, so nu = 2n + 1 for |n>); it is not
library API.
"""

import numpy as np
import pytest
import scipy.special as ssp

import photographiq as pg
from photographiq.fock_analysis import quadrature_moment, wigner

SQRT2 = np.sqrt(2)
# Closed forms with W_n = (-1)^n L_n(q^2 + p^2) e^{-(q^2 + p^2)/2} / (2 pi), dq dp = pi du.
NEGATIVE_VOLUME = {
    1: 2 * np.exp(-0.5) - 1,
    2: 0.5
    * ((8 + 4 * SQRT2) * np.exp(-(2 + SQRT2) / 2) - (8 - 4 * SQRT2) * np.exp(-(2 - SQRT2) / 2)),
}
RELATIVE_ENTROPY = {1: 2 * np.log(2), 2: 3 * np.log(3) - 2 * np.log(2)}
GRID = np.linspace(-8, 8, 401)


def _single_mode(state, cutoff=8):
    pattern = pg.Pattern(inputs=(0,))
    return pg.simulate(pattern, backend="piquasso-fock", cutoff=cutoff, inputs={0: state}).state


# Adapted from photonic-qrc-eoc (https://github.com/chinmoybiswasdeep/photonic-qrc-eoc,
# branch CV, commit 493f4c3f186ee314796964d6ddb6b2749171b7bb), MIT License, Copyright (c)
# 2026 Chinmoy Biswas; notebook 2c_adaptive_nongaussian_mbqrc_prl_preflight_colab, cell 49
# (wigner_grid, gauss_entropy).
def _laguerre_wigner(rho, xs, ps):
    """W(x, p) on an ij-indexed grid, hbar=2, from the Laguerre series."""
    X, P = np.meshgrid(xs, ps, indexing="ij")
    A = (X + 1j * P) / 2
    r2 = 4 * np.abs(A) ** 2
    W = np.zeros_like(X)
    for m in range(rho.shape[0]):
        W += rho[m, m].real * (-1) ** m * ssp.eval_genlaguerre(m, 0, r2)
        for n in range(m + 1, rho.shape[0]):
            W += 2 * np.real(
                rho[m, n]
                * (-1) ** m
                * (2 * np.conj(A)) ** (n - m)
                * np.sqrt(np.exp(ssp.gammaln(m + 1) - ssp.gammaln(n + 1)))
                * ssp.eval_genlaguerre(m, n - m, r2)
            )
    return W * np.exp(-r2 / 2) / (2 * np.pi)


def _gauss_entropy(nu):
    if nu <= 1 + 1e-12:
        return 0.0
    return (nu + 1) / 2 * np.log((nu + 1) / 2) - (nu - 1) / 2 * np.log((nu - 1) / 2)


def _relative_entropy_nongaussianity(state, node):
    mean = [quadrature_moment(state, node, 1, a) for a in (0, np.pi / 2, np.pi / 4)]
    var = [
        quadrature_moment(state, node, 2, a) - mu**2
        for a, mu in zip((0, np.pi / 2, np.pi / 4), mean)
    ]
    covariance_qp = var[2] - (var[0] + var[1]) / 2  # Var(q_pi/4) = (Vq + Vp)/2 + Cov(q, p)
    nu = np.sqrt(var[0] * var[1] - covariance_qp**2)
    assert nu > 1 - 1e-9  # nu < 1 would be a truncation artefact, not physics
    weights = np.linalg.eigvalsh(state.reduced((node,)).density_matrix)
    weights = weights[weights > 1e-14]
    return _gauss_entropy(nu) - float(-(weights * np.log(weights)).sum()), nu


def test_wigner_matches_laguerre_series_with_coherences():
    amplitudes = np.array([1, 0.5j, -0.3, 0.2 + 0.1j])
    state = _single_mode(pg.FockInput(tuple(amplitudes / np.linalg.norm(amplitudes))))
    rho = np.asarray(state.density_matrix)  # rho[m, n] = <m|rho|n>
    values = wigner(state, GRID, GRID).values  # shape (len(p), len(q))
    # The notebook series pairs rho[m, n] with (x - ip)^(n - m), so with rho[m, n] = <m|rho|n>
    # it returns W(x, -p); it equals PhotoGraphiQ's W only for conj(rho). The notebook used
    # it solely through int |W|, which is invariant under p -> -p.
    assert np.abs(values - _laguerre_wigner(rho.conj(), GRID, GRID).T).max() < 1e-14
    assert np.abs(values - _laguerre_wigner(rho, GRID, GRID).T).max() > 1e-2
    # Absolute orientation from an independent path: int (q, p) W = (<q>, <p>) from rho.
    area = (GRID[1] - GRID[0]) ** 2
    for axis, angle in ((GRID[None, :], 0.0), (GRID[:, None], np.pi / 2)):
        mean = quadrature_moment(state, 0, 1, angle)
        assert abs(mean) > 0.1
        assert (values * axis).sum() * area == pytest.approx(mean, abs=1e-10)


@pytest.mark.parametrize("n", [1, 2])
def test_negative_volume_is_half_kenfack_zyczkowski_delta(n):
    state = _single_mode(pg.FockInput.number(n))
    grid = wigner(state, GRID, GRID)
    delta_kz = np.abs(_laguerre_wigner(np.asarray(state.density_matrix), GRID, GRID))
    delta_kz = delta_kz.sum() * (GRID[1] - GRID[0]) ** 2 - 1
    # Kinked integrand: trapezoid error ~ h^2, observed 2e-5 (|1>), 3e-5 (|2>) here.
    assert grid.negative_volume == pytest.approx(NEGATIVE_VOLUME[n], abs=1e-4)
    assert delta_kz == pytest.approx(2 * NEGATIVE_VOLUME[n], abs=2e-4)
    assert grid.captured_mass == pytest.approx(1, abs=1e-10)


@pytest.mark.parametrize("n", [1, 2])
def test_fock_relative_entropy_nongaussianity_from_moments(n):
    value, nu = _relative_entropy_nongaussianity(_single_mode(pg.FockInput.number(n)), 0)
    assert nu == pytest.approx(2 * n + 1, abs=1e-12)
    assert value == pytest.approx(RELATIVE_ENTROPY[n], abs=1e-12)


def test_squeezed_vacuum_has_zero_relative_entropy_nongaussianity():
    pattern = pg.Pattern(inputs=()).extend([pg.Prepare("a", 0.4), pg.Output(("a",))]).validate()
    state = pg.simulate(pattern, backend="piquasso-fock", cutoff=60).state
    value, nu = _relative_entropy_nongaussianity(state, "a")
    assert nu == pytest.approx(1, abs=1e-8)
    assert abs(value) < 1e-8
