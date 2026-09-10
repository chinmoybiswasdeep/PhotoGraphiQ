from math import comb

import numpy as np
import pytest
from scipy.linalg import expm

import photographiq as pg
from photographiq.backends.mixed_fock import MixedFockBackend
from tests.v03_reference import basis, ladder, physical


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("eta", [0.0, 0.15, 0.6, 0.95, 1.0])
def test_binomial_channel_all_limits(n, eta):
    p = pg.Pattern(inputs=(0,)).append(pg.Loss(0, eta))
    result = pg.simulate(
        p, inputs={0: pg.FockInput.number(n)}, backend="piquasso-mixed-fock", cutoff=7
    )
    expected = [comb(n, k) * eta**k * (1 - eta) ** (n - k) if k <= n else 0 for k in range(7)]
    np.testing.assert_allclose(result.state.density_matrix, np.diag(expected), atol=2e-13)
    physical(result.state.density_matrix)
    assert result.state.photon_number(0) == pytest.approx(n * eta, abs=2e-13)
    assert min(result.state.retained_norms) == pytest.approx(1, abs=2e-13)


@pytest.mark.parametrize("eta", [0.0, 0.3, 1.0])
def test_coherent_attenuation(eta):
    alpha = 0.6 + 0.4j
    p = pg.Pattern(inputs=(0,)).append(pg.Loss(0, eta))
    actual = pg.simulate(
        p, inputs={0: pg.GaussianInput.coherent(alpha)}, backend="piquasso-mixed-fock", cutoff=18
    ).state
    expected = pg.simulate(
        pg.Pattern(inputs=(0,)),
        inputs={0: pg.GaussianInput.coherent(np.sqrt(eta) * alpha)},
        backend="piquasso-fock",
        cutoff=18,
    ).state
    assert actual.trace_distance(expected) < 2e-11
    physical(actual.density_matrix)


@pytest.mark.parametrize("eta", [0.0, 0.4, 1.0])
def test_thermal_loss_against_environment_dilation(eta):
    # Independent number-conserving system/environment beamsplitter, followed
    # by an explicit environment partial trace; no loss Kraus helpers.
    cutoff, nbar = 12, 0.12
    occupations = basis(2, cutoff)
    a, b = ladder(occupations, 0), ladder(occupations, 1)
    u = expm(np.arccos(np.sqrt(eta)) * (b.conj().T @ a - a.conj().T @ b))
    rho = np.zeros((len(occupations), len(occupations)), complex)
    local = np.array([[0.4, 0.1j], [-0.1j, 0.6]])
    # Environment tail is bounded by (nbar/(1+nbar))**(cutoff-1).
    for e in range(cutoff - 1):
        indices = [occupations.index((n, e)) for n in (0, 1)]
        rho[np.ix_(indices, indices)] = local * nbar**e / (1 + nbar) ** (e + 1)
    evolved = u @ rho @ u.conj().T
    reduced = np.zeros((cutoff, cutoff), complex)
    for i, (n, e) in enumerate(occupations):
        for j, (m, f) in enumerate(occupations):
            if e == f:
                reduced[n, m] += evolved[i, j]
    p = pg.Pattern(inputs=(0,)).append(pg.Loss(0, eta, nbar))
    result = pg.simulate(
        p,
        initial_state=pg.FockDensityMatrix(local, ((0,), (1,))),
        backend="piquasso-mixed-fock",
        cutoff=cutoff,
    )
    np.testing.assert_allclose(result.state.density_matrix, reduced, atol=2e-9, rtol=0)
    physical(result.state.density_matrix)
    assert result.state.photon_number(0) == pytest.approx(eta * 0.6 + (1 - eta) * nbar, abs=2e-8)


def test_thermal_tail_diagnostics_and_cutoff_convergence():
    p = pg.Pattern(inputs=(0,)).append(pg.Loss(0, 0.2, 1.0))
    rows = []
    for cutoff in (10, 18, 28):
        result = pg.simulate(p, backend="piquasso-mixed-fock", cutoff=cutoff)
        physical(result.state.density_matrix)
        rows.append((abs(result.state.photon_number(0) - 0.8), 1 - result.state.retained_norms[-1]))
    assert rows[-1][0] < 1e-7
    assert rows[-1][0] < rows[0][0] / 1000
    assert rows[0][1] > rows[-1][1] >= -1e-13
    with pytest.raises(ValueError, match="truncation"):
        pg.simulate(p, backend=MixedFockBackend(3))
