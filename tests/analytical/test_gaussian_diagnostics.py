import numpy as np
import pytest
import scipy.linalg

import photographiq as pg
from photographiq.gaussian import beamsplitter, omega, rotation, squeezing

pq = pytest.importorskip("piquasso")  # reference simulator: optional extra


def oracle_fidelity(m1, G1, m2, G2):
    # Adapted from photonic-qrc-eoc (https://github.com/chinmoybiswasdeep/photonic-qrc-eoc,
    # branch CV, commit 493f4c3f186ee314796964d6ddb6b2749171b7bb), MIT License,
    # Copyright (c) 2026 Chinmoy Biswas; notebook 0b_cv_mbqrc_photographiqml_colab, cell 16.
    # Banchi-Braunstein-Pirandola, PRL 115, 260501 (2015), converted from hbar=2 (V = G/2).
    n = len(m1) // 2
    Om = np.kron(np.eye(n), np.array([[0, 1], [-1, 0]]))
    V1, V2 = G1 / 2, G2 / 2
    d = (np.asarray(m2) - np.asarray(m1)) / np.sqrt(2)
    S = V1 + V2
    Vaux = Om.T @ np.linalg.solve(S, Om / 4 + V2 @ Om @ V1)
    M = np.linalg.inv(Vaux @ Om)
    inner = scipy.linalg.sqrtm(np.eye(2 * n) + M @ M / 4)
    Ftot = np.linalg.det(2 * (inner + np.eye(2 * n)) @ Vaux)
    root_F = (
        np.real(Ftot) ** 0.25 / np.linalg.det(S) ** 0.25 * np.exp(-0.25 * d @ np.linalg.solve(S, d))
    )
    return root_F**2


def random_mixed_state(rng, n):
    s = np.eye(2 * n)
    for _ in range(3):
        h = rng.normal(size=(2 * n, 2 * n))
        s = scipy.linalg.expm(omega(n) @ (h + h.T) * 0.3) @ s
    nu = rng.uniform(1, 2.5, n)
    cov = s @ np.diag(np.repeat(nu, 2)) @ s.T
    return pg.GaussianState(rng.normal(size=2 * n), (cov + cov.T) / 2, tuple(range(n))), nu


def g(nu):
    return (nu + 1) / 2 * np.log((nu + 1) / 2) - (nu - 1) / 2 * np.log((nu - 1) / 2)


def tmsv(r):
    pattern = pg.Pattern()
    pattern.append(pg.Prepare(0, state=pg.GaussianInput.squeezed(r)))
    pattern.append(pg.Prepare(1, state=pg.GaussianInput.squeezed(-r)))
    pattern.append(pg.BeamSplitter(0, 1, np.pi / 4))
    pattern.append(pg.Output((0, 1)))
    return pg.simulate(pattern, backend="gaussian", seed=0).state


@pytest.mark.parametrize("n", [1, 2])
def test_fidelity_matches_bbp_oracle_for_mixed_states(n):
    rng = np.random.default_rng(n)
    for _ in range(5):
        a, _ = random_mixed_state(rng, n)
        b, _ = random_mixed_state(rng, n)
        expected = oracle_fidelity(a.mean, a.covariance, b.mean, b.covariance)
        assert np.isclose(a.fidelity(b), expected, atol=1e-10, rtol=0)
        assert np.isclose(b.fidelity(a), expected, atol=1e-10, rtol=0)


def test_fidelity_closed_forms():
    vacuum = pg.GaussianState(np.zeros(2), np.eye(2), ("a",))
    alpha = 0.6 + 0.3j
    coherent = pg.GaussianInput.coherent(alpha).state("a")
    assert np.isclose(vacuum.fidelity(coherent), np.exp(-(abs(alpha) ** 2)), atol=1e-12)
    r = 0.7
    squeezed = pg.GaussianInput.squeezed(r).state("a")
    assert np.isclose(vacuum.fidelity(squeezed), 1 / np.cosh(r), atol=1e-12)
    # Mixed-mixed thermal states: F = (sqrt((n1+1)(n2+1)) - sqrt(n1 n2))^-2.
    n1, n2 = 0.4, 1.7
    t1 = pg.GaussianState(np.zeros(2), (2 * n1 + 1) * np.eye(2), ("a",))
    t2 = pg.GaussianState(np.zeros(2), (2 * n2 + 1) * np.eye(2), ("a",))
    expected = (np.sqrt((n1 + 1) * (n2 + 1)) - np.sqrt(n1 * n2)) ** -2
    assert np.isclose(t1.fidelity(t2), expected, atol=1e-12)
    assert not np.isclose(t1.overlap(t2), expected)
    assert np.isclose(t1.fidelity(t1), 1, atol=1e-12)


def test_fidelity_equals_overlap_for_pure_states_and_reorders_nodes():
    rng = np.random.default_rng(5)
    s1 = beamsplitter(0.4) @ scipy.linalg.block_diag(squeezing(0.3), squeezing(-0.5))
    s2 = beamsplitter(-0.9) @ scipy.linalg.block_diag(squeezing(0.6), squeezing(0.2))
    a = pg.GaussianState(rng.normal(size=4), s1 @ s1.T, ("x", "y"))
    b = pg.GaussianState(rng.normal(size=4), s2 @ s2.T, ("x", "y"))
    assert np.isclose(a.fidelity(b), a.overlap(b), atol=1e-12, rtol=0)
    assert np.isclose(a.fidelity(b.reduced(("y", "x"))), a.fidelity(b), atol=1e-12)
    with pytest.raises(ValueError):
        a.fidelity(pg.GaussianState(np.zeros(4), np.eye(4), ("x", "z")))


def test_two_mode_squeezed_vacuum_diagnostics():
    r = 0.7
    state = tmsv(r)
    nbar = np.sinh(r) ** 2
    np.testing.assert_allclose(state.symplectic_eigenvalues(), [1, 1], atol=1e-10)
    assert abs(state.entropy()) < 1e-9
    assert np.isclose(state.purity(), 1, atol=1e-10)
    reduced = state.reduced((0,))
    expected = (nbar + 1) * np.log(nbar + 1) - nbar * np.log(nbar)
    assert np.isclose(reduced.entropy(), expected, atol=1e-10)
    assert np.isclose(state.log_negativity((0,)), 2 * r / np.log(2), atol=1e-10)
    assert np.isclose(state.log_negativity((1,)), 2 * r / np.log(2), atol=1e-10)
    native = pq.GaussianState(d=1, connector=pq.NumpyConnector(), config=pq.Config(hbar=2.0))
    native.xpxp_mean_vector = reduced.mean
    native.xpxp_covariance_matrix = 2 * reduced.covariance
    assert np.isclose(reduced.purity(), float(native.get_purity()), atol=1e-12)
    assert np.isclose(reduced.purity(), 1 / (2 * nbar + 1), atol=1e-12)


def test_thermal_entropy_closed_form_and_symplectic_invariance():
    rng = np.random.default_rng(2)
    state, nu = random_mixed_state(rng, 2)
    np.testing.assert_allclose(state.symplectic_eigenvalues(), np.sort(nu), atol=1e-9)
    assert np.isclose(state.entropy(), g(nu).sum(), atol=1e-9)
    assert np.isclose(state.purity(), 1 / np.prod(nu), atol=1e-9)
    nbar = 0.8
    thermal = pg.GaussianState(np.zeros(2), (2 * nbar + 1) * np.eye(2), (0,))
    expected = (nbar + 1) * np.log(nbar + 1) - nbar * np.log(nbar)
    assert np.isclose(thermal.entropy(), expected, atol=1e-12)
    vacuum = pg.GaussianState(np.zeros(4), np.eye(4), (0, 1))
    assert vacuum.entropy() == 0
    assert vacuum.log_negativity((0,)) == pytest.approx(0, abs=1e-12)


def test_product_state_has_no_log_negativity():
    rng = np.random.default_rng(3)
    a, _ = random_mixed_state(rng, 1)
    cov = scipy.linalg.block_diag(a.covariance, squeezing(0.9) @ squeezing(0.9).T)
    assert pg.GaussianState(np.zeros(4), cov, (0, 1)).log_negativity((0,)) == pytest.approx(
        0, abs=1e-10
    )


def test_unphysical_and_invalid_inputs_raise():
    with pytest.raises(ValueError):
        pg.GaussianState(np.zeros(2), 0.5 * np.eye(2), (0,))
    state = pg.GaussianState(np.zeros(2), np.eye(2), (0,))
    state.covariance = 0.5 * np.eye(2)  # bypass construction-time validation
    with pytest.raises(ValueError):
        state.entropy()
    with pytest.raises(ValueError):
        tmsv(0.3).log_negativity(("missing",))
    with pytest.raises(ValueError):
        tmsv(0.3).log_negativity((0, 0))


def test_fidelity_is_exact_for_strongly_squeezed_pure_states():
    # The general mixed-state formula loses ~1e-6 accuracy here (adversarial review finding);
    # pure states must use the exact identity F = Tr(rho sigma) <= 1.
    # r <= 3 (~26 dB): beyond that a rotated float64 covariance itself no longer resolves
    # purity (smallest-eigenvalue relative error ~eps*exp(4r)), so no tolerance is claimed.
    for r in (1.0, 2.0, 3.0):
        s = rotation(0.3) @ squeezing(r)
        a = pg.GaussianState(np.zeros(2), s @ s.T, ("a",))
        assert abs(a.fidelity(a) - 1) < 1e-9
        mixed = pg.GaussianState(np.zeros(2), 3 * np.eye(2), ("a",))
        assert a.fidelity(mixed) <= 1 and np.isclose(
            a.fidelity(mixed), mixed.fidelity(a), rtol=1e-9
        )
