"""Native Gaussian diagnostics and batched kernels against closed forms and NumPy references."""

import numpy as np
import pytest
import scipy.linalg

import photographiq as pg
import photographiq.native
from photographiq.backends.gaussian import GaussianBackend
from photographiq.gaussian import beamsplitter, cz, omega, rotation, squeezing

ng = pytest.importorskip(
    "photographiq.backends.native_gaussian",
    reason="native 'gaussian' component not built",
    exc_type=ImportError,
)


def g(nu):
    nu = np.asarray(nu, dtype=float)
    a, b = (nu + 1) / 2, (nu - 1) / 2
    return np.sum(a * np.log(a) - np.where(b > 0, b * np.log(np.where(b > 0, b, 1)), 0))


def williamson_state(rng, n, low=1.0, high=2.5):
    """Random state S diag(nu,nu) S^T with known symplectic spectrum nu.

    Rounding S D S^T perturbs the spectrum by ~eps ||S||^2 (||S^-1|| = ||S|| for symplectic
    S); observed errors of both NumPy and native solvers reach ~22 eps ||S||^2 at 2n = 14,
    so tolerances use 20 n eps ||S||^2.
    """
    s = np.eye(2 * n)
    for _ in range(3):
        h = rng.normal(size=(2 * n, 2 * n))
        s = scipy.linalg.expm(omega(n) @ (h + h.T) * 0.3) @ s
    nu = np.sort(rng.uniform(low, high, n))
    cov = s @ np.diag(np.repeat(nu, 2)) @ s.T
    state = pg.GaussianState(rng.normal(size=2 * n), (cov + cov.T) / 2, tuple(range(n)))
    return state, nu, 20 * n * np.finfo(float).eps * np.linalg.norm(s, 2) ** 2


def tmsv(r):
    cov = beamsplitter(np.pi / 4) @ np.diag(np.exp([-2 * r, 2 * r, 2 * r, -2 * r]))
    cov = cov @ beamsplitter(np.pi / 4).T
    return pg.GaussianState(np.zeros(4), (cov + cov.T) / 2, ("a", "b"))


@pytest.mark.parametrize("n", [1, 2, 4, 7])
def test_spectral_diagnostics_match_williamson_and_numpy(n):
    rng = np.random.default_rng(n)
    for _ in range(4):
        state, nu, tol = williamson_state(rng, n)
        native_nu = ng.symplectic_eigenvalues(state)
        np.testing.assert_allclose(native_nu, nu, rtol=tol)
        np.testing.assert_allclose(native_nu, state.symplectic_eigenvalues(), rtol=tol)
        assert np.isclose(ng.purity(state), np.prod(1 / nu), rtol=tol)
        assert np.isclose(ng.entropy(state), g(nu), rtol=tol)
        assert np.isclose(ng.entropy(state), state.entropy(), rtol=tol)
        cut = tuple(range(n // 2 + 1))
        negativity = ng.log_negativity(state, cut)
        assert np.isclose(negativity, state.log_negativity(cut), atol=tol, rtol=tol)


@pytest.mark.parametrize("r", [0.1, 0.8, 2.0])
def test_tmsv_entropy_and_negativity_closed_forms(r):
    state = tmsv(r)
    assert np.isclose(ng.log_negativity(state, ("a",)), 2 * r / np.log(2), rtol=1e-12)
    assert np.isclose(ng.entropy(state.reduced(("b",))), g(np.cosh(2 * r)), rtol=1e-11)
    assert ng.entropy(state) < 1e-7  # pure global state
    assert np.isclose(ng.purity(state), 1, rtol=1e-12)


def test_thermal_entropy_and_unphysical_rejection():
    for nbar in (0.0, 0.3, 4.0):
        thermal = pg.GaussianState(np.zeros(2), (2 * nbar + 1) * np.eye(2), (0,))
        expected = (nbar + 1) * np.log(nbar + 1) - (nbar * np.log(nbar) if nbar else 0)
        assert np.isclose(ng.entropy(thermal), expected, rtol=1e-13, atol=1e-15)
    unphysical = pg.GaussianState.__new__(pg.GaussianState)  # bypass validation on purpose
    unphysical.mean, unphysical.covariance, unphysical.nodes = np.zeros(2), 0.5 * np.eye(2), (0,)
    with pytest.raises(ValueError, match="below 1"):
        ng.entropy(unphysical)


def test_fidelity_closed_forms():
    vacuum = pg.GaussianState(np.zeros(2), np.eye(2), ("a",))
    alpha = 0.6 + 0.3j
    assert np.isclose(
        ng.fidelity(vacuum, pg.GaussianInput.coherent(alpha).state("a")),
        np.exp(-(abs(alpha) ** 2)),
        rtol=1e-14,
    )
    assert np.isclose(
        ng.fidelity(vacuum, pg.GaussianInput.squeezed(0.7).state("a")), 1 / np.cosh(0.7)
    )
    for n1, n2 in ((0.4, 1.7), (0.01, 3.0), (2.0, 2.0)):
        t1 = pg.GaussianState(np.zeros(2), (2 * n1 + 1) * np.eye(2), ("a",))
        t2 = pg.GaussianState(np.zeros(2), (2 * n2 + 1) * np.eye(2), ("a",))
        expected = (np.sqrt((n1 + 1) * (n2 + 1)) - np.sqrt(n1 * n2)) ** -2
        assert np.isclose(ng.fidelity(t1, t2), expected, rtol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_fidelity_matches_numpy_states_and_symmetry(n):
    rng = np.random.default_rng(10 + n)
    for _ in range(5):
        a, _, _ = williamson_state(rng, n)
        b, _, _ = williamson_state(rng, n)
        value = ng.fidelity(a, b)
        assert 0 < value < 1
        assert np.isclose(value, a.fidelity(b), rtol=1e-10)
        assert np.isclose(value, ng.fidelity(b, a), rtol=1e-9)
        assert np.isclose(ng.fidelity(a, a), 1, rtol=1e-9)
    # Label reordering of the second state.
    if n > 1:
        perm = tuple(reversed(b.nodes))
        assert np.isclose(ng.fidelity(a, b.reduced(perm)), ng.fidelity(a, b), rtol=1e-13)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_fidelity_matches_piquasso(n):
    pq = pytest.importorskip("piquasso")

    def to_pq(state):
        result = pq.GaussianState(d=n, connector=pq.NumpyConnector(), config=pq.Config(hbar=2.0))
        result.xpxp_mean_vector = state.mean.copy()
        result.xpxp_covariance_matrix = 2 * state.covariance
        return result

    rng = np.random.default_rng(30 + n)
    for _ in range(5):
        a, _, _ = williamson_state(rng, n)
        b, _, _ = williamson_state(rng, n)
        assert np.isclose(ng.fidelity(a, b), to_pq(a).fidelity(to_pq(b)), rtol=1e-9)


def test_batch_symplectic_matches_dense_reference():
    rng = np.random.default_rng(5)
    batch, n = 7, 4
    states = [williamson_state(rng, n)[0] for _ in range(batch)]
    means = np.array([s.mean for s in states])
    covs = np.array([s.covariance for s in states])
    mats = np.array([beamsplitter(t) @ cz(0.4) for t in rng.uniform(-3, 3, batch)])
    new_means, new_covs = ng.batch_apply_symplectic(means, covs, [3, 1], mats)
    assert np.array_equal(means, [s.mean for s in states])  # inputs untouched
    for b, state in enumerate(states):
        ref = GaussianBackend()
        ref.set_state(state)
        ref.transform((3, 1), mats[b])
        np.testing.assert_allclose(new_means[b], ref.state.mean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(new_covs[b], ref.state.covariance, rtol=1e-12, atol=1e-12)
    # One shared single-mode matrix broadcast to the batch.
    shared = rotation(0.3) @ squeezing(0.2)
    _, covs1 = ng.batch_apply_symplectic(means, covs, [0], shared)
    for b, state in enumerate(states):
        ref = GaussianBackend()
        ref.set_state(state)
        ref.transform((0,), shared)
        np.testing.assert_allclose(covs1[b], ref.state.covariance, rtol=1e-12, atol=1e-12)
    bad = mats.copy()
    bad[2, 0, 0] += 0.5
    with pytest.raises(ValueError, match="symplectic"):
        ng.batch_apply_symplectic(means, covs, [3, 1], bad)


@pytest.mark.parametrize("eta,noise", [(1.0, 0.0), (0.8, 0.1)])
def test_batch_homodyne_matches_numpy_backend(eta, noise):
    rng = np.random.default_rng(6)
    batch, n = 6, 3
    states = [williamson_state(rng, n)[0] for _ in range(batch)]
    means = np.array([s.mean for s in states])
    covs = np.array([s.covariance for s in states])
    angles = rng.uniform(-3, 3, batch)
    outcomes = rng.normal(size=batch)
    expected, variance = ng.batch_homodyne_moments(means, covs, 1, angles, eta, noise)
    out_means, out_covs, densities = ng.batch_condition_homodyne(
        means, covs, 1, angles, outcomes, eta, noise
    )
    for b, state in enumerate(states):
        mu, var = state.quadrature(1, angles[b])
        assert np.isclose(expected[b], mu, rtol=1e-13, atol=1e-13)
        assert np.isclose(variance[b], var + (1 - eta) / eta + noise, rtol=1e-13)
        ref = GaussianBackend()
        ref.set_state(state)
        ref.measure(1, pg.Homodyne(angles[b], eta, noise), angles[b], outcome=outcomes[b])
        np.testing.assert_allclose(out_means[b], ref.state.mean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(out_covs[b], ref.state.covariance, rtol=1e-12, atol=1e-12)
        assert np.isclose(densities[b], ref.last_measurement["value"], rtol=1e-12)


def test_channel_kernel_matches_dense_algebra_and_rejects_non_cp():
    kernels = pg.native.require("gaussian")
    rng = np.random.default_rng(8)
    state, _, _ = williamson_state(rng, 3)
    eta, nth = 0.7, 0.4
    x = np.sqrt(eta) * rotation(0.3)  # lossy rotation
    y = (1 - eta) * (2 * nth + 1) * np.eye(2)
    mean, cov = state.mean.copy(), np.asfortranarray(state.covariance)
    kernels.apply_channel(mean, cov, [1], x, y)
    full_x, full_y = np.eye(6), np.zeros((6, 6))
    full_x[2:4, 2:4], full_y[2:4, 2:4] = x, y
    np.testing.assert_allclose(mean, full_x @ state.mean, rtol=1e-13, atol=1e-13)
    expected = full_x @ state.covariance @ full_x.T + full_y
    np.testing.assert_allclose(cov, expected, rtol=1e-12, atol=1e-12)
    # Same as the dedicated loss kernel when X = sqrt(eta) I.
    m1, c1 = state.mean.copy(), np.asfortranarray(state.covariance)
    m2, c2 = state.mean.copy(), np.asfortranarray(state.covariance)
    kernels.apply_channel(m1, c1, [2], np.sqrt(eta) * np.eye(2), y)
    kernels.apply_loss(m2, c2, 2, eta, nth)
    np.testing.assert_array_equal(c1, c2)
    with pytest.raises(ValueError, match="completely positive"):
        kernels.apply_channel(mean, cov, [0], 2 * np.eye(2), np.zeros((2, 2)))
    with pytest.raises(TypeError):  # C-ordered covariance cannot be updated in place
        kernels.apply_loss(mean, np.ascontiguousarray(np.eye(6) + 0.1 * np.ones((6, 6))), 0, 0.5)


def test_batch_kernels_use_the_symmetric_part():
    """sym(S V S^T) = S sym(V) S^T: an antisymmetric perturbation must not change results."""
    rng = np.random.default_rng(9)
    states = [williamson_state(rng, 3)[0] for _ in range(4)]
    means = np.array([s.mean for s in states])
    covs = np.array([s.covariance for s in states])
    h = rng.normal(size=covs.shape)
    skewed = covs + 0.3 * (h - h.swapaxes(1, 2))
    s = beamsplitter(0.4) @ cz(0.3)
    for a, b in zip(
        ng.batch_apply_symplectic(means, covs, [0, 2], s),
        ng.batch_apply_symplectic(means, skewed, [0, 2], s),
        strict=True,
    ):
        np.testing.assert_allclose(b, a, rtol=1e-13, atol=1e-13)
    for a, b in zip(
        ng.batch_condition_homodyne(means, covs, 1, 0.3, 0.2, 0.9, 0.0),
        ng.batch_condition_homodyne(means, skewed, 1, 0.3, 0.2, 0.9, 0.0),
        strict=True,
    ):
        np.testing.assert_allclose(b, a, rtol=1e-13, atol=1e-13)
