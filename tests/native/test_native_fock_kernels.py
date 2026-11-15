"""Native Fock kernels against closed forms and defining properties (hbar = 2).

Tolerances: exact closed forms are compared at rounding level (1e-13 or tighter).
Heisenberg checks use input states with at most 3 photons and cutoffs where the
retained norm differs from one by < 1e-13, so truncation stays below the tolerance.
"""

import numpy as np
import pytest
from scipy.special import eval_hermite, factorial

from photographiq import native

try:
    F = native.require("fock")
except ImportError as exc:  # pragma: no cover - exercised only without the extension
    pytest.skip(f"native fock component unavailable: {exc}", allow_module_level=True)


def _rotation(phi):
    return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


def _ladder(modes, cutoff, mode):
    """Annihilation operator on the truncated space, built from occupations only."""
    basis = F.basis(modes, cutoff)
    op = np.zeros((len(basis), len(basis)))
    for j, occ in enumerate(basis):
        if occ[mode]:
            lower = occ.copy()
            lower[mode] -= 1
            op[F.rank(lower[None, :], cutoff)[0], j] = np.sqrt(occ[mode])
    return op


@pytest.mark.parametrize("modes,cutoff", [(1, 5), (2, 6), (3, 5), (4, 4)])
def test_basis_order_and_ranking(modes, cutoff):
    basis = F.basis(modes, cutoff)
    assert len(basis) == F.dimension(modes, cutoff)
    np.testing.assert_array_equal(F.rank(basis, cutoff), np.arange(len(basis)))
    assert (basis.sum(axis=1)[1:] >= basis.sum(axis=1)[:-1]).all()
    fock = pytest.importorskip("piquasso._math.fock")
    np.testing.assert_array_equal(basis, fock.get_fock_space_basis(modes, cutoff))
    with pytest.raises(ValueError):
        F.rank(np.full((1, modes), cutoff, dtype=np.int32), cutoff)


def test_coherent_and_squeezed_vacuum_closed_forms():
    alpha, cutoff, r = 0.8 - 0.3j, 30, 0.55
    n = np.arange(cutoff)
    column = F.gaussian_unitary(np.eye(2), [2 * alpha.real, 2 * alpha.imag], cutoff)[:, 0]
    exact = np.exp(-(abs(alpha) ** 2) / 2) * alpha**n / np.sqrt(factorial(n))
    np.testing.assert_allclose(column, exact, atol=1e-15, rtol=0)
    column = F.gaussian_unitary(np.diag([np.exp(-r), np.exp(r)]), np.zeros(2), cutoff)[:, 0]
    k = n[::2] // 2
    even = np.sqrt(factorial(2 * k)) / (2.0**k * factorial(k)) * (-np.tanh(r)) ** k
    np.testing.assert_allclose(column[::2], even / np.sqrt(np.cosh(r)), atol=1e-15, rtol=0)
    np.testing.assert_allclose(column[1::2], 0, atol=1e-15)


def test_passive_gates_are_exact_and_unitary_per_sector():
    cutoff = 14
    rotation = F.gaussian_unitary(_rotation(0.37), np.zeros(2), cutoff)
    np.testing.assert_allclose(rotation, np.diag(np.exp(0.37j * np.arange(cutoff))), atol=1e-14)
    c, s = np.cos(0.7), np.sin(0.7)
    u = F.gaussian_unitary(np.kron([[c, -s], [s, c]], np.eye(2)), np.zeros(4), cutoff)
    np.testing.assert_allclose(u.conj().T @ u, np.eye(len(u)), atol=1e-13)
    totals = F.basis(2, cutoff).sum(axis=1)
    assert np.abs(u[totals[:, None] != totals[None, :]]).max() == 0.0


def test_hong_ou_mandel():
    u = F.gaussian_unitary(np.kron(_rotation(np.pi / 4), np.eye(2)), np.zeros(4), 4)
    i11, i20, i02 = F.rank(np.array([[1, 1], [2, 0], [0, 2]], dtype=np.int32), 4)
    assert abs(u[i11, i11]) < 1e-15
    assert abs(u[i20, i11]) ** 2 == pytest.approx(0.5, abs=1e-15)
    assert abs(u[i02, i11]) ** 2 == pytest.approx(0.5, abs=1e-15)


def test_projected_gaussian_gates_are_contractions():
    # P U P with U unitary has singular values <= 1; strong displacement at small cutoff.
    u = F.gaussian_unitary(np.eye(2), [3.0, -2.0], 10)
    assert np.linalg.svd(u, compute_uv=False).max() <= 1 + 1e-13


def test_gaussian_matrix_elements_are_numerically_stable_at_large_cutoff():
    """Regression: the bra-first Bargmann pivot amplified rounding to O(1) at cutoffs >= 60.

    Before the fix: D(|alpha| = 4) at cutoff 60 was off by 58 and S(r = 0.5) at cutoff 80
    by 1.4e-7; |<m|S(1)|n>| reached 5e4 at cutoff 150. References are expm of the exact
    generators in a 500/700-dimensional space (their own error is ~1e-15 here: the
    columns kept have negligible weight beyond 400).
    """
    from scipy.linalg import expm

    big = 700
    a = np.diag(np.sqrt(np.arange(1, big)), 1).astype(complex)
    alpha = 4 * np.exp(0.7j)
    ref = expm(alpha * a.conj().T - np.conj(alpha) * a)[:60, :60]
    disp = F.gaussian_unitary(np.eye(2), [2 * alpha.real, 2 * alpha.imag], 60)
    np.testing.assert_allclose(disp, ref, atol=1e-12, rtol=0)
    # Same gate through the Bargmann path (rotation(2 pi) != 1 in floating point).
    bargmann = F.gaussian_unitary(_rotation(2 * np.pi), [2 * alpha.real, 2 * alpha.imag], 60)
    np.testing.assert_allclose(bargmann, ref, atol=1e-6, rtol=0)  # observed 1.4e-7
    r = 0.5
    ref = expm(r / 2 * (a @ a - a.conj().T @ a.conj().T))[:80, :80]
    sq = F.gaussian_unitary(np.diag([np.exp(-r), np.exp(r)]), np.zeros(2), 80)
    np.testing.assert_allclose(sq, ref, atol=1e-12, rtol=0)  # observed 8e-14
    strong = F.gaussian_unitary(np.diag([np.exp(-1.0), np.exp(1.0)]), np.zeros(2), 150)
    assert np.abs(strong).max() <= 1 + 1e-12


@pytest.mark.parametrize("modes", [1, 2])
def test_heisenberg_action_matches_symplectic(modes):
    """<U^dag x U> = S <x> + d and Cov' = S Cov S^T, quadratures from ladder operators."""
    rng = np.random.default_rng(5)
    cutoff = 60 if modes == 1 else 44
    if modes == 1:
        symplectic = _rotation(0.4) @ np.diag([np.exp(-0.3), np.exp(0.3)]) @ [[1, 0], [0.5, 1]]
    else:
        cz = np.eye(4)
        cz[1, 2] = cz[3, 0] = 0.6
        bs = np.kron(_rotation(0.3), np.eye(2))
        sq = np.diag([np.exp(-0.2), np.exp(0.2), np.exp(0.1), np.exp(-0.1)])
        symplectic = bs @ cz @ sq
    shift = rng.normal(scale=0.4, size=2 * modes)
    u = F.gaussian_unitary(symplectic, shift, cutoff)
    basis = F.basis(modes, cutoff)
    psi = np.zeros(len(basis), complex)
    low = basis.sum(axis=1) <= 3
    psi[low] = rng.normal(size=low.sum()) + 1j * rng.normal(size=low.sum())
    psi /= np.linalg.norm(psi)
    phi = psi.copy()
    F.apply_pure(phi, modes, cutoff, list(range(modes)), u)
    assert np.vdot(phi, phi).real == pytest.approx(1, abs=1e-13)
    np.testing.assert_allclose(phi, u @ psi, atol=1e-15)
    quads = []
    for m in range(modes):
        a = _ladder(modes, cutoff, m)
        quads += [a + a.T, -1j * (a - a.T)]

    def moments(state):
        mean = np.array([np.vdot(state, x @ state).real for x in quads])
        second = np.array(
            [[np.vdot(state, (x @ y + y @ x) @ state).real / 2 for y in quads] for x in quads]
        )
        return mean, second - np.outer(mean, mean)

    mean, cov = moments(psi)
    mean_out, cov_out = moments(phi)
    np.testing.assert_allclose(mean_out, symplectic @ mean + shift, atol=1e-10)
    np.testing.assert_allclose(cov_out, symplectic @ cov @ symplectic.T, atol=1e-9)


def test_phase_kernels_kerr_and_cross_kerr():
    modes, cutoff = 2, 7
    basis = F.basis(modes, cutoff)
    psi = np.exp(1j * np.arange(len(basis))) / np.sqrt(len(basis))
    local = F.basis(2, cutoff).astype(float)
    out = psi.copy()
    F.phase_pure(
        out, modes, cutoff, [1, 0], 0.3 * local[:, 0] * local[:, 1] + 0.2 * local[:, 0] ** 2
    )
    # Local order (1, 0): theta = 0.3 n1 n0 + 0.2 n1^2.
    expected = psi * np.exp(1j * (0.3 * basis[:, 0] * basis[:, 1] + 0.2 * basis[:, 1] ** 2))
    np.testing.assert_allclose(out, expected, atol=1e-15)
    rho = np.outer(psi, psi.conj())
    F.phase_mixed(
        rho, modes, cutoff, [1, 0], 0.3 * local[:, 0] * local[:, 1] + 0.2 * local[:, 0] ** 2
    )
    np.testing.assert_allclose(rho, np.outer(expected, expected.conj()), atol=1e-15)


def test_mixed_kernel_equals_conjugation_and_reduction_is_partial_trace():
    modes, cutoff = 3, 5
    rng = np.random.default_rng(2)
    basis = F.basis(modes, cutoff)
    dim = len(basis)
    x = rng.normal(size=(dim, 3)) + 1j * rng.normal(size=(dim, 3))
    rho = x @ x.conj().T
    rho /= np.trace(rho)
    u = F.gaussian_unitary(np.diag([np.exp(-0.3), np.exp(0.3)]), [0.2, 0.1], cutoff)
    dense = np.zeros((dim, dim), complex)
    for i, a in enumerate(basis):
        for j, b in enumerate(basis):
            if a[0] == b[0] and a[2] == b[2]:
                dense[i, j] = u[a[1], b[1]]
    out = rho.copy()
    F.apply_mixed(out, modes, cutoff, [1], u)
    np.testing.assert_allclose(out, dense @ rho @ dense.conj().T, atol=1e-15)
    # Partial trace onto ordered modes (2, 0) from an explicit tensor of shape c^3 x c^3.
    full = np.zeros((cutoff,) * 6, complex)
    for i, a in enumerate(basis):
        for j, b in enumerate(basis):
            full[tuple(a) + tuple(b)] = rho[i, j]
    reduced = np.einsum("abcdbf->cafd", full)  # keep (mode 2, mode 0), trace mode 1
    keep = F.basis(2, cutoff)
    expected = reduced[keep[:, 0], keep[:, 1]][:, keep[:, 0], keep[:, 1]]
    np.testing.assert_allclose(F.reduced_mixed(rho, modes, cutoff, [2, 0]), expected, atol=1e-15)
    vector = x[:, 0] / np.linalg.norm(x[:, 0])
    np.testing.assert_allclose(
        F.reduced_pure(vector, modes, cutoff, [2, 0]),
        F.reduced_mixed(np.outer(vector, vector.conj()), modes, cutoff, [2, 0]),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        F.marginal_mixed(rho, modes, cutoff, [2, 0]), np.diag(expected).real, atol=1e-15
    )
    report = F.physicality(F.reduced_mixed(rho, modes, cutoff, [2, 0]))
    assert report["hermiticity"] < 1e-15 and abs(report["trace"] - 1) < 1e-14
    assert report["min_eigenvalue"] > -1e-15


def test_thermal_loss_closed_forms():
    cutoff, eta = 9, 0.65
    rho = np.zeros((cutoff, cutoff), complex)
    rho[5, 5] = 1
    out = F.thermal_loss(rho, 1, cutoff, 0, eta, 0.0)
    k = np.arange(6)
    binomial = factorial(5) / (factorial(k) * factorial(5 - k)) * eta**k * (1 - eta) ** (5 - k)
    np.testing.assert_allclose(out.diagonal()[:6].real, binomial, atol=1e-15)
    vacuum = np.zeros((60, 60), complex)
    vacuum[0, 0] = 1
    mean = (1 - 0.5) * 1.5
    thermal = F.thermal_loss(vacuum, 1, 60, 0, 0.5, 1.5).diagonal().real
    np.testing.assert_allclose(
        thermal[:30], mean ** np.arange(30) / (1 + mean) ** np.arange(1, 31), atol=1e-15
    )


def _hermite_functions(y, cutoff):
    n = np.arange(cutoff)[:, None]
    norm = 1 / np.sqrt(2.0**n * factorial(n) * np.sqrt(np.pi))
    return norm * eval_hermite(n, y[None, :]) * np.exp(-(y**2) / 2)


def test_cubic_phase_projected_against_direct_integration():
    gamma, cutoff = 0.3, 10
    matrix, points, change = F.cubic_phase_projected(gamma, cutoff, 1e-13)
    assert change <= 1e-13 and points > 0
    y = np.linspace(-16, 16, 400001)
    h = _hermite_functions(y, cutoff)
    integrand = h[:, None, :] * h[None, :, :] * np.exp(1j * gamma * np.sqrt(2) * y**3 / 3)
    direct = (
        np.trapz(integrand, y, axis=-1)
        if hasattr(np, "trapz")
        else np.trapezoid(integrand, y, axis=-1)
    )
    np.testing.assert_allclose(matrix, direct, atol=1e-11)
    np.testing.assert_allclose(F.cubic_phase_projected(0.0, cutoff)[0], np.eye(cutoff), atol=1e-14)


def test_cubic_phase_heisenberg_action():
    """V^dag p V = p + gamma q^2 for V = exp(i gamma q^3 / 6) (hbar = 2)."""
    gamma, cutoff = 0.05, 80
    v = F.cubic_phase_projected(gamma, cutoff, 1e-13)[0]
    rng = np.random.default_rng(3)
    psi = np.zeros(cutoff, complex)
    psi[:4] = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi /= np.linalg.norm(psi)
    a = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    q, p = a + a.T, -1j * (a - a.T)
    phi = v @ psi
    assert np.vdot(phi, phi).real == pytest.approx(1, abs=1e-12)
    shift = np.vdot(phi, p @ phi) - np.vdot(psi, p @ psi)
    assert shift.real == pytest.approx(gamma * np.vdot(psi, q @ q @ psi).real, abs=1e-10)


def test_cubic_truncated_generator_matches_piquasso_and_converges_to_projected():
    fock = pytest.importorskip("piquasso._math.fock")
    pq = pytest.importorskip("piquasso")
    gamma, cutoff = 0.2, 12
    reference = fock.get_single_mode_cubic_phase_operator(gamma, cutoff, 2.0, pq.NumpyConnector())
    np.testing.assert_allclose(
        F.cubic_phase_truncated_generator(gamma, cutoff), reference, atol=1e-12
    )
    # Independent method: the truncated generator in a space of size M converges to P V P
    # (observed max errors 1.6e-2, 5.8e-7, 3.6e-13, 2.4e-15 for M = 12, 25, 50, 100; beyond
    # that only eigendecomposition rounding remains, growing slowly with M).
    projected = F.cubic_phase_projected(gamma, cutoff)[0][:6, :6]
    errors = [
        np.abs(F.cubic_phase_truncated_generator(gamma, m)[:6, :6] - projected).max()
        for m in (12, 25, 50, 100)
    ]
    assert errors[0] > errors[1] > errors[2] > errors[3]
    assert errors[-1] < 1e-13


def test_projected_cubic_phase_reports_its_truncation_error_exactly():
    """P V psi / |P V psi| has infidelity 1 - |P V psi|^2 to V psi: the retained norm the
    backend reports IS the error. The truncated generator (Piquasso) reports norm 1 while
    its error is about twice as large (measured 3.2e-3 vs 1.6e-3 at cutoff 16)."""
    gamma, cutoff, reference_cutoff = 0.5, 16, 160
    psi = np.zeros(reference_cutoff, complex)
    psi[:2] = 1 / np.sqrt(2), 1j / np.sqrt(2)
    exact = F.cubic_phase_projected(gamma, reference_cutoff)[0] @ psi
    # Reference leakage at cutoff 160 is 1.1e-10, far below the 1.6e-9 resolution used below.
    assert np.vdot(exact, exact).real == pytest.approx(1, abs=1e-9)

    def infidelity(matrix):
        state = matrix @ psi[:cutoff]
        retained = np.vdot(state, state).real
        overlap = abs(np.vdot(exact[:cutoff], state)) ** 2 / retained
        return 1 - overlap, retained

    error, retained = infidelity(F.cubic_phase_projected(gamma, cutoff)[0])
    assert error == pytest.approx(1 - retained, rel=1e-6)
    generator_error, generator_retained = infidelity(
        F.cubic_phase_truncated_generator(gamma, cutoff)
    )
    assert generator_retained == pytest.approx(1, abs=1e-12)
    assert generator_error > 1.5 * error


def test_local_gates_on_multimode_states_match_piquasso_truncation_semantics():
    """P U P on every fibre equals Piquasso's application of a cutoff x cutoff local matrix
    (total-photon truncation, non-passive gates leaking across the cutoff)."""
    pq = pytest.importorskip("piquasso")
    modes, cutoff = 3, 7
    basis = F.basis(modes, cutoff)
    rng = np.random.default_rng(1)
    psi = rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis))
    psi /= np.linalg.norm(psi)
    gates = [
        (pq.CubicPhase(gamma=0.3), F.cubic_phase_truncated_generator(0.3, cutoff)),
        (
            pq.Squeezing(r=0.2),
            F.gaussian_unitary(np.diag([np.exp(-0.2), np.exp(0.2)]), [0, 0], cutoff),
        ),
        (
            pq.Displacement(r=0.3, phi=0.4),
            F.gaussian_unitary(np.eye(2), [0.6 * np.cos(0.4), 0.6 * np.sin(0.4)], cutoff),
        ),
    ]
    for instruction, op in gates:
        for mode in range(modes):
            with pq.Program() as program:
                for occupation, amplitude in zip(basis.tolist(), psi, strict=True):
                    pq.Q() | pq.NumberState(tuple(occupation)) * amplitude
                pq.Q(mode) | instruction
            state = pq.PureFockSimulator(d=modes, config=pq.Config(cutoff=cutoff, hbar=2.0))
            amplitudes = state.execute(program).state.fock_amplitudes_map
            reference = np.array([amplitudes[tuple(b)] for b in basis.tolist()])
            ours = psi.copy()
            F.apply_pure(ours, modes, cutoff, [mode], op)
            # Same global phase convention too: observed max deviation 4e-16.
            np.testing.assert_allclose(ours, reference, atol=1e-14)


def test_parallel_density_kernel_matches_serial_pure_kernel():
    """3 modes, cutoff 16, two-mode CZ: 1.4e8 multiply-adds, above the OpenMP threshold of
    the density kernel; the pure kernel (4.6e4 per vector) runs serially."""
    modes, cutoff = 3, 16
    dim = F.dimension(modes, cutoff)
    rng = np.random.default_rng(7)
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    psi /= np.linalg.norm(psi)
    cz = np.eye(4)
    cz[1, 2] = cz[3, 0] = 0.3
    op = F.gaussian_unitary(cz, np.zeros(4), cutoff)
    rho = np.outer(psi, psi.conj())
    F.apply_mixed(rho, modes, cutoff, [2, 0], op)
    F.apply_pure(psi, modes, cutoff, [2, 0], op)
    np.testing.assert_allclose(rho, np.outer(psi, psi.conj()), atol=1e-15)
