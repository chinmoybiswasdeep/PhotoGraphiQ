"""Single-mode Gaussian unitary <m|U|n> against 60-digit references (hbar = 2).

Regression: the Bargmann recurrence combining squeezing and displacement amplified
rounding exponentially (6e-6 at |alpha| = 4, r = 0.3, cutoff 80; 1.4e2 at |alpha| = 6,
cutoff 100), and squeezing alone lost 7e-5 at r = 1, size 400 (rows < 100). The engine now builds
U = R S_r R D(beta) from stable Jacobi and Laguerre recurrences (cpp/src/fock/gates.cpp);
measured error <= 4e-14 on the grid of docs/validation/native-fock-gaussian-unitary.md.

The reference runs the same generating-function recurrence in mpmath, where its
rounding amplification (up to 1e34 here) leaves >= 20 correct digits, and is checked
independently against the Heisenberg relation a U = U (T a + R a^dag + delta).
"""

import numpy as np
import pytest

from photographiq import native

try:
    F = native.require("fock")
except ImportError as exc:  # pragma: no cover - exercised only without the extension
    pytest.skip(f"native fock component unavailable: {exc}", allow_module_level=True)

mp = pytest.importorskip("mpmath")
DPS = 60


def _symplectic(r, phi1, phi2):
    """rotation(phi1) diag(e^-r, e^r) rotation(phi2), in double and exactly in mpmath."""

    def rot(t, lib):
        return [[lib.cos(t), -lib.sin(t)], [lib.sin(t), lib.cos(t)]]

    double = np.array(rot(phi1, np)) @ np.diag([np.exp(-r), np.exp(r)]) @ np.array(rot(phi2, np))
    with mp.workdps(DPS + 10):
        exact = mp.matrix(rot(mp.mpf(phi1), mp)) * mp.diag([mp.exp(-r), mp.exp(r)])
        exact = exact * mp.matrix(rot(mp.mpf(phi2), mp))
    return double, exact


def _reference(exact, d, rows, cols):
    """<m|U|n>, m < rows, n < cols, with U^dag x U = S x + d and the engine's phase."""
    with mp.workdps(DPS):
        i = mp.mpc(0, 1)
        s00, s01, s10, s11 = exact[0, 0], exact[0, 1], exact[1, 0], exact[1, 1]
        # U^dag a U = T a + R a^dag + delta, from a = (q + i p) / 2, q = a + a^dag, p = -i(a - a^dag).
        t = (s00 + s11 + i * (s10 - s01)) / 2
        r = (s00 - s11 + i * (s10 + s01)) / 2
        delta = mp.mpc(d[0], d[1]) / 2
        tc = mp.conj(t)
        azz, azw, aww = r / tc, 1 / tc, -mp.conj(r) / tc
        bw = -mp.conj(delta) / tc
        bz = delta + r * bw
        root = [mp.sqrt(n) for n in range(max(rows, cols) + 1)]
        g = [[mp.mpc(0)] * cols for _ in range(rows)]
        g[0][0] = mp.exp(-(abs(delta) ** 2) / 2 + mp.conj(delta) ** 2 * azz / 2) / mp.sqrt(abs(t))
        for n in range(cols - 1):
            g[0][n + 1] = (bw * g[0][n] + (aww * root[n] * g[0][n - 1] if n else 0)) / root[n + 1]
        for m in range(rows - 1):
            for n in range(cols):
                v = bz * g[m][n] + (azz * root[m] * g[m - 1][n] if m else 0)
                g[m + 1][n] = (v + (azw * root[n] * g[m][n - 1] if n else 0)) / root[m + 1]
        worst = mp.mpf(0)
        for m in range(rows - 1):
            for n in range(cols - 1):
                rhs = r * root[n + 1] * g[m][n + 1] + delta * g[m][n]
                rhs += t * root[n] * g[m][n - 1] if n else 0
                worst = max(worst, abs(root[m + 1] * g[m + 1][n] - rhs))
        assert worst < mp.mpf(10) ** -20, "reference violates the Heisenberg relation"
        return np.array([[complex(x) for x in row] for row in g])


@pytest.mark.parametrize(
    "alpha,r,phi1,phi2,cutoff,before",
    [
        (4 * np.exp(0.7j), 0.3, 0.4, -1.1, 80, 8.0e-5),  # the reported case
        (6 * np.exp(2.1j), 0.3, 1.9, 0.3, 100, 2.1e2),  # worst region of the grid
        (3 * np.exp(-0.5j), 1.0, -0.8, 2.5, 100, 6.7e-9),  # strong squeezing, large |beta|
        (0.5 * np.exp(1.3j), 0.6, 0.2, 0.9, 40, 1.1e-15),  # where the old path was accurate
    ],
)
def test_combined_squeeze_displacement_matches_high_precision(alpha, r, phi1, phi2, cutoff, before):
    del before  # max error of the old Bargmann path on this case, for the record
    double, exact = _symplectic(r, phi1, phi2)
    d = np.array([2 * alpha.real, 2 * alpha.imag])
    ref = _reference(exact, d, cutoff, cutoff)
    u = F.gaussian_unitary(double, d, cutoff)
    np.testing.assert_allclose(u, ref, atol=1e-12, rtol=0)  # observed <= 9e-15
    np.testing.assert_allclose(u[:, 0], ref[:, 0], atol=3e-14, rtol=0)  # pure GaussianInput
    # Column norms follow the exact leakage above the cutoff; leak-free columns are unit.
    norms, exact = np.linalg.norm(u, axis=0) ** 2, np.linalg.norm(ref, axis=0) ** 2
    np.testing.assert_allclose(norms, exact, atol=1e-13, rtol=0)
    kept = np.abs(1 - exact) < 1e-14
    np.testing.assert_allclose(norms[kept], 1, atol=1e-13, rtol=0)


def test_squeezing_rows_stay_accurate_at_large_size():
    """Rows < 60 of S(r = 1) at size 400 (the Bargmann recurrence: 3.3e-10)."""
    double, exact = _symplectic(1.0, 0.3, 1.1)
    ref = _reference(exact, [0.0, 0.0], 60, 400)
    u = F.gaussian_unitary(double, np.zeros(2), 400)
    np.testing.assert_allclose(u[:60], ref, atol=1e-12, rtol=0)  # observed 1e-14
    # Rows without weight beyond column 400 (S(1)|m> reaches ~e^2 m photons) are orthonormal.
    kept = np.abs(1 - np.linalg.norm(ref, axis=1) ** 2) < 1e-14
    assert kept.sum() >= 20
    block = u[:60][kept]
    np.testing.assert_allclose(block @ block.conj().T, np.eye(kept.sum()), atol=1e-13)


def test_thermal_columns_of_displaced_squeezed_input():
    """The columns a mixed GaussianInput uses: cutoff 60, |alpha| = 4, r = 0.3 (old: 6.7e-9)."""
    alpha = 4 * np.exp(0.2j)
    double, exact = _symplectic(0.3, 0.5, 0.0)
    d = np.array([2 * alpha.real, 2 * alpha.imag])
    ref = _reference(exact, d, 60, 30)
    u = F.gaussian_unitary(double, d, 60)
    np.testing.assert_allclose(u[:, :30], ref, atol=1e-13, rtol=0)  # observed 3e-15
