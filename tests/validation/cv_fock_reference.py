"""Independent dense matrix-exponential reference for single-mode CV gates (hbar = 2).

Shares no code with photographiq: ladder operators on a truncated Fock space, gates as
scipy.linalg.expm of their generators, moments as traces. Truncation is controlled by using
a cutoff well above the photon numbers involved and checking convergence in the tests.

Conventions (PhotoGraphiQ's): q = a + a^dagger, p = -i (a - a^dagger);
R(phi) = exp(i phi n) (q -> q cos phi - p sin phi); S(r) = exp(r (a^2 - a^dagger^2) / 2)
(q -> e^{-r} q); D(q0, p0) displaces <q>, <p> by (q0, p0); Kerr exp(i kappa n^2);
cubic phase exp(i gamma q^3 / 6).
"""

import numpy as np
from scipy.linalg import expm


class FockReference:
    def __init__(self, cutoff):
        self.cutoff = cutoff
        self.a = np.diag(np.sqrt(np.arange(1, cutoff)), 1).astype(complex)
        self.ad = self.a.conj().T
        self.n = self.ad @ self.a
        self.q = self.a + self.ad
        self.p = -1j * (self.a - self.ad)

    def vacuum(self):
        v = np.zeros(self.cutoff, complex)
        v[0] = 1
        return v

    def displace(self, q0, p0):
        alpha = (q0 + 1j * p0) / 2
        return expm(alpha * self.ad - np.conj(alpha) * self.a)

    def rotate(self, phi):
        return expm(1j * phi * self.n)

    def squeeze(self, r):
        return expm(r * (self.a @ self.a - self.ad @ self.ad) / 2)

    def kerr(self, kappa):
        return expm(1j * kappa * self.n @ self.n)

    def cubic(self, gamma):
        return expm(1j * gamma * np.linalg.matrix_power(self.q, 3) / 6)

    def moments(self, psi):
        """(<q>, <p>, Var q, Var p, <n>) of a state vector."""
        e = lambda op: float(np.real(np.vdot(psi, op @ psi)))  # noqa: E731
        q, p = e(self.q), e(self.p)
        return q, p, e(self.q @ self.q) - q**2, e(self.p @ self.p) - p**2, e(self.n)
