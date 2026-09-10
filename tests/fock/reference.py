"""Independent dense test oracle; never imported by the production package."""

import numpy as np
from scipy.linalg import expm


def operators(cutoff):
    a = np.diag(np.sqrt(np.arange(1, cutoff)), 1).astype(complex)
    return a, a.conj().T, a.conj().T @ a, a + a.conj().T, -1j * (a - a.conj().T)


def cubic(cutoff, gamma):
    q = operators(cutoff)[3]
    return expm(1j * gamma * q @ q @ q / 6)


def displacement(cutoff, alpha):
    a, ad, *_ = operators(cutoff)
    return expm(alpha * ad - alpha.conjugate() * a)


def squeeze(cutoff, r):
    a, ad, *_ = operators(cutoff)
    return expm(r * (a @ a - ad @ ad) / 2)


def polynomial_operators(cutoff):
    a, ad, n, q, p = operators(cutoff)
    return {
        "a": a,
        "ad": ad,
        "n": n,
        "q": q,
        "p": p,
        "q2": q @ q,
        "q3": q @ q @ q,
        "q4": np.linalg.matrix_power(q, 4),
        "n2": n @ n,
    }


def rotation(cutoff, angle):
    return np.diag(np.exp(1j * angle * np.arange(cutoff)))


def quadratic(cutoff, s):
    return expm(1j * s * polynomial_operators(cutoff)["q2"] / 4)


def kerr(cutoff, kappa):
    return np.diag(np.exp(1j * kappa * np.arange(cutoff) ** 2))


def phase_aligned_distance(a, b):
    a, b = np.asarray(a), np.asarray(b)
    overlap = np.vdot(a, b)
    return np.linalg.norm(a - b * (overlap.conjugate() / abs(overlap) if abs(overlap) else 1))
