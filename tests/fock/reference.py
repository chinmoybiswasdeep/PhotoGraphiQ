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
