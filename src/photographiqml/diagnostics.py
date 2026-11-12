"""Representation-specific diagnostics."""

import numpy as np

from .logical import statevector


def concurrence(state):
    """Pure two-qubit concurrence only; not a CV entanglement measure."""
    a, b, c, d = statevector(state, 2)
    return float(2 * abs(a * d - b * c))


def pure_qfi(state, generator):
    """4 Var(H) for a normalized pure state and Hermitian generator."""
    state = np.asarray(state, dtype=complex)
    if state.ndim != 1 or len(state) < 2:
        raise ValueError("QFI requires a logical statevector")
    state = statevector(state, int(np.log2(len(state))))
    generator = np.asarray(generator, dtype=complex)
    if (
        generator.shape != (len(state), len(state))
        or not np.isfinite(generator).all()
        or not np.allclose(generator, generator.conj().T)
    ):
        raise ValueError("QFI generator must be finite and Hermitian with matching dimension")
    image = generator @ state
    return float(4 * (np.vdot(image, image).real - np.vdot(state, image).real ** 2))
