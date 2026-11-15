"""Interferometer unitaries: Haar sampling, Clements decomposition, passive symplectics.

Convention (shared with every PhotoGraphiQ backend): an interferometer ``U`` acts on
annihilation operators in the Heisenberg picture as ``a_k -> sum_j U_kj a_j``, i.e.
``a_j^dag -> sum_k U_kj a_k^dag`` on states. ``Rotate(phi)`` is ``U = exp(i phi)`` and
``BeamSplitter(theta)`` on modes ``(u, v)`` is ``U = [[cos, -sin], [sin, cos]]``. With
``a = (q + i p)/2`` the interleaved ``(q0, p0, q1, p1, ...)`` symplectic of ``U`` has
2x2 blocks ``[[Re U_jk, -Im U_jk], [Im U_jk, Re U_jk]]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_TOL = 1e-10


def haar_unitary(modes: int, rng: np.random.Generator | int | None = None) -> np.ndarray:
    """Haar-random ``modes x modes`` unitary (Mezzadri 2007: QR of a Ginibre matrix).

    The phases of ``diag(R)`` are moved into ``Q`` so the result is exactly Haar
    distributed (plain QR output is not).
    """
    if int(modes) != modes or modes < 1:
        raise ValueError("modes must be a positive integer")
    rng = np.random.default_rng(rng)
    z = (rng.standard_normal((modes, modes)) + 1j * rng.standard_normal((modes, modes))) / np.sqrt(
        2
    )
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * (d / np.abs(d))


def check_unitary(unitary, atol: float = _TOL) -> np.ndarray:
    """Return ``unitary`` as a complex array or raise ``ValueError`` if ``U^dag U != I``."""
    u = np.asarray(unitary, dtype=complex)
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError("Expected a square matrix")
    if not np.allclose(u.conj().T @ u, np.eye(len(u)), rtol=0.0, atol=atol):
        raise ValueError("Matrix is not unitary")
    return u


def passive_symplectic(unitary) -> np.ndarray:
    """Interleaved-quadrature symplectic matrix of an interferometer (see module doc)."""
    u = check_unitary(unitary)
    s = np.empty((2 * len(u), 2 * len(u)))
    s[0::2, 0::2], s[0::2, 1::2] = u.real, -u.imag
    s[1::2, 0::2], s[1::2, 1::2] = u.imag, u.real
    return s


def mzi(modes: int, u: int, v: int, theta: float, phi: float) -> np.ndarray:
    """``BeamSplitter(theta)`` on ``(u, v)`` after ``Rotate(phi)`` on ``u``, as a unitary.

    The 2x2 block is ``[[e^{i phi} cos, -sin], [e^{i phi} sin, cos]]`` (Clements et al.).
    """
    t = np.eye(modes, dtype=complex)
    c, s, e = np.cos(theta), np.sin(theta), np.exp(1j * phi)
    t[u, u], t[u, v], t[v, u], t[v, v] = e * c, -s, e * s, c
    return t


@dataclass(frozen=True)
class ClementsDecomposition:
    """``U = diag(exp(i output_phases)) @ T_K @ ... @ T_1`` with ``T_k = mzi(*elements[k])``.

    ``elements`` holds ``(u, v, theta, phi)`` in application (time) order. Each element is
    ``Rotate(phi)`` on ``u`` followed by ``BeamSplitter(theta)`` on ``(u, v)``; the final
    layer is ``Rotate(output_phases[i])`` on every mode ``i``.
    """

    modes: int
    elements: tuple[tuple[int, int, float, float], ...]
    output_phases: np.ndarray

    def unitary(self) -> np.ndarray:
        """Reconstruct the decomposed unitary."""
        total = np.eye(self.modes, dtype=complex)
        for u, v, theta, phi in self.elements:
            total = mzi(self.modes, u, v, theta, phi) @ total
        return np.exp(1j * self.output_phases)[:, None] * total

    def operations(self) -> list[tuple]:
        """PhotoGraphiQ gate list ``("rotate", mode, angle)`` / ``("beamsplitter", u, v, theta)``."""
        ops: list[tuple] = []
        for u, v, theta, phi in self.elements:
            ops += [("rotate", u, phi), ("beamsplitter", u, v, theta)]
        ops += [("rotate", i, float(a)) for i, a in enumerate(self.output_phases)]
        return ops


def clements_decomposition(unitary, atol: float = _TOL) -> ClementsDecomposition:
    """Rectangular-mesh decomposition of Clements et al., Optica 3, 1460 (2016).

    Nulls the lower triangle alternately from the right (``U T^-1``) and the left
    (``T U``), then moves the residual diagonal phases to the output through
    ``T(theta, phi)^dag D = D' T(theta, phi')`` with ``phi' = alpha - beta + pi`` and
    ``D'`` phases ``(beta - phi + pi, beta)``. ``m(m-1)/2`` elements; the reconstruction
    error is checked against ``atol`` and a ``RuntimeError`` raised if exceeded.
    """
    u = check_unitary(unitary, atol)
    m = len(u)
    work = u.copy()
    right: list[tuple[int, int, float, float]] = []
    left: list[tuple[int, int, float, float]] = []
    for k, i in enumerate(range(m - 2, -1, -1)):
        if k % 2 == 0:
            for j in reversed(range(m - 1 - i)):
                row, col = i + j + 1, j  # null work[row, col] from the right with cols (j, j+1)
                if abs(work[row, col + 1]) < 1e-300:
                    theta, phi = np.pi / 2, 0.0
                else:
                    ratio = work[row, col] / work[row, col + 1]
                    theta, phi = np.arctan(abs(ratio)), np.angle(ratio)
                right.append((col, col + 1, theta, phi))
                work = work @ mzi(m, col, col + 1, theta, phi).conj().T
        else:
            for j in range(m - 1 - i):
                row, col = i + j + 1, j  # null work[row, col] from the left with rows (row-1, row)
                if abs(work[row - 1, col]) < 1e-300:
                    theta, phi = np.pi / 2, 0.0
                else:
                    ratio = -work[row, col] / work[row - 1, col]
                    theta, phi = np.arctan(abs(ratio)), np.angle(ratio)
                left.append((row - 1, row, theta, phi))
                work = mzi(m, row - 1, row, theta, phi) @ work
    # Now U = L_1^dag ... L_K^dag D R_last ... R_1 with D = diag(work).
    phases = np.angle(np.diag(work)).copy()
    moved = []
    for a, b, theta, phi in reversed(left):
        alpha, beta = phases[a], phases[b]
        moved.append((a, b, theta, float(np.mod(alpha - beta + np.pi, 2 * np.pi))))
        phases[a], phases[b] = beta - phi + np.pi, beta
    elements = tuple((a, b, float(t), float(p)) for a, b, t, p in right) + tuple(moved)
    result = ClementsDecomposition(m, elements, np.mod(phases, 2 * np.pi))
    error = np.abs(result.unitary() - u).max()
    if error > max(atol, 1e3 * m * np.finfo(float).eps):
        raise RuntimeError(f"Clements reconstruction error {error:.2e} exceeds tolerance")
    return result
