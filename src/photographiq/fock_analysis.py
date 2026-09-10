"""Occupation-aligned finite-Fock metrics and hbar=2 phase-space diagnostics."""

from dataclasses import dataclass

import numpy as np
from scipy.special import eval_genlaguerre, gammaln


def _aligned(left, right):
    if left.nodes != right.nodes:
        raise ValueError("State labels and order must agree")
    basis = sorted(set(left.basis) | set(right.basis))
    index = {b: i for i, b in enumerate(basis)}
    matrices = []
    for state in (left, right):
        rho = np.zeros((len(basis), len(basis)), dtype=complex)
        indices = [index[b] for b in state.basis]
        rho[np.ix_(indices, indices)] = state.density_matrix
        matrices.append(rho)
    return matrices


def _sqrt_psd(matrix):
    values, vectors = np.linalg.eigh((matrix + matrix.conj().T) / 2)
    if values.min() < -1e-8:
        raise ValueError("Density matrix is not positive")
    return (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.conj().T


def fidelity(left, right) -> float:
    """Squared Uhlmann fidelity; occupation alignment permits differing cutoffs."""
    a, b = _aligned(left, right)
    if any(
        state.native is None or hasattr(state.native, "state_vector") for state in (left, right)
    ):
        return float(np.clip(np.einsum("ij,ji->", a, b).real, 0, 1))
    root = _sqrt_psd(a)
    return float(np.clip(np.trace(_sqrt_psd(root @ b @ root)).real ** 2, 0, 1))


def trace_distance(left, right) -> float:
    """Half the trace norm of the occupation-aligned density difference."""
    a, b = _aligned(left, right)
    return float(np.abs(np.linalg.eigvalsh(a - b)).sum() / 2)


@dataclass
class WignerGrid:
    q: np.ndarray
    p: np.ndarray
    values: np.ndarray

    def _integral(self, values):
        # np.trapezoid is unavailable in the minimum supported NumPy 1.26.
        return float(
            np.sum(
                (values[:-1, :-1] + values[1:, :-1] + values[:-1, 1:] + values[1:, 1:])
                * np.diff(self.p)[:, None]
                * np.diff(self.q)[None, :]
                / 4
            )
        )

    @property
    def captured_mass(self):
        return self._integral(self.values)

    @property
    def negative_volume(self):
        """Finite-window integral of max(-W,0); grid/tail convergence is separate."""
        return self._integral(np.maximum(-self.values, 0))

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        extent = max(abs(self.values.min()), abs(self.values.max()))
        artist = ax.pcolormesh(
            self.q, self.p, self.values, shading="auto", cmap="RdBu_r", vmin=-extent, vmax=extent
        )
        ax.set(xlabel="q", ylabel="p", aspect="equal")
        return ax, artist


def wigner(state, q, p, node=None) -> WignerGrid:
    """Single-mode reduced Wigner density, normalized over dq dp.

    Uses analytic displacement matrix elements in Tr[rho D(q+ip) parity]/(2pi),
    not a truncated matrix exponential of displacement.
    """
    if node is not None:
        state = state.reduced((node,))
    if len(state.nodes) != 1:
        raise ValueError("Choose one mode for the Wigner marginal")
    axes = [np.asarray(v, dtype=float) for v in (q, p)]
    if any(
        v.ndim != 1 or len(v) < 2 or not np.isfinite(v).all() or np.any(np.diff(v) <= 0)
        for v in axes
    ):
        raise ValueError("Wigner axes must be finite, increasing vectors with >=2 points")
    q, p = axes
    beta = q[None, :] + 1j * p[:, None]
    radius = abs(beta) ** 2
    rho = state.density_matrix
    total = np.zeros_like(beta, dtype=complex)
    for i, (n,) in enumerate(state.basis):
        for j, (m,) in enumerate(state.basis):
            if abs(rho[i, j]) < 1e-16:
                continue
            if m >= n:
                element = (
                    np.exp((gammaln(n + 1) - gammaln(m + 1)) / 2)
                    * beta ** (m - n)
                    * eval_genlaguerre(n, m - n, radius)
                )
            else:
                element = (
                    np.exp((gammaln(m + 1) - gammaln(n + 1)) / 2)
                    * (-beta.conj()) ** (n - m)
                    * eval_genlaguerre(m, n - m, radius)
                )
            total += rho[i, j] * (-1) ** n * element
    values = (total * np.exp(-radius / 2) / (2 * np.pi)).real
    if not np.isfinite(values).all():
        raise ArithmeticError("Wigner recurrence overflow; reduce grid/cutoff")
    return WignerGrid(q, p, values)
