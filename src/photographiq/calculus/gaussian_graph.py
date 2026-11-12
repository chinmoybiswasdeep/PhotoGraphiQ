"""Graphical calculus for Gaussian pure states (Menicucci, Flammia, van Loock 2011).

Reference: N. C. Menicucci, S. T. Flammia, P. van Loock, "Graphical calculus for Gaussian pure
states", Phys. Rev. A 83, 042335 (2011), arXiv:1007.0725 ("MFvL" below, equation numbers from
arXiv v3).

A zero-mean N-mode Gaussian pure state is, up to a global phase, a complex symmetric matrix
``Z = V + iU`` with ``U = Im Z > 0`` (MFvL Eqs. 2.21-2.22). Read as the adjacency matrix of a
complex-weighted graph, ``Re Z`` holds CZ-type edge weights and self-loops, ``Im Z`` the
finite-squeezing part. In PhotoGraphiQ conventions (hbar = 2, ``[q, p] = 2i``, vacuum
covariance ``I``, interleaved ``(q0, p0, q1, p1, ...)``) the position wave function is

    psi(q) ∝ exp((i/4) q^T Z q)

and the covariance in qq/pp block form is exactly ``2 x`` MFvL Eq. 2.19 (which uses hbar = 1):

    cov = [[U^-1, U^-1 V], [V U^-1, U + V U^-1 V]].

``Z`` itself is the same matrix in both conventions (q_here = sqrt(2) q_MFvL), so vacuum is
``Z = iI`` and the p-squeezed CV cluster state is ``Z = A + i e^{-2r} I`` (MFvL Eq. 2.29),
``r`` being the resource squeezing of :meth:`GaussianBackend.prepare` (covariance
``diag(e^{2r}, e^{-2r})``). Symplectic matrices are also convention independent, so a Gaussian
unitary with symplectic ``S = [[A, B], [C, D]]`` (qq/pp blocks) acts by MFvL Eq. 2.24,
``Z -> (C + D Z)(A + B Z)^{-1}``.

Displacements are not part of the graph (MFvL track states "up to displacement"); this module
tracks them exactly in a separate interleaved ``mean`` vector, so every object here is a full
Gaussian pure state. Only pure Gaussian states, Gaussian unitaries and ideal q/p homodyne
measurements are covered: no mixed states, loss, non-Gaussian elements or CV-ZX.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence

import numpy as np
from scipy.linalg import block_diag

from .. import gaussian as gm
from ..states import GaussianState

__all__ = ["GaussianGraph"]


def _blocks(n: int) -> np.ndarray:
    """Index permutation interleaved (q0, p0, q1, ...) -> block (q0, q1, ..., p0, p1, ...)."""
    return np.r_[0 : 2 * n : 2, 1 : 2 * n : 2]


class GaussianGraph:
    """Pure Gaussian state as a complex-weighted graph ``Z`` plus an exact mean vector.

    Mathematically the state is ``D(mean) |Z>``, where ``<q|Z> ∝ exp((i/4) q^T Z q)`` in
    hbar = 2 units and ``D`` is the phase-space displacement by the interleaved ``mean``.
    Instances are immutable; every rule returns a new graph.

    Args:
        Z (array-like): Complex symmetric N x N matrix with positive definite ``Im Z``.
        nodes (Sequence): N unique hashable mode labels, row order of ``Z``. Default 0..N-1.
        mean (array-like): Interleaved hbar = 2 quadrature mean (length 2N). Default zero.

    Raises:
        ValueError: Z not square, not finite, not symmetric or Im Z not positive definite.
        ValueError: Labels not unique or mean of the wrong length.
    """

    def __init__(
        self,
        Z,
        nodes: Sequence[Hashable] | None = None,
        mean=None,
    ):
        Z = np.array(Z, dtype=complex)
        n = Z.shape[0] if Z.ndim == 2 else -1
        if Z.shape != (n, n):
            raise ValueError("Z must be a square matrix")
        if not np.isfinite(Z).all():
            raise ValueError("Z must be finite")
        scale = max(1.0, float(np.abs(Z).max(initial=0.0)))
        if not np.allclose(Z, Z.T, rtol=0, atol=1e-10 * scale):
            raise ValueError("Z must be complex symmetric (Z = Z^T, not Hermitian)")
        Z = (Z + Z.T) / 2
        try:
            np.linalg.cholesky(Z.imag)
        except np.linalg.LinAlgError:
            raise ValueError("Im Z must be positive definite (normalizable pure state)") from None
        self.nodes = tuple(range(n)) if nodes is None else tuple(nodes)
        if len(self.nodes) != n or len(set(self.nodes)) != n:
            raise ValueError("Provide one unique label per mode")
        mean = np.zeros(2 * n) if mean is None else np.array(mean, dtype=float)
        if mean.shape != (2 * n,) or not np.isfinite(mean).all():
            raise ValueError("mean must be a finite interleaved vector of length 2N")
        Z.setflags(write=False)
        mean.setflags(write=False)
        self.Z, self.mean = Z, mean

    def __repr__(self):
        return f"GaussianGraph(modes={len(self.nodes)}, nodes={self.nodes!r})"

    # ------------------------------------------------------------------ constructors

    @classmethod
    def vacuum(cls, nodes: Sequence[Hashable]) -> GaussianGraph:
        """Vacuum on ``nodes``: ``Z = iI`` (covariance ``I``), no edges."""
        nodes = tuple(nodes)
        return cls(1j * np.eye(len(nodes)), nodes)

    @classmethod
    def squeezed(cls, nodes: Sequence[Hashable], r: float) -> GaussianGraph:
        """p-squeezed resource states, covariance ``diag(e^{2r}, e^{-2r})`` per mode.

        Matches :meth:`GaussianBackend.prepare` with squeezing ``r``. Graph: ``Z = i e^{-2r} I``
        (imaginary self-loops only; MFvL Eq. 2.28-2.29 with A = 0).
        """
        nodes = tuple(nodes)
        return cls(1j * np.exp(-2.0 * r) * np.eye(len(nodes)), nodes)

    @classmethod
    def cluster(cls, adjacency, r: float, nodes: Sequence[Hashable] | None = None) -> GaussianGraph:
        """Canonical approximate CV cluster state ``Z = A + i e^{-2r} I`` (MFvL Eq. 2.29).

        Prepare every mode p-squeezed with resource squeezing ``r`` and apply
        ``CZ(A_uv)`` (``p_u += A_uv q_v``) on every edge; ``Z -> A`` as ``r -> inf`` (Eq. 2.30).

        Args:
            adjacency (array-like): Real symmetric weighted adjacency matrix, zero diagonal.
            r (float): Resource squeezing (finite).
            nodes (Sequence): Mode labels in row order.
        """
        a = np.asarray(adjacency, dtype=float)
        if a.ndim != 2 or not np.allclose(a, a.T, rtol=0, atol=0) or np.any(np.diag(a)):
            raise ValueError("adjacency must be real symmetric with zero diagonal")
        return cls(a + 1j * np.exp(-2.0 * r) * np.eye(len(a)), nodes)

    @classmethod
    def from_covariance(
        cls, covariance, nodes: Sequence[Hashable] | None = None, mean=None, *, rtol=1e-9
    ) -> GaussianGraph:
        """Graph of a pure Gaussian covariance (MFvL Eqs. 2.25-2.27, times 2 for hbar = 2).

        With qq/pp blocks of the covariance, ``U = (cov_qq)^{-1}`` and ``V = cov_pq U``. The
        state is pure iff the covariance equals the one rebuilt from ``Z = V + iU``; this is
        checked to relative tolerance ``rtol`` (max-norm). Mixed states have no graph.

        Raises:
            ValueError: Covariance malformed, or not pure (mixed or unphysical).
        """
        cov = np.asarray(covariance, dtype=float)
        n2 = cov.shape[0] if cov.ndim == 2 else -1
        if cov.shape != (n2, n2) or n2 % 2 or not np.isfinite(cov).all():
            raise ValueError("covariance must be a finite 2N x 2N matrix")
        n = n2 // 2
        b = cov[np.ix_(_blocks(n), _blocks(n))]
        qq, pq = b[:n, :n], b[n:, :n]
        try:
            u = np.linalg.inv(qq)
        except np.linalg.LinAlgError:
            raise ValueError("covariance has a singular qq block (unphysical)") from None
        Z = pq @ u + 1j * u
        Z = (Z + Z.T) / 2
        try:
            graph = cls(Z, nodes, mean)
        except ValueError as exc:
            raise ValueError(f"covariance is not a pure Gaussian state: {exc}") from None
        rebuilt = graph.to_covariance()
        residual = np.abs(rebuilt - cov).max() / max(1.0, np.abs(cov).max())
        if residual > rtol:
            purity = float(np.exp(-0.5 * np.linalg.slogdet(cov)[1]))
            raise ValueError(
                "covariance is not a pure Gaussian state (mixed states have no Gaussian graph): "
                f"relative residual {residual:.2e} > {rtol:.0e}, det(V)^(-1/2) = {purity:.6g}"
            )
        return graph

    @classmethod
    def from_state(cls, state: GaussianState, *, rtol=1e-9) -> GaussianGraph:
        """Graph of a pure :class:`GaussianState` (labels and mean kept); see from_covariance."""
        return cls.from_covariance(state.covariance, state.nodes, state.mean, rtol=rtol)

    # ------------------------------------------------------------------ views

    def to_covariance(self) -> np.ndarray:
        """Interleaved hbar = 2 covariance: ``[[U^-1, U^-1 V], [V U^-1, U + V U^-1 V]]``."""
        n = len(self.nodes)
        u, v = self.Z.imag, self.Z.real
        ui = np.linalg.inv(u)
        ui = (ui + ui.T) / 2
        block = np.block([[ui, ui @ v], [v @ ui, u + v @ ui @ v]])
        inverse = np.argsort(_blocks(n))
        cov = block[np.ix_(inverse, inverse)]
        return (cov + cov.T) / 2

    def to_state(self) -> GaussianState:
        """The same state as a labelled :class:`GaussianState` (mean and covariance)."""
        return GaussianState(self.mean, self.to_covariance(), self.nodes)

    def edges(self, atol: float = 0.0) -> dict[tuple, complex]:
        """Complex edge weights ``{(u, v): Z_uv}`` for ``u <= v`` with ``|Z_uv| > atol``.

        Diagonal entries ``(u, u)`` are self-loops; ``Im Z_uu > 0`` always.
        """
        n = len(self.nodes)
        return {
            (self.nodes[i], self.nodes[j]): complex(self.Z[i, j])
            for i in range(n)
            for j in range(i, n)
            if abs(self.Z[i, j]) > atol
        }

    def _index(self, node) -> int:
        try:
            return self.nodes.index(node)
        except ValueError:
            raise KeyError(f"unknown node {node!r}") from None

    def _replace(self, Z, nodes=None, mean=None) -> GaussianGraph:
        return GaussianGraph(
            Z, self.nodes if nodes is None else nodes, self.mean if mean is None else mean
        )

    # ------------------------------------------------------------------ Gaussian unitaries

    def apply(self, symplectic, nodes: Sequence[Hashable]) -> GaussianGraph:
        """Gaussian unitary with interleaved symplectic ``S`` on ``nodes`` (MFvL Eq. 2.24).

        Heisenberg action ``x -> S x`` (covariance ``S cov S^T``, mean ``S mean``), embedded as
        the identity on other modes. In qq/pp blocks ``S = [[A, B], [C, D]]`` and
        ``Z' = (C + D Z)(A + B Z)^{-1}``; MFvL Eq. 3.7 is the same map for a local block.

        Raises:
            ValueError: Repeated nodes, wrong shape, or ``S`` not real symplectic.
        """
        nodes = tuple(nodes)
        if len(set(nodes)) != len(nodes):
            raise ValueError("Repeated gate modes")
        s = np.asarray(symplectic)
        k = len(nodes)
        if s.shape != (2 * k, 2 * k) or not np.isrealobj(s) or not np.isfinite(s).all():
            raise ValueError("symplectic must be a finite real 2k x 2k matrix for k nodes")
        tol = 1e-10 * max(1.0, float(np.linalg.norm(s, 2)) ** 2)
        if not np.allclose(s @ gm.omega(k) @ s.T, gm.omega(k), rtol=0, atol=tol):
            raise ValueError("Transformation must be symplectic (S Omega S^T = Omega)")
        n = len(self.nodes)
        idx = [j for node in nodes for j in (2 * self._index(node), 2 * self._index(node) + 1)]
        full = gm.embed(s.astype(float), idx, 2 * n)
        b = full[np.ix_(_blocks(n), _blocks(n))]
        a_, b_, c_, d_ = b[:n, :n], b[:n, n:], b[n:, :n], b[n:, n:]
        # Z' = (C + D Z)(A + B Z)^{-1}  <=>  Z'^T = (A + B Z)^{-T} (C + D Z)^T
        Z = np.linalg.solve((a_ + b_ @ self.Z).T, (c_ + d_ @ self.Z).T).T
        return self._replace((Z + Z.T) / 2, mean=full @ self.mean)

    def cz(self, u: Hashable, v: Hashable, weight: float = 1.0) -> GaussianGraph:
        """``CZ(g)``: ``p_u += g q_v``, ``p_v += g q_u``. Graph rule ``Z_uv += g`` (Eq. 3.23)."""
        i, j = self._index(u), self._index(v)
        if i == j:
            raise ValueError("CZ needs two distinct modes")
        Z = self.Z.copy()
        Z[i, j] += weight
        Z[j, i] += weight
        return self._replace(
            Z,
            mean=gm.embed(gm.cz(weight), [2 * i, 2 * i + 1, 2 * j, 2 * j + 1], 2 * len(self.nodes))
            @ self.mean,
        )

    def quadratic_phase(self, node: Hashable, s: float) -> GaussianGraph:
        """Shear ``p += s q`` (backend ``quadratic_phase``): self-loop ``Z_kk += s`` (Eq. 3.14)."""
        k = self._index(node)
        Z = self.Z.copy()
        Z[k, k] += s
        mean = self.mean.copy()
        mean[2 * k + 1] += s * mean[2 * k]
        return self._replace(Z, mean=mean)

    def rotate(self, node: Hashable, angle: float) -> GaussianGraph:
        """Phase rotation ``gaussian.rotation(angle)`` (MFvL ``S_phase(-angle)``, Eq. 3.18)."""
        return self.apply(gm.rotation(angle), (node,))

    def squeeze(self, node: Hashable, r: float) -> GaussianGraph:
        """Squeezing ``diag(e^-r, e^r)`` (MFvL ``S_squeeze(-r)``, Eq. 3.16).

        Graph rule: ``Z_kk -> e^{2r} Z_kk``, edges at ``k`` scaled by ``e^r``.
        """
        return self.apply(gm.squeezing(r), (node,))

    def beamsplitter(self, u: Hashable, v: Hashable, theta: float) -> GaussianGraph:
        """Beam splitter ``gaussian.beamsplitter(theta)`` = MFvL ``S_BS(theta)`` (Eqs. 3.10-3.12)."""
        return self.apply(gm.beamsplitter(theta), (u, v))

    # ------------------------------------------------------------------ measurements

    def _condition(self, node, outcome, quadrature: int) -> GaussianGraph:
        k = self._index(node)
        if np.ndim(outcome) or not np.isrealobj(outcome) or not np.isfinite(outcome):
            raise ValueError("Homodyne outcome must be a finite real scalar")
        rest = [j for j in range(len(self.nodes)) if j != k]
        z = self.Z[rest, k]
        # Outcome for the zero-mean graph state: the local displacement D(mean_k) only shifts it.
        m = float(outcome) - self.mean[2 * k + quadrature]
        if quadrature == 0:
            Z, b = self.Z[np.ix_(rest, rest)], m * z
        else:
            Z = self.Z[np.ix_(rest, rest)] - np.outer(z, z) / self.Z[k, k]
            b = m * z / self.Z[k, k]
        # Conditional wave function ∝ exp((i/4) q^T Z q + (i/2) b^T q): mean q = -U^{-1} Im b,
        # mean p = Re Z q + Re b (from p psi = (Z q + b) psi with p = -2i d/dq).
        q = -np.linalg.solve(Z.imag, b.imag) if rest else np.zeros(0)
        p = Z.real @ q + b.real
        shift = np.empty(2 * len(rest))
        shift[0::2], shift[1::2] = q, p
        mean = self.mean[[j for i in rest for j in (2 * i, 2 * i + 1)]] + shift
        return GaussianGraph((Z + Z.T) / 2, tuple(self.nodes[i] for i in rest), mean)

    def measure_q(self, node: Hashable, outcome: float) -> GaussianGraph:
        """Ideal destructive q homodyne with outcome ``m``: delete the vertex (MFvL Sec. III.D-E).

        ``Z' = Z`` with row/column ``k`` removed. The outcome enters only through the mean:
        projecting ``<q_k = m|`` leaves the linear phase ``exp((i/2) m Z_rk^T q_r)``.
        Homodyne at angle ``theta`` (record ``q cos theta + p sin theta``) is
        ``rotate(node, -theta).measure_q(node, m)``.
        """
        return self._condition(node, outcome, 0)

    def measure_p(self, node: Hashable, outcome: float) -> GaussianGraph:
        """Ideal destructive p homodyne: Schur complement ``Z' = Z_rr - z z^T / Z_kk``.

        ``z = Z_rk``. This is MFvL Sec. III.D: inverse Fourier transform (Eq. 3.20 rule) then
        vertex deletion; equivalently the Gaussian integral ``∫ dq_k e^{-i m q_k/2} psi``.
        Since ``Im Z_kk > 0``, ``Z_kk != 0`` and the rule is always defined.
        """
        return self._condition(node, outcome, 1)

    # ------------------------------------------------------------------ composition

    def tensor(self, other: GaussianGraph) -> GaussianGraph:
        """Tensor product: direct sum ``Z ⊕ Z'`` (disjoint union of graphs), means concatenated.

        Raises:
            ValueError: The two graphs share a node label.
        """
        if set(self.nodes) & set(other.nodes):
            raise ValueError("Tensor product needs disjoint node labels")
        return GaussianGraph(
            block_diag(self.Z, other.Z), self.nodes + other.nodes, np.r_[self.mean, other.mean]
        )

    def relabel(self, mapping: Mapping[Hashable, Hashable]) -> GaussianGraph:
        """Rename nodes (graph isomorphism; ``Z`` and order unchanged). Unmapped labels kept."""
        return self._replace(self.Z, nodes=tuple(mapping.get(n, n) for n in self.nodes))

    def reorder(self, nodes: Iterable[Hashable]) -> GaussianGraph:
        """Permute modes to the given label order: ``Z -> P Z P^T`` (same state)."""
        order = [self._index(n) for n in nodes]
        if sorted(order) != list(range(len(self.nodes))):
            raise ValueError("reorder needs a permutation of all node labels")
        idx = [j for i in order for j in (2 * i, 2 * i + 1)]
        return GaussianGraph(
            self.Z[np.ix_(order, order)], tuple(self.nodes[i] for i in order), self.mean[idx]
        )
