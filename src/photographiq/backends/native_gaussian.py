"""Gaussian backend whose state updates run in the compiled ``photographiq._native.gaussian``.

``NativeGaussianBackend`` has the semantics of :class:`GaussianBackend` (hbar=2, interleaved
quadratures, statistical covariance with vacuum V = I, exact Schur-complement conditioning,
incident-quadrature calibrated homodyne records, postselection with densities, loss with
thermal photons, quadratic phase, heterodyne/generaldyne, displacement frames through
``simulate(..., frame=True)``). Only storage and numerics differ:

* the state is kept as a contiguous mean vector and a Fortran-ordered covariance that the
  native kernels update in place, so a gate on k of n modes costs O(n k) instead of the
  O(n^3) dense ``S V S^T`` of the reference backend;
* outcomes are sampled in Python from natively computed moments with exactly the same
  ``numpy.random.Generator`` calls as :class:`GaussianBackend`, so equal seeds give equal
  trajectories up to floating-point summation order (observed agreement ~1e-13).

The module also exposes diagnostics (symplectic eigenvalues, purity, entropy, logarithmic
negativity, Uhlmann fidelity) and OpenMP-batched kernels as a functional API. There is no
batched trajectory runner: use ``simulate``/``run_shots`` with this backend.

Importing this module raises ``ImportError`` if the ``gaussian`` native component is not built.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence

import numpy as np

from .. import native
from .._checks import is_finite_real_scalar
from ..measurements import Generaldyne, Heterodyne, Homodyne
from ..states import GaussianInput, GaussianState
from .base import resolve_angle
from .gaussian import GaussianBackend

_kernels = native.require("gaussian")

__all__ = [
    "NativeGaussianBackend",
    "batch_apply_symplectic",
    "batch_condition_homodyne",
    "batch_homodyne_moments",
    "entropy",
    "fidelity",
    "log_negativity",
    "purity",
    "symplectic_eigenvalues",
]


class NativeGaussianBackend(GaussianBackend):
    """Drop-in :class:`GaussianBackend` with native in-place kernels (see module docstring)."""

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self._load(np.zeros(0), np.zeros((0, 0)), ())
        self.last_measurement: dict = {}

    # ----- storage -------------------------------------------------------------------
    def _load(self, mean, cov, nodes, symmetrise: bool = True) -> None:
        # Kernels assume an exactly symmetric covariance. External input is symmetrised
        # here (the reference does so on its next update); kernel outputs already are.
        cov = np.asarray(cov, dtype=float)
        self._mean = np.array(mean, dtype=float)  # never alias a caller's array
        self._cov = np.asfortranarray((cov + cov.T) / 2 if symmetrise else cov)
        self._nodes = tuple(nodes)
        self._index = {node: i for i, node in enumerate(self._nodes)}

    def _mode(self, node: Hashable) -> int:
        try:
            return self._index[node]
        except KeyError:
            raise ValueError(f"{node!r} is not a mode of the current state") from None

    @property
    def state(self) -> GaussianState:
        """Validated snapshot (O(n^3) physicality check); the backend keeps its own arrays."""
        return GaussianState(self._mean, self._cov, self._nodes)

    @state.setter
    def state(self, value: GaussianState) -> None:
        self._load(value.mean, value.covariance, value.nodes)

    def set_state(self, state):
        self.state = state.copy().validate()

    def get_state(self, nodes=None):
        state = self.state
        return state if nodes is None else state.reduced(nodes)

    # ----- preparation and Gaussian unitaries ----------------------------------------
    def prepare(self, node, squeezing=1.0, state=None):
        if node in self._index:
            raise ValueError("Node already prepared")
        if not np.isfinite(squeezing) or squeezing < 0:
            raise ValueError("Resource squeezing must be finite and nonnegative")
        if state is None:
            state = GaussianInput.squeezed(-squeezing)
        if not isinstance(state, GaussianInput):
            raise NotImplementedError("Gaussian backend requires GaussianInput")
        local = state.state(node)  # validates the single-mode input
        block = (local.covariance + local.covariance.T) / 2  # kernels need exact symmetry
        mean, cov = _kernels.append_mode(self._mean, self._cov, local.mean, block)
        self._load(mean, cov, self._nodes + (node,), symmetrise=False)

    def transform(self, nodes, matrix):
        if len(set(nodes)) != len(nodes):
            raise ValueError("Repeated gate modes")
        modes = [self._mode(n) for n in nodes]
        _kernels.apply_symplectic(self._mean, self._cov, modes, np.asarray(matrix, dtype=float))

    def displace(self, node, q=0.0, p=0.0):
        _kernels.displace(self._mean, self._mode(node), float(q), float(p))

    def loss(self, node, transmissivity, thermal_photons=0.0):
        _kernels.apply_loss(
            self._mean, self._cov, self._mode(node), float(transmissivity), float(thermal_photons)
        )

    # ----- measurements ----------------------------------------------------------------
    def measure(self, node, measurement, angle=None, *, outcome=None):
        """Destructive measurement with the contract of :meth:`GaussianBackend.measure`."""
        angle = resolve_angle(measurement, angle)
        postselected = outcome is not None
        mode = self._mode(node)
        if isinstance(measurement, Homodyne):
            eta, noise = measurement.efficiency, measurement.noise
            expected, variance = _kernels.homodyne_moments(
                self._mean, self._cov, mode, float(angle), eta, noise
            )
            if outcome is None:
                outcome = self.rng.normal(expected, np.sqrt(variance))
            elif not is_finite_real_scalar(outcome):
                raise ValueError("Homodyne outcome must be a finite real scalar")
            outcome = float(outcome)
            mean, cov, density = _kernels.condition_homodyne(
                self._mean, self._cov, mode, float(angle), eta, noise, outcome
            )
        elif isinstance(measurement, (Heterodyne, Generaldyne)):
            detector = (
                np.eye(2)
                if isinstance(measurement, Heterodyne)
                else np.asarray(measurement.covariance, dtype=float)
            )
            mu, total = _kernels.generaldyne_moments(self._mean, self._cov, mode, detector)
            if outcome is None:
                vector = self.rng.multivariate_normal(mu, total)
            else:
                vector = np.asarray(outcome)
                if (
                    vector.shape != (2,)
                    or not np.isrealobj(vector)
                    or not np.isfinite(vector).all()
                ):
                    raise ValueError("Phase-space outcome must be a finite real pair (q, p)")
                vector = vector.astype(float)
            mean, cov, density = _kernels.condition_generaldyne(
                self._mean, self._cov, mode, detector, vector
            )
            outcome = tuple(map(float, vector))
        else:
            raise NotImplementedError(
                "Photon counting has non-Gaussian conditional states; use native-fock, "
                "native-mixed-fock, piquasso-fock or piquasso-mixed-fock"
            )
        if postselected and not density > 0:
            raise ValueError("Postselected outcome has zero numerical density")
        self._load(mean, cov, tuple(n for n in self._nodes if n != node), symmetrise=False)
        self.last_measurement = {"kind": "density", "value": float(density)}
        return outcome


# ----- diagnostics (semantics of GaussianState methods) --------------------------------
def symplectic_eigenvalues(state: GaussianState) -> np.ndarray:
    """Sorted symplectic eigenvalues (Cholesky + Hermitian eigensolver); vacuum gives 1."""
    return np.asarray(_kernels.symplectic_eigenvalues(state.covariance))


def purity(state: GaussianState) -> float:
    """Tr(rho^2) = |det V|^(-1/2)."""
    return float(_kernels.purity(state.covariance))


def entropy(state: GaussianState, tolerance: float = 1e-7) -> float:
    """Von Neumann entropy in nats; raises ValueError below 1 - tolerance."""
    return float(_kernels.entropy(state.covariance, tolerance))


def log_negativity(state: GaussianState, nodes: Iterable[Hashable]) -> float:
    """Base-2 logarithmic negativity of the bipartition (nodes | rest)."""
    nodes = tuple(nodes)
    if len(set(nodes)) != len(nodes) or not set(nodes) <= set(state.nodes):
        raise ValueError("Provide unique labels of this state")
    modes = [state.nodes.index(n) for n in nodes]
    return float(_kernels.log_negativity(state.covariance, modes))


def fidelity(state: GaussianState, other: GaussianState, pure_tolerance: float = 1e-9) -> float:
    """Squared Uhlmann fidelity; ``other`` is reordered to ``state``'s labels."""
    if set(state.nodes) != set(other.nodes):
        raise ValueError("State node sets must agree")
    other = other.reduced(state.nodes)
    return float(
        _kernels.fidelity(
            state.mean, state.covariance, other.mean, other.covariance, pure_tolerance
        )
    )


# ----- batched functional API ----------------------------------------------------------
def _batch(means, covariances, copy: bool) -> tuple[np.ndarray, np.ndarray, int, int]:
    # C-ordered arrays; private copies when the kernel updates them in place. The kernels use
    # each covariance's symmetric part (V + V^T)/2, which is exact for these linear updates.
    convert = np.array if copy else np.ascontiguousarray
    means = convert(means, dtype=float)
    covs = convert(covariances, dtype=float)
    if means.ndim != 2 or covs.shape != (means.shape[0], means.shape[1], means.shape[1]):
        raise ValueError("Expected means (B, 2n) and covariances (B, 2n, 2n)")
    batch, dim = means.shape
    return means, covs, batch, dim


def batch_apply_symplectic(
    means, covariances, modes: Sequence[int], matrices
) -> tuple[np.ndarray, np.ndarray]:
    """Apply S_b to the listed mode indices of every state b; returns new (means, covs).

    Args:
        means: Shape (B, 2n) interleaved means.
        covariances: Shape (B, 2n, 2n) statistical covariances (symmetric part used).
        modes: Mode indices (not labels) the 2k x 2k matrices act on, in order.
        matrices: Shape (B, 2k, 2k) symplectic matrices, or one (2k, 2k) for all states.

    Raises:
        ValueError: Shape mismatch, repeated/out-of-range modes or a non-symplectic matrix.
    """
    means, covs, batch, dim = _batch(means, covariances, copy=True)
    k = 2 * len(modes)
    mats = np.broadcast_to(np.asarray(matrices, dtype=float), (batch, k, k))
    flat = covs.reshape(batch, dim * dim)
    _kernels.batch_apply_symplectic(
        means, flat, list(modes), np.ascontiguousarray(mats).reshape(batch, k * k)
    )
    return means, flat.reshape(batch, dim, dim)


def batch_homodyne_moments(
    means, covariances, mode: int, angles, efficiency: float = 1.0, noise: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Per-state (expected, variance) of the calibrated homodyne record on mode index ``mode``."""
    means, covs, batch, dim = _batch(means, covariances, copy=False)
    angles = np.broadcast_to(np.asarray(angles, dtype=float), (batch,))
    return _kernels.batch_homodyne_moments(
        means, covs.reshape(batch, dim * dim), mode, angles, efficiency, noise
    )


def batch_condition_homodyne(
    means, covariances, mode: int, angles, outcomes, efficiency: float = 1.0, noise: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Condition every state on its GIVEN homodyne outcome; returns (means, covs, densities).

    The densities are outcome densities given each pre-measurement state; a batch conditioned
    on chosen outcomes is a set of postselected branches, not a deterministic operation.
    """
    means, covs, batch, dim = _batch(means, covariances, copy=False)
    angles = np.broadcast_to(np.asarray(angles, dtype=float), (batch,))
    outcomes = np.broadcast_to(np.asarray(outcomes, dtype=float), (batch,))
    out_means, out_covs, densities = _kernels.batch_condition_homodyne(
        means, covs.reshape(batch, dim * dim), mode, angles, efficiency, noise, outcomes
    )
    return out_means, out_covs.reshape(batch, dim - 2, dim - 2), densities
