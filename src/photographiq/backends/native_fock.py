"""Native C++ Fock engines: drop-in replacements of the Piquasso-backed Fock backends.

``NativeFockBackend`` (pure states) and ``NativeMixedFockBackend`` (density matrices)
reuse every Python code path of :class:`PiquassoFockBackend` and
:class:`MixedFockBackend` (preparation, renormalisation with retained-norm diagnostics,
photon counting, homodyne, instruments, GKP readout, ladder operations) and replace only
the numerical engine: the state object ``self.native`` and the gate kernels. Piquasso is
never imported.

Conventions (docs/getting-started/conventions.md): hbar = 2, q = a + a^dag,
p = -i (a - a^dag), interleaved (q, p) per mode; truncation keeps occupations with TOTAL
photon number < cutoff, in Piquasso's basis order (sectors of total photon number, first
mode descending inside a sector).

Numerics and their relation to Piquasso:

* Gaussian gates (Displace, Squeeze, Entangle/CZ, QuadraticPhase, BeamSplitter, and
  Gaussian inputs) use EXACT Fock matrix elements of the full Gaussian unitary from the
  Bargmann recurrence, applied as P U P on the truncated space. Piquasso computes
  displacement and squeezing the same way, but applies CZ, QuadraticPhase and general
  Gaussian transforms through an Euler (Bloch-Messiah) decomposition with a truncation
  after every factor, and prepares Gaussian inputs as P D P R P S|0>. Both converge to the
  same state as the cutoff grows; at a fixed cutoff they differ at truncation level.
* Rounding at large cutoffs: the recurrence pivots on the most occupied mode of the side
  with more photons. Displace (S = 1) uses the Laguerre recurrence along each diagonal.
  Measured against expm and 60-digit references, the error is 1e-15 for |alpha| <= 7 at
  cutoffs <= 120 (displacement), and 8e-14 for r = 0.5 at cutoff 80 and 1.4e-12 for r = 1
  at cutoff 120 (squeezing). The earlier bra-first pivot, and Piquasso 8.0.1's
  displacement and squeezing matrices, amplify rounding at the same sizes. Piquasso
  measured: 1.3e-3 for |alpha| = 4 at cutoff 40, 5.9e-2 for |alpha| = 3 at cutoff 60, and
  1.3e-3 for r = 1 at cutoff 100. A Piquasso mismatch at such sizes is Piquasso's
  rounding, not truncation. One case is still unresolved: a single Gaussian unitary that
  combines squeezing and displacement loses accuracy at large |alpha| and cutoff
  (8e-5 for |alpha| = 4, r = 0.3, cutoff 80). Its vacuum column, which is all a pure
  ``GaussianInput`` uses, stays at 1e-16.
* Gate matrices are defined up to a global phase (the vacuum amplitude of the
  displacement-free part is real positive); no observable depends on it.
* Rotation, Kerr exp(i kappa n^2) and cross-Kerr exp(i kappa n_u n_v) are exact phases.
* CubicPhase exp(i gamma q^3/6) has two definitions, both converging to the exact gate:
  ``cubic_phase="truncated_generator"`` (default) is Piquasso's expm(i gamma Q^3/6) with
  Q = P q P the cutoff x cutoff position matrix (identical to Piquasso to 1e-13, so the
  backend is a drop-in). It is unitary on the truncated space, so it never reports
  leakage: it equals the Gauss-Hermite (DVR) quadrature of exp(i gamma q^3/6) at the
  cutoff nodes. ``cubic_phase="projected"`` uses the exact projected elements <m|V|n>,
  m, n < cutoff, by a converged nested trapezoidal rule (tolerance ``cubic_tolerance``),
  so leakage above the cutoff appears in the retained-norm diagnostics (and can raise the
  norm-tolerance error where Piquasso silently continues). Measured on V|0>, gamma = 0.8:
  max relative error of <p^2>, <p^3>, <p^4> at cutoffs 80/160/320 is 8.9e-3/2.8e-4/1.4e-6
  (projected) vs 5.0e-3/1.2e-4/4.2e-7 (truncated generator); for gamma = 0.2 both reach
  1e-13 at cutoff 80. The projected state has the smaller infidelity at a fixed cutoff
  (1.6e-3 vs 3.2e-3 at cutoff 16, gamma = 0.5, tests/native).
* Mixed Gaussian inputs: rho = U (sum_k p_k |k><k|) U^dag with U = D R S from the
  Williamson form and thermal weights p_k summed until the omitted tail is below 1e-15
  (trace norm); Piquasso evaluates the Gaussian density matrix directly.
* Thermal loss uses the same Kraus decomposition as ``MixedFockBackend.loss``.

Cross-Kerr has no Pattern command; it is available as the backend method
:meth:`cross_kerr`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from types import ModuleType
from typing import Any

import numpy as np

from .. import native
from ..states import GaussianInput
from .fock import PiquassoFockBackend
from .mixed_fock import MixedFockBackend

_THERMAL_TAIL = 1e-15
_MAX_THERMAL_TERMS = 4096


def _engine() -> ModuleType:
    return native.require("fock")


@lru_cache(maxsize=64)
def _occupations(modes: int, cutoff: int) -> np.ndarray:
    table = _engine().basis(modes, cutoff)
    table.setflags(write=False)
    return table


@lru_cache(maxsize=64)
def _keys(modes: int, cutoff: int) -> tuple[tuple[int, ...], ...]:
    return tuple(map(tuple, _occupations(modes, cutoff).tolist()))


@lru_cache(maxsize=64)
def _totals(modes: int, cutoff: int) -> np.ndarray:
    totals = _occupations(modes, cutoff).sum(axis=1)
    totals.setflags(write=False)
    return totals


@lru_cache(maxsize=32)
def _cubic_phase_matrix(gamma: float, cutoff: int, method: str, tolerance: float) -> np.ndarray:
    """Cached cubic-phase matrix (measured: ~1 ms at cutoff 30, ~1 s at cutoff 100, gamma 1)."""
    if method == "projected":
        matrix = _engine().cubic_phase_projected(gamma, cutoff, tolerance)[0]
    else:
        matrix = _engine().cubic_phase_truncated_generator(gamma, cutoff)
    matrix.setflags(write=False)
    return matrix


def _ranks(basis, modes: int, cutoff: int) -> np.ndarray:
    occupations = np.asarray([tuple(int(n) for n in b) for b in basis], dtype=np.int32)
    return _engine().rank(occupations.reshape(len(basis), modes), cutoff)


class _NativeFockState:
    """Shared part of the native state objects (data in Piquasso basis order)."""

    _probabilities: Callable[[], np.ndarray]  # defined by each subclass

    def __init__(self, data: np.ndarray, modes: int, cutoff: int):
        self._data = np.ascontiguousarray(data, dtype=complex)
        self.d = modes
        self.cutoff = cutoff

    def copy(self):
        return type(self)(self._data.copy(), self.d, self.cutoff)

    @property
    def fock_probabilities_map(self) -> dict:
        return dict(zip(_keys(self.d, self.cutoff), self._probabilities().tolist(), strict=True))

    def photon_marginal(self, modes) -> np.ndarray:
        """Joint photon-number probabilities of ordered modes, in their local basis order."""
        kind = "pure" if isinstance(self, NativePureFockState) else "mixed"
        return getattr(_engine(), f"marginal_{kind}")(self._data, self.d, self.cutoff, list(modes))

    def boundary_population(self, threshold: int) -> float:
        totals = _totals(self.d, self.cutoff)
        return float(self._probabilities()[totals >= threshold].sum())


class NativePureFockState(_NativeFockState):
    """Pure truncated Fock state with the Piquasso PureFockState interface used here."""

    @property
    def state_vector(self) -> np.ndarray:
        return self._data.copy()

    @property
    def fock_amplitudes_map(self) -> dict:
        return dict(zip(_keys(self.d, self.cutoff), self._data.tolist(), strict=True))

    @property
    def density_matrix(self) -> np.ndarray:
        return np.outer(self._data, self._data.conj())

    @property
    def norm(self) -> float:
        return float(np.vdot(self._data, self._data).real)

    def normalize(self) -> None:
        self._data /= np.sqrt(self.norm)

    def _probabilities(self) -> np.ndarray:
        return np.abs(self._data) ** 2

    def reduced(self, modes) -> NativeMixedFockState:
        matrix = _engine().reduced_pure(self._data, self.d, self.cutoff, list(modes))
        return NativeMixedFockState(matrix, len(modes), self.cutoff)

    def apply(self, modes, op: np.ndarray) -> None:
        _engine().apply_pure(self._data, self.d, self.cutoff, list(modes), op)

    def phase(self, modes, phases: np.ndarray) -> None:
        _engine().phase_pure(self._data, self.d, self.cutoff, list(modes), phases)


class NativeMixedFockState(_NativeFockState):
    """Density matrix with the Piquasso FockState interface; it has no state_vector."""

    @property
    def density_matrix(self) -> np.ndarray:
        return self._data.copy()

    @property
    def norm(self) -> float:
        return float(np.trace(self._data).real)

    def normalize(self) -> None:
        self._data /= self.norm

    def _probabilities(self) -> np.ndarray:
        return self._data.diagonal().real.copy()

    def reduced(self, modes) -> NativeMixedFockState:
        matrix = _engine().reduced_mixed(self._data, self.d, self.cutoff, list(modes))
        return NativeMixedFockState(matrix, len(modes), self.cutoff)

    def apply(self, modes, op: np.ndarray) -> None:
        _engine().apply_mixed(self._data, self.d, self.cutoff, list(modes), op)

    def phase(self, modes, phases: np.ndarray) -> None:
        _engine().phase_mixed(self._data, self.d, self.cutoff, list(modes), phases)

    def thermal_loss(self, mode: int, transmissivity: float, thermal_photons: float) -> None:
        self._data = _engine().thermal_loss(
            self._data, self.d, self.cutoff, mode, transmissivity, thermal_photons
        )

    def physicality(self) -> dict:
        """Hermiticity residual, trace and extreme eigenvalues of the density matrix."""
        return _engine().physicality(self._data)


def _single_mode_gaussian(state: GaussianInput | None, squeezing: float):
    """(S, d, n_th) with V = (2 n_th + 1) S S^T, mean d; None means p-squeezed vacuum."""
    if state is None:
        return np.diag([np.exp(squeezing), np.exp(-squeezing)]), np.zeros(2), 0.0
    cov = np.array(state.covariance, dtype=float)
    nu = float(np.sqrt(np.linalg.det(cov)))
    values, vectors = np.linalg.eigh(cov / nu)
    r = -0.5 * np.log(values[0])
    phi = np.arctan2(vectors[1, 0], vectors[0, 0])
    rotation = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    symplectic = rotation @ np.diag([np.exp(-r), np.exp(r)])
    return symplectic, np.array(state.mean, dtype=float), max(0.0, (nu - 1) / 2)


class _NativeGates:
    """Gate methods shared by the native pure and mixed backends (operate on self.native)."""

    # Provided by the PiquassoFockBackend / MixedFockBackend this mixin is combined with.
    cutoff: int
    nodes: tuple
    native: Any
    _check: Callable[..., Any]
    _guard: Callable[[int], None]

    capabilities = PiquassoFockBackend.capabilities | frozenset({"cross_kerr"})

    def _init_native(self, cubic_phase: str, cubic_tolerance: float) -> None:
        _engine()  # raise ImportError now if the extension is unavailable
        if cubic_phase not in ("projected", "truncated_generator"):
            raise ValueError("cubic_phase must be 'projected' or 'truncated_generator'")
        if not 0 < cubic_tolerance < 1e-6:
            raise ValueError("cubic_tolerance must be in (0, 1e-6)")
        self.cubic_phase_method, self.cubic_tolerance = cubic_phase, cubic_tolerance

    def _boundary_population(self, state):
        return state.boundary_population(max(1, self.cutoff - 2)), state._probabilities().size

    def _modes(self, nodes) -> list[int]:
        if len(set(nodes)) != len(nodes):
            raise ValueError("Repeated gate mode")
        # Same per-gate guard as the parent _gate: MixedFockBackend checks, the pure one not.
        if isinstance(self, MixedFockBackend):
            self._guard(len(self.nodes))
        return [self.nodes.index(n) for n in nodes]

    def _apply(self, nodes, op: np.ndarray, operation: str) -> None:
        # Evolve a copy so a failed norm check leaves the pre-gate state, as with Piquasso.
        modes, state = self._modes(nodes), self.native.copy()
        state.apply(modes, op)
        self.native = self._check(state, operation)

    def _phase(self, nodes, phases: np.ndarray, operation: str) -> None:
        modes, state = self._modes(nodes), self.native.copy()
        state.phase(modes, np.asarray(phases, dtype=float))
        self.native = self._check(state, operation)

    def _gaussian(self, nodes, symplectic, displacement, operation: str) -> None:
        op = _engine().gaussian_unitary(
            np.asarray(symplectic, dtype=float), np.asarray(displacement, dtype=float), self.cutoff
        )
        self._apply(nodes, op, operation)

    def _local_numbers(self, modes: int) -> np.ndarray:
        return _occupations(modes, self.cutoff).astype(float)

    def entangle(self, u, v, weight=1.0):
        """Weighted CZ exp(i g q_u q_v / 2): p_u += g q_v, p_v += g q_u (exact elements)."""
        symplectic = np.eye(4)
        symplectic[1, 2] = symplectic[3, 0] = weight
        self._gaussian((u, v), symplectic, np.zeros(4), "GaussianTransform")

    def displace(self, node, q=0.0, p=0.0):
        """Quadrature translation by (q, p) in hbar = 2 units, alpha = (q + i p) / 2."""
        self._gaussian((node,), np.eye(2), [q, p], "Displacement")

    def rotate(self, node, angle):
        """exp(i angle n): q -> q cos(angle) - p sin(angle)."""
        self._phase((node,), angle * self._local_numbers(1)[:, 0], "Phaseshifter")

    def squeeze(self, node, r):
        """S = diag(e^-r, e^r) on (q, p); r > 0 squeezes q."""
        self._gaussian((node,), np.diag([np.exp(-r), np.exp(r)]), np.zeros(2), "Squeezing")

    def beamsplitter(self, u, v, theta):
        """q_u' = c q_u - s q_v, q_v' = s q_u + c q_v (same for p); exact per sector."""
        c, s = np.cos(theta), np.sin(theta)
        symplectic = np.kron(np.array([[c, -s], [s, c]]), np.eye(2))
        self._gaussian((u, v), symplectic, np.zeros(4), "Beamsplitter")

    def quadratic_phase(self, node, s):
        """exp(i s q^2 / 4): p -> p + s q (exact elements, up to a global phase)."""
        self._gaussian((node,), np.array([[1.0, 0.0], [s, 1.0]]), np.zeros(2), "QuadraticPhase")

    def kerr(self, node, kappa):
        """exp(i kappa n^2)."""
        self._phase((node,), kappa * self._local_numbers(1)[:, 0] ** 2, "Kerr")

    def cross_kerr(self, u, v, kappa):
        """exp(i kappa n_u n_v); backend method only (there is no Pattern command)."""
        numbers = self._local_numbers(2)
        self._phase((u, v), kappa * numbers[:, 0] * numbers[:, 1], "CrossKerr")

    def cubic_phase(self, node, gamma):
        """exp(i gamma q^3 / 6); see the module docstring for the two matrix definitions."""
        if not np.isfinite(gamma):
            raise ValueError("Cubic-phase gamma must be finite")
        op = _cubic_phase_matrix(
            float(gamma), self.cutoff, self.cubic_phase_method, self.cubic_tolerance
        )
        self._apply((node,), op, "CubicPhase")

    def _gate(self, nodes, instruction):
        raise TypeError("Native Fock backends do not execute Piquasso instructions")


class NativeFockBackend(_NativeGates, PiquassoFockBackend):
    """Pure-state Fock backend on the native C++ engine (same semantics and diagnostics).

    Args:
        cutoff (int): Exclusive total-photon cutoff (>= 2).
        cubic_phase (str): ``"truncated_generator"`` (default, Piquasso's expm of the
            truncated generator) or ``"projected"`` (exact P V P, reports leakage); see
            the module docstring.
        cubic_tolerance (float): Convergence tolerance of the cubic-phase quadrature.
        **kwargs (dict): ``norm_tolerance``, ``max_dimension``, ``boundary_warning``.

    Raises:
        ImportError: The native ``fock`` component is not compiled.
    """

    _registry_name = "native-fock"

    def __init__(
        self, cutoff=None, *, cubic_phase="truncated_generator", cubic_tolerance=1e-12, **kwargs
    ):
        self._init_native(cubic_phase, cubic_tolerance)
        super().__init__(cutoff, **kwargs)

    def _vector(self, amplitudes, modes):
        self._guard(modes)
        vector = np.zeros(self.dimension(modes), dtype=complex)
        if amplitudes:
            vector[_ranks(amplitudes, modes, self.cutoff)] = list(amplitudes.values())
        return NativePureFockState(vector, modes, self.cutoff)

    def _gaussian_vector(self, squeezing, state):
        symplectic, mean, _ = _single_mode_gaussian(state, squeezing)
        column = _engine().gaussian_unitary(symplectic, mean, self.cutoff)[:, 0]
        return NativePureFockState(column, 1, self.cutoff)


class NativeMixedFockBackend(_NativeGates, MixedFockBackend):
    """Density-matrix Fock backend on the native C++ engine (same semantics and diagnostics).

    Args:
        cutoff (int): Exclusive total-photon cutoff (>= 2).
        cubic_phase (str): See :class:`NativeFockBackend`.
        cubic_tolerance (float): Convergence tolerance of the cubic-phase quadrature.
        **kwargs (dict): ``max_matrix_bytes`` and the pure backend settings.
    """

    _registry_name = "native-mixed-fock"
    capabilities = MixedFockBackend.capabilities | frozenset({"cross_kerr"})

    def __init__(
        self, cutoff=None, *, cubic_phase="truncated_generator", cubic_tolerance=1e-12, **kwargs
    ):
        self._init_native(cubic_phase, cubic_tolerance)
        super().__init__(cutoff, **kwargs)

    def _density(self, matrix, basis):
        modes = len(basis[0])
        self._guard(modes)
        dimension = self.dimension(modes)
        ranks = _ranks(basis, modes, self.cutoff)
        full = np.zeros((dimension, dimension), dtype=complex)
        full[np.ix_(ranks, ranks)] = matrix
        return NativeMixedFockState(full, modes, self.cutoff)

    def _pure_helper(self):
        return NativeFockBackend(
            self.cutoff,
            norm_tolerance=self.norm_tolerance,
            cubic_phase=self.cubic_phase_method,
            cubic_tolerance=self.cubic_tolerance,
        )

    def _gaussian_density(self, state):
        symplectic, mean, thermal = _single_mode_gaussian(state, 0.0)
        ratio = thermal / (1 + thermal)
        terms = 1 if ratio == 0 else int(np.ceil(np.log(_THERMAL_TAIL) / np.log(ratio)))
        size = max(self.cutoff, terms)
        if size > _MAX_THERMAL_TERMS:
            raise NotImplementedError(
                f"Thermal occupation {thermal:.3g} needs {terms} Fock terms (> {_MAX_THERMAL_TERMS})"
            )
        columns = _engine().gaussian_unitary(symplectic, mean, size)[: self.cutoff, :terms]
        weights = (1 - ratio) * ratio ** np.arange(terms)
        return (columns * weights) @ columns.conj().T, tuple((n,) for n in range(self.cutoff))

    def loss(self, node, transmissivity, thermal_photons=0.0):
        """Thermal attenuator (Kraus decomposition of MixedFockBackend.loss, in C++).

        Raises:
            ValueError: Loss requires 0 <= transmissivity <= 1 and thermal_photons >= 0.
        """
        if (
            not np.isfinite([transmissivity, thermal_photons]).all()
            or not 0 <= transmissivity <= 1
            or thermal_photons < 0
        ):
            raise ValueError("Loss requires 0 <= transmissivity <= 1 and thermal_photons >= 0")
        state = self.native.copy()
        state.thermal_loss(self.nodes.index(node), float(transmissivity), float(thermal_photons))
        self.native = self._check(state, "ThermalLoss")
