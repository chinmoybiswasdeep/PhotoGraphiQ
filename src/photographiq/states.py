"""Backend-neutral Gaussian states and input preparation descriptions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gaussian import omega, rotation


@dataclass
class GaussianState:
    mean: np.ndarray
    covariance: np.ndarray
    nodes: tuple = ()

    def __post_init__(self):
        self.mean = np.array(self.mean, dtype=float, copy=True)
        self.covariance = np.array(self.covariance, dtype=float, copy=True)
        self.nodes = tuple(self.nodes)
        self.validate()

    def validate(self):
        n = self.mean.size
        if self.mean.shape != (n,) or n % 2 or self.covariance.shape != (n, n):
            raise ValueError("Invalid Gaussian moment dimensions")
        if len(self.nodes) != n // 2 or len(set(self.nodes)) != len(self.nodes):
            raise ValueError("Provide one unique label per mode")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.covariance).all():
            raise ValueError("Moments must be finite")
        if not np.allclose(self.covariance, self.covariance.T, atol=1e-10, rtol=0):
            raise ValueError("Covariance must be symmetric")
        if n and np.linalg.eigvalsh(self.covariance + 1j * omega(n // 2)).min() < -1e-7:
            raise ValueError("Covariance violates the uncertainty relation V+i Omega >= 0")
        return self

    def copy(self):
        return GaussianState(self.mean, self.covariance, self.nodes)

    def reduced(self, nodes):
        nodes = tuple(nodes)
        indices = [
            j
            for node in nodes
            for j in (2 * self.nodes.index(node), 2 * self.nodes.index(node) + 1)
        ]
        return GaussianState(self.mean[indices], self.covariance[np.ix_(indices, indices)], nodes)

    def quadrature(self, node, angle=0.0):
        i = self.nodes.index(node)
        v = np.array([np.cos(angle), np.sin(angle)])
        return float(v @ self.mean[2 * i : 2 * i + 2]), float(
            v @ self.covariance[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] @ v
        )

    def photon_number(self, node):
        state = self.reduced((node,))
        return float((np.trace(state.covariance) + state.mean @ state.mean - 2) / 4)

    def parity(self, nodes=None):
        state = self if nodes is None else self.reduced(nodes)
        return float(
            np.exp(-0.5 * state.mean @ np.linalg.solve(state.covariance, state.mean))
            / np.sqrt(np.linalg.det(state.covariance))
        )

    def overlap(self, other):
        """Tr(rho sigma); equals fidelity if at least one state is pure."""
        if self.nodes != other.nodes:
            raise ValueError("State labels/order must agree")
        cov = self.covariance + other.covariance
        delta = self.mean - other.mean
        return float(
            2 ** len(self.nodes)
            * np.exp(-0.5 * delta @ np.linalg.solve(cov, delta))
            / np.sqrt(np.linalg.det(cov))
        )


@dataclass(frozen=True)
class GaussianInput:
    mean: tuple = (0.0, 0.0)
    covariance: tuple = ((1.0, 0.0), (0.0, 1.0))

    def state(self, node):
        return GaussianState(np.asarray(self.mean), np.asarray(self.covariance), (node,))

    @classmethod
    def coherent(cls, alpha: complex):
        return cls((2 * complex(alpha).real, 2 * complex(alpha).imag))

    @classmethod
    def squeezed(cls, r: float, angle: float = 0.0):
        rot = rotation(angle)
        cov = rot @ np.diag([np.exp(-2 * r), np.exp(2 * r)]) @ rot.T
        return cls(covariance=tuple(map(tuple, cov)))


@dataclass(frozen=True)
class FockInput:
    """Single-mode normalized amplitudes, from vacuum up to an explicit cutoff."""

    amplitudes: tuple[complex, ...]

    def __post_init__(self):
        a = np.asarray(self.amplitudes, dtype=complex)
        if (
            a.ndim != 1
            or not len(a)
            or not np.isfinite(a).all()
            or not np.isclose(np.vdot(a, a), 1)
        ):
            raise ValueError("Fock input amplitudes must be finite and normalized")

    @classmethod
    def number(cls, n: int):
        if not isinstance(n, int) or n < 0:
            raise ValueError("Photon number must be a nonnegative integer")
        return cls((0j,) * n + (1 + 0j,))

    @classmethod
    def cat(cls, alpha: complex, cutoff: int, parity: int = 1):
        import math

        if cutoff < 1 or parity not in (-1, 1):
            raise ValueError("Invalid cutoff or parity")
        a = np.array(
            [
                (alpha**n + parity * (-alpha) ** n) / np.sqrt(float(math.factorial(n)))
                for n in range(cutoff)
            ],
            dtype=complex,
        )
        norm = np.linalg.norm(a)
        if norm == 0:
            raise ValueError("Zero cat vector")
        return cls(tuple(a / norm))

    def photon_added(self):
        """Normalized offline a-dagger resource; no heralding probability claim."""
        a = np.r_[0j, np.sqrt(np.arange(1, len(self.amplitudes) + 1)) * self.amplitudes]
        return FockInput(tuple(a / np.linalg.norm(a)))

    def photon_subtracted(self):
        """Normalized offline a resource; subtraction from vacuum is impossible."""
        a = np.sqrt(np.arange(1, len(self.amplitudes))) * self.amplitudes[1:]
        if np.linalg.norm(a) == 0:
            raise ValueError("Photon subtraction has zero norm")
        return FockInput(tuple(a / np.linalg.norm(a)))
