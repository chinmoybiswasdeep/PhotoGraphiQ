"""Independent phase-space mathematics: hbar=2 and statistical covariance."""

from __future__ import annotations

import numpy as np

HBAR = 2.0


def omega(modes: int) -> np.ndarray:
    return np.kron(np.eye(modes), [[0.0, 1.0], [-1.0, 0.0]])


def rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def squeezing(r: float) -> np.ndarray:
    """Positive r squeezes q for a gate; resource preparation uses -r."""
    return np.diag([np.exp(-r), np.exp(r)])


def cz(weight: float) -> np.ndarray:
    result = np.eye(4)
    result[1, 2] = result[3, 0] = weight
    return result


def beamsplitter(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, -s, 0], [0, c, 0, -s], [s, 0, c, 0], [0, s, 0, c]])


def embed(matrix, indices, size):
    result = np.eye(size)
    result[np.ix_(indices, indices)] = matrix
    return result


def teleportation_matrix(k: float) -> np.ndarray:
    return np.array([[-k, -1.0], [1.0, 0.0]])


def wire_channel(shears, squeezing_values):
    """Unconditional Gaussian channel (S,N), not a conditional trajectory."""
    shears = list(shears)
    rs = np.asarray(
        [squeezing_values] * len(shears)
        if np.isscalar(squeezing_values)
        else list(squeezing_values),
        dtype=float,
    )
    if len(rs) != len(shears) or not np.all(np.isfinite(rs)) or np.any(np.array(rs) < 0):
        raise ValueError("Supply one finite nonnegative resource squeezing per step")
    total, noise = np.eye(2), np.zeros((2, 2))
    for k, r in zip(shears, rs):
        step = teleportation_matrix(k)
        total = step @ total
        noise = step @ noise @ step.T + np.diag([0.0, np.exp(-2 * r)])
    return total, noise
