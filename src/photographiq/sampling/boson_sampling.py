"""Fock-state boson sampling through a passive interferometer.

With the PhotoGraphiQ convention ``a_j^dag -> sum_k U_kj a_k^dag`` (see
:mod:`photographiq.sampling.unitaries`), an input occupation ``s`` reaches the output
occupation ``t`` with amplitude ``Perm(U_{t,s}) / sqrt(prod s_i! prod t_j!)``, where
``U_{t,s}`` repeats row ``k`` of ``U`` ``t_k`` times and column ``j`` ``s_j`` times. The
permanent is evaluated with these multiplicities directly (no repeated matrix is built).
Photon number is conserved: ``P(t|s) = 0`` unless ``sum t = sum s``.

Samplers:

* ``method="clifford"``: Clifford & Clifford (2018), algorithm B, exact for inputs
  with at most one photon per mode. Cost ``O(n sum_k k 2^k)`` permanent work for ``n``
  photons here (each ``(k-1)``-minor permanent is evaluated separately rather than by the
  paper's single Laplace pass), plus ``O(m n^2)``.
* ``method="enumerate"``: exact sampling from the fully enumerated distribution over all
  ``C(n+m-1, n)`` output patterns; for tiny systems and any input (including collisions).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Literal

import numpy as np

from .linalg import Backend, permanent
from .unitaries import check_unitary

Occupation = Sequence[int] | np.ndarray


def _occupation(pattern: Occupation, modes: int, name: str) -> np.ndarray:
    result = np.asarray(pattern)
    if (
        result.shape != (modes,)
        or not np.issubdtype(result.dtype, np.integer)
        or (result < 0).any()
    ):
        raise ValueError(f"{name} must be {modes} nonnegative integers")
    return result.astype(np.int64)


def transition_amplitude(
    unitary, inputs: Occupation, outputs: Occupation, *, backend: Backend = "auto"
) -> complex:
    """``<t| W(U) |s> = Perm(U_{t,s}) / sqrt(prod s! prod t!)``; zero if photon numbers differ."""
    u = check_unitary(unitary)
    s = _occupation(inputs, len(u), "inputs")
    t = _occupation(outputs, len(u), "outputs")
    if s.sum() != t.sum():
        return 0.0j
    norm = math.prod(math.factorial(int(x)) for x in np.r_[s, t])
    return permanent(u, rows=t, columns=s, backend=backend) / math.sqrt(norm)


def output_probability(
    unitary, inputs: Occupation, outputs: Occupation, *, backend: Backend = "auto"
) -> float:
    """``P(t|s) = |Perm(U_{t,s})|^2 / (prod s_i! prod t_j!)``."""
    return abs(transition_amplitude(unitary, inputs, outputs, backend=backend)) ** 2


def output_patterns(modes: int, photons: int) -> list[tuple[int, ...]]:
    """All occupation tuples of ``photons`` bosons in ``modes`` modes (lexicographic)."""
    patterns = []
    for bars in itertools.combinations(range(photons + modes - 1), modes - 1):
        edges = (-1,) + bars + (photons + modes - 1,)
        patterns.append(tuple(edges[k + 1] - edges[k] - 1 for k in range(modes)))
    return sorted(patterns)


def output_distribution(
    unitary, inputs: Occupation, *, backend: Backend = "auto"
) -> dict[tuple[int, ...], float]:
    """Exact output distribution over all ``C(n+m-1, n)`` patterns.

    Raises:
        ValueError: The probabilities do not sum to 1 within ``1e-9`` (numerical failure).
    """
    u = check_unitary(unitary)
    s = _occupation(inputs, len(u), "inputs")
    dist = {
        t: output_probability(u, s, t, backend=backend)
        for t in output_patterns(len(u), int(s.sum()))
    }
    total = math.fsum(dist.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Boson-sampling distribution sums to {total!r}, not 1")
    return dist


def is_collision_free(pattern: Occupation) -> bool:
    """True if no mode holds more than one photon."""
    return max(pattern, default=0) <= 1


def collision_probabilities(distribution: dict[tuple[int, ...], float]) -> dict[str, float]:
    """Split an output distribution into collision-free and collision (bunched) mass."""
    free = math.fsum(p for t, p in distribution.items() if is_collision_free(t))
    return {"collision_free": free, "collision": math.fsum(distribution.values()) - free}


def sample(
    unitary,
    inputs: Occupation,
    shots: int,
    *,
    rng: np.random.Generator | int | None = None,
    method: Literal["auto", "clifford", "enumerate"] = "auto",
    backend: Backend = "auto",
) -> np.ndarray:
    """Exact boson-sampling outputs, shape ``(shots, modes)`` of occupations.

    ``method="auto"`` uses Clifford & Clifford for collision-free inputs and full
    enumeration otherwise.

    Raises:
        ValueError: ``"clifford"`` requested for an input with more than one photon per mode.
    """
    u = check_unitary(unitary)
    s = _occupation(inputs, len(u), "inputs")
    rng = np.random.default_rng(rng)
    if method == "auto":
        method = "clifford" if is_collision_free(s) else "enumerate"
    if method == "enumerate":
        dist = output_distribution(u, s, backend=backend)
        patterns = np.array(list(dist), dtype=np.int64).reshape(len(dist), len(u))
        p = np.fromiter(dist.values(), float)
        return patterns[rng.choice(len(p), size=shots, p=p / p.sum())]
    if method != "clifford":
        raise ValueError(f"Unknown method {method!r}")
    if not is_collision_free(s):
        raise ValueError("Clifford & Clifford sampling here requires at most one photon per mode")
    columns = u[:, np.flatnonzero(s)]
    out = np.zeros((shots, len(u)), dtype=np.int64)
    for shot in range(shots):
        for mode in _clifford_shot(columns, rng, backend):
            out[shot, mode] += 1
    return out


def _clifford_shot(columns: np.ndarray, rng: np.random.Generator, backend: Backend) -> list[int]:
    """One sample of Clifford & Clifford algorithm B for an ``m x n`` isometry ``columns``."""
    m, n = columns.shape
    if n == 0:
        return []
    a = columns[:, rng.permutation(n)]
    weights = np.abs(a[:, 0]) ** 2
    rows = [int(rng.choice(m, p=weights / weights.sum()))]
    for k in range(2, n + 1):
        # Perm of B_k with column l removed, B_k = a[rows, :k] ((k-1) x k).
        minors = np.array(
            [
                permanent(np.delete(a[np.ix_(rows, range(k))], col, axis=1), backend=backend)
                for col in range(k)
            ]
        )
        weights = np.abs(a[:, :k] @ minors) ** 2
        rows.append(int(rng.choice(m, p=weights / weights.sum())))
    return rows
