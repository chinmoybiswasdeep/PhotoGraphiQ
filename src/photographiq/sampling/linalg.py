"""Permanent, hafnian, loop hafnian, torontonian and loop torontonian.

Each public function dispatches to the compiled ``photographiq._native.sampling`` kernels
(``backend="native"``), to the NumPy implementations in this module (``"python"``), or
to the native kernels when compiled and NumPy otherwise (``"auto"``, the default). Both
implementations use the same algorithms:

* permanent: Ryser's inclusion-exclusion formula with row/column multiplicities,
  ``Perm = (-1)^N sum_x (-1)^{|x|} prod_j C(c_j, x_j) prod_i (sum_j x_j M_ij)^{r_i}``
  over ``0 <= x_j <= c_j``, costing ``prod_j (c_j + 1)`` terms instead of ``2^N``
  (the cheaper of rows/columns is enumerated; Perm(M) = Perm(M^T)).
* hafnian / loop hafnian: the Glynn-type power-trace formula (Bjorklund, Gupt, Quesada
  2019; finite-difference sieve as in Bulmer et al. 2022) with the loop correction of
  Bjorklund, Cygan, Koivisto, Kaski: ``haf(A) = 2^{1-n} sum_{x in {+-1}^n, x_0 = 1}
  (prod x) [eta^n] exp(sum_k (tr(C^k)/(2k) + v^T X_x C^{k-1} v / 2) eta^k)`` with
  ``C = A X_x`` (zero diagonal), loop weights ``v`` (absent for the plain hafnian) and
  ``X_x`` swapping the fixed index pairs ``(2p, 2p+1)`` with weight ``x_p``. Power traces
  are taken from explicit matrix powers, not eigenvalues (measured ~1e-15 relative error
  at ``2n = 24`` vs ~1e-11 for the subset/eigenvalue variant). Cost ``O(n^4 2^n)``. An
  odd-dimensional loop hafnian is padded with an isolated vertex of loop weight 1.
* torontonian: ``Tor(O) = sum_{Z subset [n]} (-1)^{n-|Z|} / sqrt(det(I - O_Z))`` and
  ``ltor(O, g) = sum_Z (-1)^{n-|Z|} exp(g_Z^T (I - O_Z)^{-1} g_Z^* / 2) / sqrt(det(I-O_Z))``
  for a ``2n x 2n`` matrix in the *block* ordering ``(a_1..a_n, a_1^dag..a_n^dag)``, so
  ``O_Z`` keeps rows/columns ``{i, i+n : i in Z}`` (Quesada, Arrazola, Killoran 2018;
  Bulmer et al. 2022). The principal square root is used; in Gaussian boson sampling
  ``I - O_Z`` is Hermitian positive definite, so the determinant is real and positive.
  Terms are summed with compensated (Neumaier / ``math.fsum``) summation; the alternating
  sum still loses about ``log10(max term / |result|)`` digits by cancellation.

The ``*_by_definition`` functions evaluate the defining combinatorial sums (permutations,
perfect matchings, matchings with loops) and serve as independent oracles for small sizes.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy.special import comb

from .. import native

Backend = Literal["auto", "native", "python"]

_CHUNK = 1 << 15


def _kernels(backend: Backend):
    """Return the native sampling submodule, ``None`` for NumPy, or raise for bad input."""
    if backend == "python":
        return None
    if backend not in ("auto", "native"):
        raise ValueError(f"Unknown backend {backend!r}; use 'auto', 'native' or 'python'")
    try:
        return native.require("sampling")
    except ImportError:
        if backend == "native":
            raise
        return None


def _square(matrix) -> np.ndarray:
    array = np.asarray(matrix, dtype=complex)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("Expected a square matrix")
    if not np.isfinite(array).all():
        raise ValueError("Matrix entries must be finite")
    return array


def _multiplicities(values, size: int, name: str) -> np.ndarray:
    if values is None:
        return np.ones(size, dtype=np.int64)
    result = np.asarray(values)
    if result.shape != (size,) or not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"{name} must be {size} nonnegative integers")
    if (result < 0).any():
        raise ValueError(f"{name} must be nonnegative")
    return result.astype(np.int64)


def repeat_indices(matrix, repetitions: Sequence[int]) -> np.ndarray:
    """Return the matrix with row and column ``i`` repeated ``repetitions[i]`` times."""
    array = np.asarray(matrix)
    reps = _multiplicities(repetitions, array.shape[0], "repetitions")
    index = np.repeat(np.arange(array.shape[0]), reps)
    return array[np.ix_(index, index)]


# --------------------------------------------------------------------------- permanent


def permanent(matrix, rows=None, columns=None, *, backend: Backend = "auto") -> complex:
    """Permanent of ``matrix`` with row ``i`` repeated ``rows[i]`` and column ``j`` ``columns[j]``.

    Without multiplicities this is the permanent of a square matrix. With them, ``matrix``
    may be rectangular (e.g. a full interferometer unitary): the permanent of the
    ``N x N`` matrix with repeated rows/columns is evaluated by the multiplicity-aware
    Ryser formula without building the repeated matrix. ``N = sum(rows) = sum(columns)``.

    Raises:
        ValueError: Non-square matrix without multiplicities, or ``sum(rows) != sum(columns)``.
    """
    array = np.asarray(matrix, dtype=complex)
    if array.ndim != 2:
        raise ValueError("Expected a matrix")
    if rows is None and columns is None and array.shape[0] != array.shape[1]:
        raise ValueError("Expected a square matrix")
    if not np.isfinite(array).all():
        raise ValueError("Matrix entries must be finite")
    r = _multiplicities(rows, array.shape[0], "rows")
    c = _multiplicities(columns, array.shape[1], "columns")
    if r.sum() != c.sum():
        raise ValueError("Row and column multiplicities must have equal sums")
    kernels = _kernels(backend)
    if kernels is not None:
        return complex(kernels.permanent_repeated(array, r.tolist(), c.tolist()))
    return _ryser_repeated(array, r, c)


def _ryser_repeated(matrix: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> complex:
    keep_r, keep_c = rows > 0, columns > 0
    matrix, rows, columns = matrix[np.ix_(keep_r, keep_c)], rows[keep_r], columns[keep_c]
    total_photons = int(rows.sum())
    if total_photons == 0:
        return 1.0 + 0.0j
    if np.prod(rows + 1.0) < np.prod(columns + 1.0):
        matrix, rows, columns = matrix.T, columns, rows
    radices = tuple(int(x) + 1 for x in columns)
    binomials = [comb(c, np.arange(c + 1)) for c in columns]
    count = math.prod(radices)
    real: list[float] = []
    imag: list[float] = []
    for start in range(0, count, _CHUNK):
        digits = np.stack(
            np.unravel_index(np.arange(start, min(count, start + _CHUNK)), radices), 1
        )
        sums = digits @ matrix.T
        terms = np.prod(sums**rows, axis=1)
        weight = np.prod([binomials[j][digits[:, j]] for j in range(len(radices))], axis=0)
        sign = np.where((total_photons - digits.sum(1)) % 2, -1.0, 1.0)
        chunk = terms * weight * sign
        real.extend(chunk.real)
        imag.extend(chunk.imag)
    return complex(math.fsum(real), math.fsum(imag))


def permanent_by_definition(matrix) -> complex:
    """Permanent as the sum over all permutations (``O(n! n)``; oracle for n <= 8)."""
    array = _square(matrix)
    n = array.shape[0]
    return complex(
        sum(np.prod(array[np.arange(n), list(p)]) for p in itertools.permutations(range(n)))
        if n
        else 1.0
    )


# --------------------------------------------------------------------- hafnian family


def hafnian(matrix, *, backend: Backend = "auto") -> complex:
    """Hafnian of a symmetric matrix (sum over perfect matchings; diagonal ignored).

    Raises:
        ValueError: Non-square, non-finite or non-symmetric input.
    """
    array = _symmetric(matrix)
    if array.shape[0] % 2:
        return 0.0 + 0.0j
    kernels = _kernels(backend)
    if kernels is not None:
        return complex(kernels.hafnian(array))
    return _power_trace(array, None)


def loop_hafnian(matrix, diagonal=None, *, backend: Backend = "auto") -> complex:
    """Loop hafnian of a symmetric matrix; loop weights are ``diagonal`` or the matrix diagonal.

    Raises:
        ValueError: Non-square, non-finite or non-symmetric input, or wrong diagonal size.
    """
    array = _symmetric(matrix).copy()
    if diagonal is not None:
        loops = np.asarray(diagonal, dtype=complex)
        if loops.shape != (array.shape[0],) or not np.isfinite(loops).all():
            raise ValueError("diagonal must be a finite vector matching the matrix size")
        np.fill_diagonal(array, loops)
    kernels = _kernels(backend)
    if kernels is not None:
        return complex(kernels.loop_hafnian(array))
    loops = np.diag(array).copy()
    if array.shape[0] % 2:
        array = np.pad(array, ((0, 1), (0, 1)))
        loops = np.append(loops, 1.0)
    return _power_trace(array, loops)


def _symmetric(matrix) -> np.ndarray:
    array = _square(matrix)
    scale = max(1.0, float(np.abs(array).max(initial=0.0)))
    if not np.allclose(array, array.T, rtol=0.0, atol=1e-12 * scale):
        raise ValueError("Hafnians are defined for symmetric matrices")
    return array


def _exp_series_coefficient(g: np.ndarray, n: int) -> complex:
    """Return ``[eta^n] exp(sum_{k>=1} g[k-1] eta^k)`` by the recursion e_j = sum k g_k e_{j-k}/j."""
    e = np.zeros(n + 1, dtype=complex)
    e[0] = 1.0
    for j in range(1, n + 1):
        e[j] = np.dot(np.arange(1, j + 1) * g[:j], e[j - 1 :: -1]) / j
    return e[n]


def _power_trace(array: np.ndarray, loops: np.ndarray | None) -> complex:
    """Glynn-type power-trace sum (see module doc); ``2^(n-1)`` terms of size ``2n``."""
    n = array.shape[0] // 2
    if n == 0:
        return 1.0 + 0.0j
    swap = np.arange(2 * n) ^ 1
    off = array.copy()
    np.fill_diagonal(off, 0.0)
    ax = off[:, swap]  # A X with the diagonal removed
    h = (n + 1) // 2
    real, imag = [], []
    for code in range(1 << (n - 1)):
        x = np.ones(n)
        x[1:] = np.where((code >> np.arange(n - 1)) & 1, -1.0, 1.0)
        xs = np.repeat(x, 2)
        c = ax * xs
        powers = [c]
        for _ in range(h - 1):
            powers.append(powers[-1] @ c)
        traces = [np.trace(p) for p in powers] + [
            np.sum(powers[h - 1] * powers[k - h - 1].T) for k in range(h + 1, n + 1)
        ]
        g = np.array(traces) / (2 * np.arange(1, n + 1))
        if loops is not None:
            xv, w = xs * loops[swap], loops.copy()
            for k in range(n):
                g[k] += 0.5 * (xv @ w)
                w = c @ w
        term = _exp_series_coefficient(g, n) * np.prod(x)
        real.append(term.real)
        imag.append(term.imag)
    return complex(math.fsum(real), math.fsum(imag)) / 2 ** (n - 1)


def _matchings(vertices: tuple[int, ...], loops: bool):
    """Yield perfect matchings (as edge lists) of ``vertices``; loops are edges (i, i)."""
    if not vertices:
        yield []
        return
    first, rest = vertices[0], vertices[1:]
    if loops:
        for tail in _matchings(rest, loops):
            yield [(first, first)] + tail
    for k, partner in enumerate(rest):
        for tail in _matchings(rest[:k] + rest[k + 1 :], loops):
            yield [(first, partner)] + tail


def hafnian_by_definition(matrix) -> complex:
    """Hafnian as the sum over perfect matchings (``(2n-1)!!`` terms; oracle for 2n <= 12)."""
    array = _square(matrix)
    return complex(
        sum(
            math.prod(array[i, j] for i, j in m)
            for m in _matchings(tuple(range(len(array))), False)
        )
    )


def loop_hafnian_by_definition(matrix) -> complex:
    """Loop hafnian as the sum over matchings with loops (diagonal entries are loop weights)."""
    array = _square(matrix)
    return complex(
        sum(
            math.prod(array[i, j] for i, j in m) for m in _matchings(tuple(range(len(array))), True)
        )
    )


# ------------------------------------------------------------------- torontonian family


def torontonian(matrix, *, backend: Backend = "auto") -> complex:
    """Torontonian of a ``2n x 2n`` matrix in block ``(a, a^dag)`` ordering (see module doc).

    Raises:
        ValueError: Non-square, odd-dimensional or non-finite input.
    """
    array = _even(matrix)
    kernels = _kernels(backend)
    if kernels is not None:
        return complex(kernels.torontonian(array))
    return _torontonian(array, None)


def loop_torontonian(matrix, gamma, *, backend: Backend = "auto") -> complex:
    """Loop torontonian ``ltor(O, gamma)`` in block ``(a, a^dag)`` ordering (see module doc).

    Raises:
        ValueError: Invalid matrix, or ``gamma`` not a finite vector of matching size.
    """
    array = _even(matrix)
    g = np.asarray(gamma, dtype=complex)
    if g.shape != (array.shape[0],) or not np.isfinite(g).all():
        raise ValueError("gamma must be a finite vector matching the matrix size")
    kernels = _kernels(backend)
    if kernels is not None:
        return complex(kernels.loop_torontonian(array, g))
    return _torontonian(array, g)


def _even(matrix) -> np.ndarray:
    array = _square(matrix)
    if array.shape[0] % 2:
        raise ValueError("Torontonians take a 2n x 2n matrix")
    return array


def _torontonian(array: np.ndarray, gamma: np.ndarray | None) -> complex:
    n = array.shape[0] // 2
    real, imag = [], []
    for mask in range(1 << n):
        modes = [i for i in range(n) if mask >> i & 1]
        index = modes + [i + n for i in modes]
        m = np.eye(len(index)) - array[np.ix_(index, index)]
        term = 1.0 / np.sqrt(complex(np.linalg.det(m))) if index else 1.0 + 0.0j
        if gamma is not None and index:
            g = gamma[index]
            term *= np.exp(0.5 * (g @ np.linalg.solve(m, g.conj())))
        term *= (-1.0) ** (n - len(modes))
        real.append(term.real)
        imag.append(term.imag)
    return complex(math.fsum(real), math.fsum(imag))
