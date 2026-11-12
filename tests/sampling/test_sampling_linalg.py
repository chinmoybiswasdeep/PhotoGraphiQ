"""Permanent / hafnian / torontonian against definitions, identities and Piquasso.

Every check runs for the NumPy implementation and, when compiled, the native kernels.
Tolerance: relative 1e-12 for sizes up to 10; observed worst relative errors against the
defining sums are 1.3e-14 (permanent, n <= 7), 1.9e-15 (hafnian, 2n <= 10) and 1.1e-14
(loop hafnian), i.e. a ~100x margin.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from photographiq import native
from photographiq.sampling import linalg as L

RTOL = 1e-12


def _native_available() -> bool:
    try:
        native.require("sampling")
    except ImportError:
        return False
    return True


@pytest.fixture(
    params=[
        "python",
        pytest.param(
            "native",
            marks=pytest.mark.skipif(not _native_available(), reason="native sampling not built"),
        ),
    ]
)
def backend(request):
    return request.param


def _random(rng, *shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _symmetric(rng, d):
    a = _random(rng, d, d)
    return a + a.T


def _close(got, expected, rtol=RTOL):
    assert abs(got - expected) <= rtol * max(1.0, abs(expected)), (got, expected)


@pytest.mark.parametrize("n", range(0, 8))
def test_permanent_matches_definition(backend, n):
    m = _random(np.random.default_rng(n), n, n)
    _close(L.permanent(m, backend=backend), L.permanent_by_definition(m))


def test_permanent_with_multiplicities_matches_repeated_matrix(backend):
    rng = np.random.default_rng(7)
    u = _random(rng, 4, 6)
    rows, cols = [2, 0, 3, 1], [1, 3, 0, 1, 1, 0]
    repeated = u[np.ix_(np.repeat(range(4), rows), np.repeat(range(6), cols))]
    _close(L.permanent(u, rows, cols, backend=backend), L.permanent_by_definition(repeated))
    # Both enumeration sides (rows cheaper vs columns cheaper) must agree.
    _close(L.permanent(u.T, cols, rows, backend=backend), L.permanent_by_definition(repeated.T))


def test_permanent_identities(backend):
    for n in range(1, 11):
        _close(L.permanent(np.ones((n, n)), backend=backend), math.factorial(n))
    d = np.diag([1.5, -2.0, 0.5j])
    _close(L.permanent(d, backend=backend), 1.5 * -2.0 * 0.5j)
    # A single photon in column j of a unitary: Perm reduces to the matrix element.
    u = _random(np.random.default_rng(3), 3, 3)
    _close(L.permanent(u, [0, 0, 1], [1, 0, 0], backend=backend), u[2, 0])


def test_permanent_rejects_inconsistent_multiplicities(backend):
    with pytest.raises(ValueError):
        L.permanent(np.eye(2), [1, 1], [2, 1], backend=backend)
    with pytest.raises(ValueError):
        L.permanent(np.ones((2, 3)), backend=backend)


@pytest.mark.parametrize("d", range(0, 11))
def test_hafnians_match_definition(backend, d):
    a = _symmetric(np.random.default_rng(100 + d), d)
    _close(L.hafnian(a, backend=backend), L.hafnian_by_definition(a))
    _close(L.loop_hafnian(a, backend=backend), L.loop_hafnian_by_definition(a))


def test_hafnian_identities(backend):
    rng = np.random.default_rng(11)
    b = _random(rng, 6, 6)
    block = np.block([[np.zeros((6, 6)), b], [b.T, np.zeros((6, 6))]])
    _close(L.hafnian(block, backend=backend), L.permanent_by_definition(b))
    _close(L.loop_hafnian(block, backend=backend), L.permanent_by_definition(b))
    a = _symmetric(rng, 8)
    _close(L.loop_hafnian(a, diagonal=np.zeros(8), backend=backend), L.hafnian(a, backend=backend))
    # Haf(J_2n) = (2n-1)!!; lhaf(J_n) counts involutions (telephone numbers).
    for n in range(1, 7):
        _close(L.hafnian(np.ones((2 * n, 2 * n)), backend=backend), math.prod(range(1, 2 * n, 2)))
    telephone = [1, 1, 2, 4, 10, 26, 76, 232, 764, 2620, 9496]
    for n, t in enumerate(telephone):
        _close(L.loop_hafnian(np.ones((n, n)), backend=backend), t)
    with pytest.raises(ValueError):
        L.hafnian(_random(rng, 4, 4), backend=backend)  # not symmetric


def _single_mode_blocks(n, rng):
    """Block-ordered O that is a direct sum over modes, with Tor/ltor in closed form."""
    o = np.zeros((2 * n, 2 * n), dtype=complex)
    g = np.zeros(2 * n, dtype=complex)
    tor, ltor = 1.0, 1.0
    for i in range(n):
        x, y, z = 0.4 * rng.random(), 0.2 * _random(rng), 0.5 * _random(rng)
        o[i, i] = o[i + n, i + n] = x
        o[i, i + n], o[i + n, i] = y, np.conj(y)
        g[i], g[i + n] = z, np.conj(z)
        m = np.eye(2) - np.array([[x, y], [np.conj(y), x]])
        det = np.linalg.det(m)
        gi = np.array([z, np.conj(z)])
        tor *= 1 / np.sqrt(det) - 1
        ltor *= np.exp(0.5 * gi @ np.linalg.solve(m, gi.conj())) / np.sqrt(det) - 1
    return o, g, tor, ltor


def test_torontonian_product_identities(backend):
    rng = np.random.default_rng(5)
    for n in range(0, 6):
        o, g, tor, ltor = _single_mode_blocks(n, rng)
        _close(L.torontonian(o, backend=backend), tor, 1e-12)
        _close(L.loop_torontonian(o, g, backend=backend), ltor, 1e-12)
        _close(L.loop_torontonian(o, np.zeros(2 * n), backend=backend), tor, 1e-12)
    _close(L.torontonian(np.zeros((8, 8)), backend=backend), 0.0)
    with pytest.raises(ValueError):
        L.torontonian(np.zeros((3, 3)), backend=backend)


# ----------------------------------------------------------------------- Piquasso oracle


def test_permanent_matches_piquasso(backend):
    pq_perm = pytest.importorskip("piquasso._math.permanent")
    rng = np.random.default_rng(21)
    for n in (3, 6, 9, 12):
        m = _random(rng, n, n) / np.sqrt(n)
        ones = np.ones(n, dtype=np.int32)
        _close(L.permanent(m, backend=backend), complex(pq_perm.permanent(m, ones, ones)), 1e-12)
    u = _random(rng, 5, 5)
    rows, cols = np.array([3, 0, 1, 2, 1]), np.array([1, 2, 2, 0, 2])
    expected = complex(pq_perm.permanent(u, rows.astype(np.int32), cols.astype(np.int32)))
    _close(L.permanent(u, rows, cols, backend=backend), expected, 1e-12)


def test_hafnians_match_piquasso(backend):
    pq_haf = pytest.importorskip("piquasso._math.hafnian")
    rng = np.random.default_rng(22)
    for d in (4, 8, 12):
        a = _symmetric(rng, d) / np.sqrt(d)
        ones = np.ones(d, dtype=int)
        _close(
            L.hafnian(a, backend=backend), complex(pq_haf.hafnian_with_reduction(a, ones)), 1e-12
        )
        diag = _random(rng, d)
        expected = complex(pq_haf.loop_hafnian_with_reduction(a, diag, ones))
        _close(L.loop_hafnian(a, diag, backend=backend), expected, 1e-12)


def test_torontonians_match_piquasso(backend):
    """Piquasso takes real matrices with pairs (2i, 2i+1); ours pairs (i, i+n)."""
    pq_tor = pytest.importorskip("piquasso._math.torontonian")
    rng = np.random.default_rng(23)
    for n in (1, 3, 5):
        x = rng.normal(size=(2 * n, 2 * n))
        sigma = np.eye(2 * n) + 0.3 * x @ x.T  # real symmetric, sigma >= I
        o = np.eye(2 * n) - np.linalg.inv(sigma)
        gamma = rng.normal(size=2 * n)
        to_block = np.r_[np.arange(0, 2 * n, 2), np.arange(1, 2 * n, 2)]
        ob, gb = o[np.ix_(to_block, to_block)], gamma[to_block]
        _close(L.torontonian(ob, backend=backend), float(pq_tor.torontonian(o)), 1e-11)
        expected = float(np.real(pq_tor.loop_torontonian(o, gamma)))
        _close(L.loop_torontonian(ob, gb, backend=backend), expected, 1e-11)
