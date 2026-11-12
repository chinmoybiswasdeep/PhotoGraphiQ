"""Haar unitaries, Clements decomposition and the passive-symplectic convention."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from photographiq import gaussian as gm
from photographiq.sampling import (
    clements_decomposition,
    haar_unitary,
    mzi,
    passive_symplectic,
)


def test_haar_unitary_is_unitary_and_reproducible():
    u = haar_unitary(6, rng=1)
    assert np.allclose(u.conj().T @ u, np.eye(6), atol=1e-13)
    assert np.array_equal(u, haar_unitary(6, rng=1))


def test_haar_statistics():
    """|U_00|^2 ~ Beta(1, m-1) and E[U_00] = 0 (fails for QR without the phase fix)."""
    m, samples = 4, 4000
    rng = np.random.default_rng(2026)
    entries = np.array([haar_unitary(m, rng)[0, 0] for _ in range(samples)])
    assert stats.kstest(np.abs(entries) ** 2, stats.beta(1, m - 1).cdf).pvalue > 1e-3
    # std(U_00) = 1/sqrt(m); a 5 sigma bound on the sample mean of a centred variable.
    assert abs(entries.mean()) < 5 / np.sqrt(m * samples)
    second = np.mean(np.abs(entries) ** 4)
    assert abs(second - 2 / (m * (m + 1))) < 5 * np.std(np.abs(entries) ** 4) / np.sqrt(samples)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 7, 10])
def test_clements_reconstructs_unitary_and_photographiq_gates(m):
    u = haar_unitary(m, rng=m)
    d = clements_decomposition(u)
    assert len(d.elements) == m * (m - 1) // 2
    assert np.abs(d.unitary() - u).max() < 1e-13
    # Compose the PhotoGraphiQ gate symplectics (gaussian.py) in application order.
    s = np.eye(2 * m)
    for op in d.operations():
        if op[0] == "rotate":
            g = gm.embed(gm.rotation(op[2]), [2 * op[1], 2 * op[1] + 1], 2 * m)
        else:
            _, a, b, theta = op
            g = gm.embed(gm.beamsplitter(theta), [2 * a, 2 * a + 1, 2 * b, 2 * b + 1], 2 * m)
        s = g @ s
    assert np.abs(s - passive_symplectic(u)).max() < 1e-13


def test_passive_symplectic_matches_gate_conventions():
    phi, theta = 0.37, -1.1
    assert np.allclose(passive_symplectic([[np.exp(1j * phi)]]), gm.rotation(phi))
    bs = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    assert np.allclose(passive_symplectic(bs), gm.beamsplitter(theta))
    s = passive_symplectic(haar_unitary(5, rng=3))
    assert np.allclose(s @ gm.omega(5) @ s.T, gm.omega(5))
    assert np.allclose(s @ s.T, np.eye(10))  # passive = orthogonal symplectic


def test_mzi_is_beamsplitter_after_rotation():
    t = mzi(2, 0, 1, 0.4, 1.3)
    rot = np.diag([np.exp(1.3j), 1.0])
    bs = np.array([[np.cos(0.4), -np.sin(0.4)], [np.sin(0.4), np.cos(0.4)]])
    assert np.allclose(t, bs @ rot)


def test_clements_rejects_non_unitary():
    with pytest.raises(ValueError):
        clements_decomposition(np.ones((2, 2)))
