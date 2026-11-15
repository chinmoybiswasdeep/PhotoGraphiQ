"""Compiled sampling kernels: dispatch, reproducibility across threads, larger sizes."""

from __future__ import annotations

import numpy as np
import pytest

from photographiq import native
from photographiq.sampling import linalg as L


@pytest.fixture(scope="module")
def kernels():
    try:
        return native.require("sampling")
    except ImportError as exc:
        pytest.skip(str(exc))


def _random(rng, *shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def test_native_agrees_with_python_at_larger_sizes(kernels):
    rng = np.random.default_rng(31)
    m = _random(rng, 14, 14) / np.sqrt(14)
    expected = L.permanent(m, backend="python")
    assert abs(L.permanent(m, backend="native") - expected) < 1e-10 * max(1, abs(expected))
    a = _random(rng, 16, 16)
    a = (a + a.T) / np.sqrt(32)
    for f in (L.hafnian, L.loop_hafnian):
        expected = f(a, backend="python")
        assert abs(f(a, backend="native") - expected) < 1e-10 * max(1, abs(expected))
    o = 0.1 * _random(rng, 16, 16)
    expected = L.torontonian(o, backend="python")
    assert abs(L.torontonian(o, backend="native") - expected) < 1e-10 * max(1, abs(expected))


def test_results_do_not_depend_on_thread_count(kernels):
    rng = np.random.default_rng(32)
    m = _random(rng, 13, 13)
    a = _random(rng, 14, 14)
    a = a + a.T
    before = kernels.max_threads()
    try:
        values = []
        for threads in (1, 3, max(1, before)):
            kernels.set_num_threads(threads)
            values.append((kernels.permanent(m), kernels.hafnian(a), kernels.torontonian(0.05 * a)))
    finally:
        kernels.set_num_threads(before)
    assert all(v == values[0] for v in values)  # bitwise: fixed chunking and summation order


def test_native_errors_map_to_value_error(kernels):
    with pytest.raises(ValueError):
        kernels.permanent_repeated(np.eye(2, dtype=complex), [1, 1], [2, 1])
    with pytest.raises(ValueError):
        kernels.hafnian(np.array([[0, 1], [2, 0]], dtype=complex))
    with pytest.raises(ValueError):
        kernels.torontonian(np.zeros((3, 3), dtype=complex))
    with pytest.raises(ImportError):
        native.require("not-a-component")
