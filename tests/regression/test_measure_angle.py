"""Direct backend ``measure`` calls honour ``Homodyne.angle`` (previously measured q)."""

import numpy as np
import pytest

import photographiq as pg
from photographiq.backends import GaussianBackend
from photographiq.backends.native_gaussian import NativeGaussianBackend


def backends():
    yield GaussianBackend()
    yield NativeGaussianBackend()
    try:
        from photographiq.backends.native_fock import NativeFockBackend, NativeMixedFockBackend

        yield NativeFockBackend(30)
        yield NativeMixedFockBackend(20)
    except ImportError:  # native fock component not built
        pass


@pytest.mark.parametrize("backend", list(backends()), ids=lambda b: type(b).__name__)
def test_homodyne_angle_used_without_explicit_angle(backend):
    backend.prepare(0, state=pg.GaussianInput.coherent(1j))  # <q> = 0, <p> = 2, Var = 1
    backend.measure(0, pg.Homodyne(np.pi / 2), outcome=2.0)
    # p was measured: N(2; 2, 1); measuring q instead would give N(2; 0, 1) = 0.054
    assert backend.last_measurement["value"] == pytest.approx(1 / np.sqrt(2 * np.pi), abs=1e-6)


def test_explicit_angle_contradicting_homodyne_raises():
    backend = GaussianBackend()
    backend.prepare(0, state=pg.GaussianInput.coherent(1j))
    with pytest.raises(ValueError, match="contradicts"):
        backend.measure(0, pg.Homodyne(np.pi / 2), 0.0, outcome=2.0)


def test_explicit_angle_with_default_homodyne_still_wins():
    backend = GaussianBackend()
    backend.prepare(0, state=pg.GaussianInput.coherent(1j))
    backend.measure(0, pg.Homodyne(), np.pi / 2, outcome=2.0)
    assert backend.last_measurement["value"] == pytest.approx(1 / np.sqrt(2 * np.pi))
