"""Regression: deep, strongly anti-squeezed Gaussian circuits must not trip the symmetry check.

Before re-symmetrising after every gate, both cases below raised
``ValueError: Covariance must be symmetric`` (relative asymmetry ~1e-16, absolute ~1e-10).
"""

import numpy as np
import pytest

import photographiq as pg
from photographiq import gaussian as gm
from photographiq.backends import GaussianBackend, PiquassoBackend


def _close(left, right, rtol):
    scale = max(np.abs(left).max(), 1.0)
    return np.abs(left - right).max() <= rtol * scale


def test_large_squeezing_cluster_with_tap_agrees_across_backends():
    # r = 8 gives anti-squeezed variances e^16 ~ 9e6; failed on both backends before the fix.
    pattern = pg.Pattern(pg.CVGraph.line(3, squeezing=8.0))
    pattern.extend([pg.Prepare("v", 0.0), pg.BeamSplitter(1, "v", 0.7), pg.Output((0, 1, 2, "v"))])
    gaussian = pg.simulate(pattern, backend="gaussian").state
    native = pg.simulate(pattern, backend="piquasso").state
    for state in (gaussian, native):
        np.testing.assert_array_equal(state.covariance, state.covariance.T)
    assert _close(gaussian.covariance, native.covariance, 1e-12)


@pytest.mark.parametrize("backend", [GaussianBackend, PiquassoBackend])
def test_long_random_gate_sequence(backend):
    # Failed at gate 466 (NumPy) / 496 (Piquasso) before the fix, with max|V| ~ 2e5-9e5.
    # Reference: the same gates multiplied as plain symplectic matrices, outside any backend.
    rng = np.random.default_rng(3)
    engine = backend()
    for node in range(4):
        engine.prepare(node, squeezing=1.0)
    total = np.eye(8)
    for k in range(600):
        u, v = (int(x) for x in rng.choice(4, 2, replace=False))
        if k % 4 == 0:
            w = rng.uniform(-1, 1)
            engine.entangle(u, v, w)
            gate, modes = gm.cz(w), (u, v)
        elif k % 4 == 1:
            theta = rng.uniform(0, np.pi)
            engine.beamsplitter(u, v, theta)
            gate, modes = gm.beamsplitter(theta), (u, v)
        elif k % 4 == 2:
            r = rng.uniform(-0.3, 0.3)
            engine.squeeze(u, r)
            gate, modes = gm.squeezing(r), (u,)
        else:
            phi = rng.uniform(0, 2 * np.pi)
            engine.rotate(u, phi)
            gate, modes = gm.rotation(phi), (u,)
        total = gm.embed(gate, [j for m in modes for j in (2 * m, 2 * m + 1)], 8) @ total
    initial = np.diag(np.tile([np.e**2, np.e**-2], 4))  # Prepare(r=1): p-squeezed nodes
    reference = total @ initial @ total.T
    state = engine.state
    assert np.abs(reference).max() > 1e5
    np.testing.assert_array_equal(state.covariance, state.covariance.T)
    assert _close(state.covariance, reference, 1e-9)
