"""Optional PhotoGraphiQML hybrid forward inference agrees with the native Fock path."""

import numpy as np
import pytest

from photographiq import native
from photographiqml.cv import CVFeatureMap, NonGaussianProgram
from photographiqml.cv.hybrid import hybrid_moments

pytestmark = pytest.mark.skipif(
    not native.AVAILABLE or "fock" not in native.components(), reason="native fock missing"
)


def test_hybrid_forward_matches_native_fock_moments():
    program = NonGaussianProgram(2).kerr(0, 0.15).beamsplitter(0, 1, 0.6).cubic_phase(1, 0.08)
    fmap = CVFeatureMap(2, scale=0.8)
    x = np.array([0.5, -0.3])
    means, variances, _ = program.fock_moments({}, x, fmap, 40, (0, 1))
    h_means, h_vars, weight, truncation = hybrid_moments(program, {}, x, fmap, 24, (0, 1))
    np.testing.assert_allclose(h_means, means, atol=1e-6)
    np.testing.assert_allclose(h_vars, variances, atol=1e-5)
    assert weight > 1 - 1e-9 and truncation["max_boundary_population"] < 1e-6
