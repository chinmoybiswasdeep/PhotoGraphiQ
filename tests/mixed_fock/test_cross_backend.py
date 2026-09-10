import numpy as np
import pytest

import photographiq as pg


@pytest.mark.parametrize("noise", [False, True])
def test_overlapping_gaussian_regime_across_backends(noise):
    pattern = pg.Pattern(inputs=("a", "b")).extend(
        [
            pg.Squeeze("a", 0.08),
            pg.Rotate("a", 0.2),
            pg.Displace("b", 0.12, -0.08),
            pg.BeamSplitter("a", "b", 0.2),
            pg.Entangle("a", "b", 0.08),
        ]
    )
    backends = ["gaussian", "piquasso", "piquasso-mixed-fock"]
    if noise:
        pattern.append(pg.Loss("a", 0.7, 0.1))
    else:
        backends.append("piquasso-fock")
    states = [pg.simulate(pattern, backend=b, cutoff=16).state for b in backends]
    for state in states[1:]:
        for node in ("a", "b"):
            for angle in (0.0, 0.3, np.pi / 2):
                np.testing.assert_allclose(
                    state.quadrature(node, angle),
                    states[0].quadrature(node, angle),
                    atol=2e-7,
                    rtol=0,
                )
            assert state.photon_number(node) == pytest.approx(
                states[0].photon_number(node), abs=2e-7
            )
