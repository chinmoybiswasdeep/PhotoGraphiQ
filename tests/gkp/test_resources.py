import numpy as np
import pytest

import photographiq as pg


def test_comb_projection_and_decoder():
    resource = pg.GKPResource(0, peak_width=0.5, envelope=0.5, peaks=4, grid_points=2049)
    low, m1 = resource.project(24)
    high, m2 = resource.project(48)
    assert 0.99 < m1 <= m2 <= 1 + 1e-9
    assert np.linalg.norm(high.amplitudes) == pytest.approx(1.0)
    assert np.linalg.norm(np.array(high.amplitudes)[1::2]) < 1e-10
    refined = pg.GKPResource(0, 0.5, 0.5, 5, 4097).fock(48)
    assert abs(np.vdot(refined.amplitudes, high.amplitudes)) ** 2 > 1 - 1e-9
    residual, bit = pg.gkp.decode_shift(pg.gkp.SPACING + 0.1)
    assert residual == pytest.approx(0.1)
    assert bit == 1
    assert pg.gkp.logical_displacement("a", "X").q == pg.gkp.SPACING
    assert pg.gkp.logical_displacement("a", "Z").p == pg.gkp.SPACING
    p = pg.Pattern().append(pg.Prepare("g", state=resource))
    result = pg.simulate(p, backend="piquasso-fock", cutoff=48)
    stabilizers = pg.gkp.stabilizers(result.state)
    assert all(0 < v.real < 1 for v in stabilizers.values())
    assert p.to_json() == pg.Pattern.from_json(p.to_json()).to_json()


def test_gkp_syndrome_pattern():
    ancilla = pg.gkp.superposition(1, 1, cutoff=32, peak_width=0.6, envelope=0.6)
    pattern = pg.gkp.correction_pattern(resource=ancilla)
    assert pattern.outputs == ("in",)
    assert pattern.validate() is pattern
