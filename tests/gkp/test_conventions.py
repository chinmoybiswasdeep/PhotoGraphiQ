import numpy as np
import pytest

import photographiq as pg


def test_weyl_logical_anticommutation_and_stabilizer_commutation():
    length = pg.gkp.SPACING
    assert np.exp(-0.5j * length**2) == pytest.approx(-1, abs=1e-14)
    assert np.exp(-2j * length**2) == pytest.approx(1, abs=1e-14)
    assert pg.gkp.logical_displacement(0, "X").q == length
    assert pg.gkp.logical_displacement(0, "Z").p == length


@pytest.mark.parametrize("cell", [-10000, -3, -1, 0, 1, 4, 10000])
@pytest.mark.parametrize("fraction", [-0.5, -0.5 + 1e-8, 0.0, 0.5 - 1e-8, 0.5])
def test_decoder_cells_and_ties(cell, fraction):
    length = pg.gkp.SPACING
    residual, parity = pg.gkp.decode_shift((cell + fraction) * length)
    expected_cell = cell + int(fraction == 0.5)
    assert residual == pytest.approx((cell + fraction - expected_cell) * length, abs=5e-12)
    assert parity == expected_cell % 2


@pytest.mark.parametrize("cell", [-4, -1, 0, 3, 10])
def test_decoder_adjacent_floats_at_boundary(cell):
    boundary = (cell + 0.5) * pg.gkp.SPACING
    assert pg.gkp.decode_shift(np.nextafter(boundary, -np.inf))[1] == cell % 2
    assert pg.gkp.decode_shift(boundary)[1] == (cell + 1) % 2
    assert pg.gkp.decode_shift(np.nextafter(boundary, np.inf))[1] == (cell + 1) % 2
