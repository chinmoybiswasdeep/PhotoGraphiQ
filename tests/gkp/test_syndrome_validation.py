import numpy as np
import pytest
from scipy.integrate import simpson

import photographiq as pg
from tests.v03_reference import hermites


@pytest.mark.parametrize("quadrature", ["q", "p"])
@pytest.mark.parametrize(
    "shift,outcome",
    [
        (0.15, 0.1),
        (0.9, pg.gkp.SPACING / 2 - 0.03),
        (0.9, pg.gkp.SPACING / 2 + 0.03),
        (pg.gkp.SPACING + 0.15, pg.gkp.SPACING + 0.1),
    ],
)
def test_syndrome_filter_probability_and_corrected_mean(quadrature, shift, outcome):
    errors = []
    for cutoff in (32, 48, 72):
        ancilla = pg.gkp.superposition(
            1, 1, cutoff=cutoff, peak_width=0.7, envelope=0.7, peaks=4, grid_points=2049
        )
        pattern = pg.gkp.correction_pattern(quadrature=quadrature, resource=ancilla)
        source = pg.GaussianInput.coherent((1 if quadrature == "q" else 1j) * shift / 2)
        result = pg.simulate(
            pattern,
            inputs={"in": source},
            backend="piquasso-fock",
            cutoff=cutoff,
            measurement_outcomes={"syndrome": outcome},
        )
        # The independent SUM kernel is psi(x)*ancilla(m-x). Fourier
        # conjugation makes x=p for the p branch, with the same positive sign.
        x = np.linspace(-14, 14, 6001)
        anc = hermites(outcome - x, cutoff) @ np.asarray(ancilla.amplitudes)
        density = np.exp(-((x - shift) ** 2) / 2) / np.sqrt(2 * np.pi) * abs(anc) ** 2
        probability = simpson(density, x=x)
        residual = outcome - np.floor(outcome / pg.gkp.SPACING + 0.5) * pg.gkp.SPACING
        mean = simpson((x - residual) * density, x=x) / probability
        actual = result.state.quadrature("in", 0 if quadrature == "q" else np.pi / 2)[0]
        errors.append(max(abs(actual - mean), abs(np.exp(result.log_likelihood) - probability)))
    assert errors[-1] < 2e-6, errors
    assert errors[-1] <= errors[0] + 1e-10, errors
