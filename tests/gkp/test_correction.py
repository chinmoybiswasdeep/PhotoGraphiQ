import numpy as np
from scipy.integrate import simpson

import photographiq as pg
from photographiq.fock_measurements import wavefunctions


def test_finite_syndrome_correction_matches_wavefunction_filter():
    errors = []
    for cutoff in (24, 36):
        ancilla = pg.gkp.superposition(
            1, 1, cutoff=cutoff, peak_width=0.7, envelope=0.7, peaks=4, grid_points=2049
        )
        pattern = pg.gkp.correction_pattern(resource=ancilla)
        result = pg.simulate(
            pattern,
            inputs={"in": pg.GaussianInput.coherent(0.1)},
            backend="piquasso-fock",
            cutoff=cutoff,
            measurement_outcomes={"syndrome": 0.1},
        )
        q = np.linspace(-12, 12, 4001)
        phi = np.array([np.dot(wavefunctions(-x, cutoff), ancilla.amplitudes) for x in q])
        # SUM q measurement m=.1 followed by shift -m: psi(q+m)*phi(-q).
        density = np.exp(-((q + 0.1 - 0.2) ** 2) / 2) * abs(phi) ** 2
        density /= simpson(density, x=q)
        reference_mean = simpson(q * density, x=q)
        errors.append(abs(result.state.quadrature("in")[0] - reference_mean))
    assert errors[-1] < 5e-6
    assert errors[-1] < errors[0] / 10
