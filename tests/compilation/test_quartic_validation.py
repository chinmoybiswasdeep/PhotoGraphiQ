import numpy as np
import pytest

from tests.v03_reference import synthesis_observation


@pytest.mark.parametrize("strength,angle", [(0.0005, 0.0), (-0.002, 0.3), (0.004, 0.0)])
@pytest.mark.parametrize("source", ["vacuum", "superposition", "coherent"])
def test_quartic_refinement_and_independent_target(strength, angle, source):
    rows = [synthesis_observation("quartic", strength, n, 48, source, angle) for n in (4, 16, 64)]
    assert rows[-1]["amplitude_error"] < rows[0]["amplitude_error"]
    assert rows[-1]["infidelity"] < 2e-4
    fine = synthesis_observation("quartic", strength, 64, 72, source, angle)
    assert abs(fine["amplitude_error"] - rows[-1]["amplitude_error"]) < 2e-5


def test_quartic_commutator_amplitude_order():
    counts = np.array([16, 64, 256, 1024])
    errors = [
        synthesis_observation("quartic", 0.001, int(n), 64, "superposition")["amplitude_error"]
        for n in counts
    ]
    order = -np.polyfit(np.log(counts), np.log(errors), 1)[0]
    assert 0.4 < order < 0.6, (errors, order)
