import numpy as np
import pytest

from tests.v03_reference import synthesis_observation


@pytest.mark.parametrize("source", ["vacuum", "one", "superposition", "coherent", "cat"])
@pytest.mark.parametrize("strength", [-0.002, 0.005])
def test_kerr_multiple_inputs_strengths_and_cutoffs(source, strength):
    rows = [synthesis_observation("kerr", strength, n, 48, source) for n in (4, 16, 64)]
    assert rows[-1]["amplitude_error"] < rows[0]["amplitude_error"]
    assert rows[-1]["infidelity"] < 1e-5
    fine = synthesis_observation("kerr", strength, 64, 72, source)
    assert abs(fine["amplitude_error"] - rows[-1]["amplitude_error"]) < 2e-6


def test_kerr_observed_amplitude_order():
    counts = np.array([16, 64, 256, 1024])
    errors = [
        synthesis_observation("kerr", 0.002, int(n), 64, "superposition")["amplitude_error"]
        for n in counts
    ]
    order = -np.polyfit(np.log(counts), np.log(errors), 1)[0]
    assert 0.4 < order < 0.6, (errors, order)
