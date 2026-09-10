import numpy as np
import pytest

from tests.v03_reference import synthesis_observation


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_exact_lower_degree_ideal_primitives(degree):
    from scipy.linalg import expm

    import photographiq as pg
    from tests.v03_reference import circuit_unitary, vector

    cutoff, strength, angle = 32, 0.02, 0.3
    a = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    x = np.exp(-1j * angle) * a + np.exp(1j * angle) * a.T
    source = vector("superposition", cutoff)
    circuit, report = pg.synthesis.quadrature_polynomial([(strength, angle, degree)], steps=4)
    actual = circuit_unitary(circuit, cutoff) @ source
    expected = expm(1j * strength * np.linalg.matrix_power(x, degree)) @ source
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
    assert not report.approximate


def test_lower_degree_lie_sum_order():
    from scipy.linalg import expm

    import photographiq as pg
    from tests.v03_reference import circuit_unitary, vector

    cutoff = 32
    a = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    q, p = a + a.T, -1j * (a - a.T)
    source = vector("superposition", cutoff)
    expected = expm(1j * (0.02 * q + 0.03 * (p @ p))) @ source
    counts = [4, 16, 64]
    errors = []
    for count in counts:
        circuit, report = pg.synthesis.quadrature_polynomial(
            [(0.02, 0.0, 1), (0.03, np.pi / 2, 2)], steps=count
        )
        actual = circuit_unitary(circuit, cutoff) @ source
        actual *= np.exp(-1j * np.angle(np.vdot(expected, actual)))
        errors.append(np.linalg.norm(actual - expected))
        assert report.order == 1
    assert 0.95 < -np.polyfit(np.log(counts), np.log(errors), 1)[0] < 1.05


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
