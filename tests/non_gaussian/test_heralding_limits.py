"""Physical tap Kraus maps versus mathematical ladders, without conflating success."""

import math

import numpy as np
import pytest

import photographiq as pg
from tests.fock.reference import phase_aligned_distance
from tests.non_gaussian.test_sensitivity import source_vector


@pytest.mark.parametrize("kind", ["coherent", "cat", "superposition"])
@pytest.mark.parametrize("count", [0, 1, 2])
def test_all_count_kraus_amplitudes_and_probabilities(kind, count):
    c, theta = 24, 0.27
    vector = source_vector(kind, c)
    if np.linalg.norm(vector[count:]) == 0:
        with pytest.raises(ValueError, match="zero probability"):
            pg.simulate(
                pg.non_gaussian.photon_subtraction(theta),
                backend="piquasso-fock",
                cutoff=c,
                inputs={"in": pg.FockInput(tuple(vector))},
                measurement_outcomes={"count": count},
            )
        return
    expected = np.zeros(c, complex)
    for n in range(count, c):
        expected[n - count] = (
            vector[n]
            * np.sqrt(math.comb(n, count))
            * np.sin(theta) ** count
            * np.cos(theta) ** (n - count)
        )
    probability = np.vdot(expected, expected).real
    expected /= np.sqrt(probability)
    result = pg.simulate(
        pg.non_gaussian.photon_subtraction(theta),
        backend="piquasso-fock",
        cutoff=c,
        inputs={"in": pg.FockInput(tuple(vector))},
        measurement_outcomes={"count": count},
    )
    # Number-conserving BS and exact finite superposition have no support loss;
    # only floating-point binomial/interferometer arithmetic contributes here.
    assert phase_aligned_distance(result.state.state_vector, expected) < 1e-11
    assert np.isclose(np.exp(result.log_likelihood), probability, atol=1e-13, rtol=1e-11)


@pytest.mark.parametrize("kind", ["coherent", "cat", "superposition"])
def test_weak_tap_approaches_ideal_subtraction_quantitatively(kind):
    vector = source_vector(kind, 32)
    if kind == "superposition":
        vector[:4] = np.array([1, 1j, 0.7, -0.5j])
        vector /= np.linalg.norm(vector)
    ideal = np.r_[np.sqrt(np.arange(1, 32)) * vector[1:], 0j]
    ideal /= np.linalg.norm(ideal)
    distances = []
    for theta in (0.3, 0.15, 0.075):
        output = pg.simulate(
            pg.non_gaussian.photon_subtraction(theta),
            backend="piquasso-fock",
            cutoff=32,
            inputs={"in": pg.FockInput(tuple(vector))},
            measurement_outcomes={"count": 1},
        )
        distances.append(phase_aligned_distance(ideal, output.state.state_vector))
    # cos(theta)^n = 1 - n theta^2/2 + O(theta^4): halving theta
    # quarters the leading amplitude error. Allow 20% asymptotic remainder.
    assert distances[-1] < distances[0]
    assert 0.2 < distances[-1] / distances[-2] < 0.3, distances


def test_ideal_coherent_and_cat_subtraction_identities():
    alpha = 0.8 + 0.3j
    for parity in (-1, 1):
        actual = pg.FockInput.cat(alpha, 40, parity).photon_subtracted()
        expected = pg.FockInput.cat(alpha, 39, -parity)
        assert phase_aligned_distance(actual.amplitudes, expected.amplitudes) < 1e-12
    vector = source_vector("coherent", 40)
    actual = pg.FockInput(tuple(vector)).photon_subtracted()
    # a|alpha> = alpha|alpha>; omitted c=40 tail is below numerical precision.
    assert phase_aligned_distance(actual.amplitudes, vector[:-1]) < 1e-12
