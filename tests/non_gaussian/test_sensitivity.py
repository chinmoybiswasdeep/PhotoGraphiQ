"""Algebraic finite-model tests, distinct from infinite-space convergence claims."""

import numpy as np
import piquasso as pq
import pytest
from scipy.special import gammaln

import photographiq as pg
from tests.fock.reference import cubic, kerr, phase_aligned_distance, quadratic, rotation


def source_vector(kind, cutoff):
    vector = np.zeros(cutoff, complex)
    if kind == "vacuum":
        vector[0] = 1
    elif kind == "superposition":
        vector[:2] = [1 / np.sqrt(2), 1j / np.sqrt(2)]
    else:
        alpha = 0.4 + 0.2j if kind == "coherent" else 0.8 + 0.2j
        n = np.arange(cutoff)
        vector = np.exp(-(abs(alpha) ** 2) / 2 + n * np.log(alpha) - gammaln(n + 1) / 2)
        if kind == "cat":
            vector *= 1 + (-1) ** n
        vector /= np.linalg.norm(vector)
    return vector


def native_state(vector, instruction):
    with pq.Program() as program:
        for n, a in enumerate(vector):
            if abs(a):
                pq.Q() | pq.NumberState((n,)) * a
        pq.Q(0) | instruction
    state = (
        pq.PureFockSimulator(d=1, config=pq.Config(cutoff=len(vector), hbar=2))
        .execute(program)
        .state
    )
    state.normalize()
    return state.state_vector


@pytest.mark.parametrize("gamma", [0.05, 0.2, 0.5, 0.8])
@pytest.mark.parametrize("kind", ["vacuum", "coherent", "superposition", "cat"])
def test_cubic_sensitivity_separate_native_and_adapter_oracles(gamma, kind):
    c = 48
    vector = source_vector(kind, c)
    expected = cubic(c, gamma) @ vector
    pattern = pg.Pattern().extend(
        [pg.Prepare(0, state=pg.FockInput(tuple(vector))), pg.CubicPhase(0, gamma)]
    )
    actual = pg.simulate(pattern, backend="piquasso-fock", cutoff=c).state.state_vector
    raw = native_state(vector, pq.CubicPhase(gamma))
    # Same finite matrix definition; tolerance budgets float64 exponentiation,
    # not Hilbert truncation. Global phase is irrelevant to this physical check.
    assert phase_aligned_distance(actual, expected) < 2e-11
    assert phase_aligned_distance(raw, expected) < 2e-11
    assert phase_aligned_distance(vector, expected) > 1e-3


@pytest.mark.parametrize("kappa", [0.1, 0.3, 0.7, 1.1])
@pytest.mark.parametrize("kind", ["vacuum", "coherent", "superposition", "cat"])
def test_kerr_sensitivity_and_cutoff_invariant_support(kappa, kind):
    vector = source_vector(kind, 16)
    expected = kerr(16, kappa) @ vector
    p = pg.Pattern().extend([pg.Prepare(0, state=pg.FockInput(tuple(vector))), pg.Kerr(0, kappa)])
    for c in (16, 24):
        actual = pg.simulate(p, backend="piquasso-fock", cutoff=c).state.state_vector
        # Kerr is exactly diagonal and does not enlarge occupation support.
        np.testing.assert_allclose(actual[:16], expected, atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(actual[16:], 0, atol=2e-13)
    np.testing.assert_allclose(
        native_state(vector, pq.Kerr(kappa)), expected, atol=2e-13, rtol=2e-13
    )


@pytest.mark.parametrize("angle,s", [(-0.7, 0.3), (0.4, -0.4)])
def test_rotation_quadratic_conventions_against_independent_exponential(angle, s):
    vector = source_vector("superposition", 64)
    expected = quadratic(64, s) @ rotation(64, angle) @ vector
    pattern = pg.Pattern().extend(
        [
            pg.Prepare(0, state=pg.FockInput(tuple(vector))),
            pg.Rotate(0, angle),
            pg.QuadraticPhase(0, s),
        ]
    )
    actual = pg.simulate(pattern, backend="piquasso-fock", cutoff=64).state.state_vector
    raw = native_state(rotation(64, angle) @ vector, pq.QuadraticPhase(s))
    # Infinite Gaussian matrix elements and exp(i s (PqP)^2/4) differ near the
    # boundary. c=64 and a low-energy source resolve that tail below this target.
    assert phase_aligned_distance(actual, expected) < 1e-9
    assert phase_aligned_distance(raw, expected) < 1e-9
