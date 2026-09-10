import numpy as np
import piquasso as pq
import pytest
from scipy.integrate import quad_vec
from scipy.special import eval_hermitenorm, gammaln

import photographiq as pg


def reference_injection(cutoff, m, *, gamma=None, r=0, alpha=None):
    n = np.arange(cutoff)

    def integrand(q):
        vacuum = (2 * np.pi) ** (-0.25) * np.exp(-q * q / 4)
        if gamma is not None:
            resource = (2 * np.pi * np.exp(2 * r)) ** (-0.25) * np.exp(
                -((q + m) ** 2) / (4 * np.exp(2 * r))
            )
            filtered = vacuum * resource * np.exp(1j * gamma * q**3 / 6)
        else:
            # Real even cat, normalized in the infinite Hilbert space.
            resource = (
                (2 * np.pi) ** (-0.25)
                * (
                    np.exp(-((q + m - 2 * alpha) ** 2) / 4)
                    + np.exp(-((q + m + 2 * alpha) ** 2) / 4)
                )
                / np.sqrt(2 * (1 + np.exp(-2 * alpha**2)))
            )
            filtered = vacuum * resource
        return eval_hermitenorm(n, q) * np.exp(-gammaln(n + 1) / 2) * vacuum * filtered

    vector, error = quad_vec(integrand, -15, 15, epsabs=1e-11, epsrel=1e-11)
    assert error < 1e-8
    return vector / np.linalg.norm(vector)


@pytest.mark.parametrize("kind", ["cubic", "cat"])
def test_resource_injection_independent_wavefunction_and_fixed_branch(kind):
    cutoff, m = 36, 0.4
    if kind == "cubic":
        pattern = pg.non_gaussian.cubic_injection(0.3, 0.2)
        expected = reference_injection(cutoff, m, gamma=0.3, r=0.2)
    else:
        pattern = pg.non_gaussian.cat_injection(0.8)
        expected = reference_injection(cutoff, m, alpha=0.8)
    result = pg.simulate(
        pattern, backend="piquasso-fock", cutoff=cutoff, measurement_outcomes={"m": m}
    )
    overlap = abs(np.vdot(expected, result.state.state_vector)) ** 2
    assert overlap > 1 - 2e-5, overlap
    if kind == "cubic":
        density = np.exp(-m * m / (2 * (1 + np.exp(0.4)))) / np.sqrt(2 * np.pi * (1 + np.exp(0.4)))
        assert abs(result.measurement_statistics["m"]["value"] - density) < 2e-5
        assert pattern.dependencies().has_edge(4, 5)
        assert pattern.dependencies().has_edge(4, 6)
    restored = pg.Pattern.from_json(pattern.to_json())
    assert restored.inspect() == pattern.inspect()


@pytest.mark.parametrize("cutoff,tolerance", [(24, 5e-7), (48, 3e-8)])
def test_raw_piquasso_resource_entanglement_before_measurement(cutoff, tolerance):
    gamma, r = 0.25, 0.15
    pattern = pg.non_gaussian.cubic_injection(gamma, r)
    pattern.commands = pattern.commands[:4]
    actual = pg.simulate(pattern, backend="piquasso-fock", cutoff=cutoff).state
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(-r)
        pq.Q(1) | pq.CubicPhase(gamma)
        pq.Q(0, 1) | pq.GaussianTransform(
            passive=np.array([[1, 0.5], [-0.5, 1]]), active=np.array([[0, -0.5], [-0.5, 0]])
        )
    raw = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=cutoff, hbar=2)).execute(program).state
    raw.normalize()
    # Direct inverse-SUM Bogoliubov blocks and Fourier-CZ synthesis differ.
    # Finite-space projections in different Gaussian decompositions need not agree
    # exactly. These tolerances tighten with cutoff; their norms are also monitored.
    assert abs(np.vdot(actual.state_vector, raw.state_vector)) ** 2 > 1 - tolerance
    with pq.Program() as synthesized:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(-r)
        pq.Q(1) | pq.CubicPhase(gamma)
        pq.Q(1) | pq.Phaseshifter(np.pi / 2)
        pq.Q(0, 1) | pq.GaussianTransform(
            passive=np.array([[1, -0.5j], [-0.5j, 1]]), active=np.array([[0, -0.5j], [-0.5j, 0]])
        )
        pq.Q(1) | pq.Phaseshifter(-np.pi / 2)
    raw = (
        pq.PureFockSimulator(d=2, config=pq.Config(cutoff=cutoff, hbar=2))
        .execute(synthesized)
        .state
    )
    raw.normalize()
    np.testing.assert_allclose(actual.state_vector, raw.state_vector, atol=1e-12)
