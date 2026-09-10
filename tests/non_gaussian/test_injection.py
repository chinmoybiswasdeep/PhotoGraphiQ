import numpy as np
import piquasso as pq
import pytest
from scipy.integrate import quad_vec
from scipy.special import eval_hermitenorm, gammaln

import photographiq as pg
from photographiq.backends.fock import PiquassoFockBackend


def reference_injection(cutoff, m, *, gamma=None, r=0, alpha=None, eps=1e-11, bound=15):
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

    vector, error = quad_vec(integrand, -bound, bound, epsabs=eps, epsrel=eps)
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
        # Regression: c=36 failed on Linux at 2.0235e-5. No sampling/integration
        # runs at fixed m; compare the same branch at higher Hilbert cutoff.
        fine = pg.simulate(
            pattern, backend="piquasso-fock", cutoff=96, measurement_outcomes={"m": m}
        )
        coarse_error = abs(result.measurement_statistics["m"]["value"] - density)
        fine_error = abs(fine.measurement_statistics["m"]["value"] - density)
        assert fine_error < coarse_error
        # Explicit accuracy target: 1e-6 absolute density, twenty times stricter
        # than the old test, imposed after resolving the Fock tail, not at c=36.
        assert fine_error < 1e-6
        assert pattern.dependencies().has_edge(4, 5)
        assert pattern.dependencies().has_edge(4, 6)
    restored = pg.Pattern.from_json(pattern.to_json())
    assert restored.inspect() == pattern.inspect()


@pytest.mark.parametrize("cutoff", [24, 48])
def test_raw_piquasso_resource_entanglement_before_measurement(cutoff):
    gamma, r = 0.25, 0.15
    pattern = pg.non_gaussian.cubic_injection(gamma, r)
    pattern.commands = pattern.commands[:4]
    actual = pg.simulate(pattern, backend="piquasso-fock", cutoff=cutoff).state
    # Preserve both originally failing cutoffs. Equal physical decompositions
    # are tested separately below; identical instruction sequences should agree
    # up to float64 gate arithmetic and normalization (not truncation error).
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


def decomposition_observation(cutoff):
    """Both realizations and diagnostics, including deliberately underresolved c=12."""
    pattern = pg.non_gaussian.cubic_injection(0.25, 0.15)
    pattern.commands = pattern.commands[:4]
    # Diagnostic scan only: production default rejects the underresolved c=12
    # trajectory. Record loss rather than pretending it satisfies that default.
    actual = pg.simulate(pattern, backend=PiquassoFockBackend(cutoff, norm_tolerance=0.02)).state
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(-0.15)
        pq.Q(1) | pq.CubicPhase(0.25)
        pq.Q(0, 1) | pq.GaussianTransform(
            passive=np.array([[1, 0.5], [-0.5, 1]]), active=np.array([[0, -0.5], [-0.5, 0]])
        )
    raw = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=cutoff, hbar=2)).execute(program).state
    norm = float(raw.norm)
    raw.normalize()
    return {
        "cutoff": cutoff,
        "infidelity": float(max(0, 1 - abs(np.vdot(actual.state_vector, raw.state_vector)) ** 2)),
        "loss_a": abs(1 - min(actual.retained_norms)),
        "loss_b": abs(1 - norm),
        "boundary_a": actual.diagnostics[-1]["boundary_population"],
        "boundary_b": float(
            sum(p for b, p in raw.fock_probabilities_map.items() if sum(b) >= cutoff - 2)
        ),
    }


def test_regression_ci_inverse_sum_decomposition_converges():
    rows = [decomposition_observation(c) for c in (12, 18, 24, 32, 48)]
    # Finite projections through different Bloch-Messiah decompositions do not
    # commute. Test an asymptotic trend, not equality at one platform's c=24/48.
    # Independent analytic injection below anchors the common physical limit.
    for metric in ("infidelity", "loss_a", "loss_b", "boundary_a", "boundary_b"):
        assert rows[-1][metric] < rows[2][metric], (metric, rows)
        assert rows[-1][metric] < rows[0][metric], (metric, rows)
    # Windows and original Linux evidence show >20x improvement from 24 to 48.
    # Require only one order of magnitude, leaving margin for eigenspace choices.
    assert rows[-1]["infidelity"] < 0.1 * rows[2]["infidelity"], rows


@pytest.mark.parametrize(
    "gamma,r,m", [(0.1, 0, -0.6), (0.3, 0.2, 0.4), (-0.25, 0.1, 0.8), (0.2, 0.3, 0)]
)
def test_regression_ci_cubic_injection_error_budget(gamma, r, m):
    pattern = pg.non_gaussian.cubic_injection(gamma, r)
    observations = []
    for cutoff in (36, 64, 96):
        output = pg.simulate(
            pattern, backend="piquasso-fock", cutoff=cutoff, measurement_outcomes={"m": m}
        )
        oracle = reference_injection(cutoff, m, gamma=gamma, r=r)
        density = np.exp(-m * m / (2 * (1 + np.exp(2 * r)))) / np.sqrt(
            2 * np.pi * (1 + np.exp(2 * r))
        )
        observations.append(
            (
                abs(output.measurement_statistics["m"]["value"] - density),
                max(0, 1 - abs(np.vdot(oracle, output.state.state_vector)) ** 2),
            )
        )
    # Independent integration error is resolved separately below. These targets
    # budget finite resource/gate projection, not detector binning or CDF error.
    assert observations[-1][0] < 1e-6, observations
    assert observations[-1][1] < 1e-7, observations
    for i in (0, 1):
        assert observations[-1][i] <= observations[0][i] + 1e-12, observations


def test_independent_injection_quadrature_error_separate_from_cutoff():
    a = reference_injection(96, 0.4, gamma=0.3, r=0.2, eps=1e-9, bound=12)
    b = reference_injection(96, 0.4, gamma=0.3, r=0.2, eps=1e-12, bound=16)
    # Normalized vector change from both quadrature refinement and wider tails
    # must be far below the 1e-7 Hilbert-space infidelity target above.
    assert np.linalg.norm(a - b) < 1e-9
