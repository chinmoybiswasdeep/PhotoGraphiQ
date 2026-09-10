"""Analytical distributions, finite-support tails, and numerical CDF budgets."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import eval_hermitenorm, gammaln, ndtr

import photographiq as pg
from photographiq.fock_measurements import homodyne_projection


def normal_pdf(x):
    return np.exp(-x * x / 2) / np.sqrt(2 * np.pi)


@pytest.mark.parametrize("n", [0, 1, 2])
@pytest.mark.parametrize("angle", [0, 0.63, np.pi / 2])
def test_number_homodyne_density_cdf_and_normalized_condition(n, angle):
    amplitudes = {(n,): 1.0}
    for x in (-2.1, -0.4, 0.7, 2.8):
        _, vector, density = homodyne_projection(
            amplitudes, 0, 8, angle, np.random.default_rng(0), x
        )
        expected = normal_pdf(x) * eval_hermitenorm(n, x) ** 2 / np.exp(gammaln(n + 1))
        # Algebraic Hermite identity; no sampling or cutoff tail in this comparison.
        assert np.isclose(density, expected, atol=1e-13, rtol=1e-12)
        assert np.isclose(sum(abs(a) ** 2 for a in vector.values()), 1)
    for seed in (3, 7, 19):
        x, _, _ = homodyne_projection(amplitudes, 0, 8, angle, np.random.default_rng(seed))
        cdf = ndtr(x)
        if n == 1:
            cdf -= x * normal_pdf(x)
        if n == 2:
            cdf -= (x**3 + x) * normal_pdf(x) / 2
        # Probability-space comparison budgets quad + root tolerances, avoiding
        # an ill-conditioned x tolerance near a zero of a number-state density.
        assert abs(cdf - np.random.default_rng(seed).random()) < 2e-8


@pytest.mark.parametrize(
    "alpha,angle", [(1.2 + 0.7j, 0), (1.2 + 0.7j, 0.8), (4 + 0j, 0), (2j, np.pi / 2)]
)
def test_displaced_rotated_homodyne_distribution(alpha, angle):
    pattern = pg.Pattern().append(pg.Prepare(0, state=pg.GaussianInput.coherent(alpha)))
    state = pg.simulate(pattern, backend="piquasso-fock", cutoff=96).state
    mean = 2 * np.real(alpha * np.exp(-1j * angle))
    amplitudes = state.native.fock_amplitudes_map
    for x in (mean - 2, mean + 0.2, mean + 1.7):
        _, _, density = homodyne_projection(amplitudes, 0, 96, angle, np.random.default_rng(0), x)
        assert abs(density - normal_pdf(x - mean)) < 1e-9  # resolved coherent tail at c=96
    a = homodyne_projection(amplitudes, 0, 96, angle, np.random.default_rng(21))[0]
    b = homodyne_projection(amplitudes, 0, 96, angle, np.random.default_rng(21))[0]
    assert a == b
    assert abs(ndtr(a - mean) - np.random.default_rng(21).random()) < 2e-8


@pytest.mark.parametrize(
    "resource,gate",
    [
        (pg.GaussianInput.coherent(4), None),
        (pg.GaussianInput.squeezed(-0.7), None),
        (pg.CatResource(2), None),
        (pg.FockInput.number(0), pg.CubicPhase(0, 0.8)),
    ],
)
def test_homodyne_window_normalization_for_challenging_finite_states(resource, gate):
    cutoff = 96
    pattern = pg.Pattern().append(pg.Prepare(0, state=resource))
    if gate is not None:
        pattern.append(gate)
    state = pg.simulate(pattern, backend="piquasso-fock", cutoff=cutoff).state
    n = np.arange(cutoff)
    bound = 2 * np.sqrt(cutoff) + 10
    vector = state.state_vector * np.exp(-0.7j * n)

    def density(x):
        h = eval_hermitenorm(n, x) * np.exp(-gammaln(n + 1) / 2) * np.sqrt(normal_pdf(x))
        return float(abs(h @ vector) ** 2)

    mass, error = quad(density, -bound, bound, epsabs=1e-10, limit=400)
    assert error < 1e-7
    assert abs(mass - 1) < 1e-8

    # Every normalized cutoff-c marginal is bounded by sum |h_n|^2 via
    # Cauchy-Schwarz. Test the beyond-turning-point tail envelope independently.
    def envelope(x):
        h = eval_hermitenorm(n, x) * np.exp(-gammaln(n + 1) / 2) * np.sqrt(normal_pdf(x))
        return float(h @ h)

    tail = 2 * quad(envelope, bound, bound + 20, epsabs=1e-12)[0]
    assert tail < 1e-10
    homodyne_projection(state.native.fock_amplitudes_map, 0, cutoff, 0.7, np.random.default_rng(6))


def test_unrepresented_displacement_fails_instead_of_sampling_wrong_tail():
    with pytest.raises(ValueError, match="truncation"):
        pg.simulate(
            pg.Pattern().append(pg.Prepare(0, state=pg.GaussianInput.coherent(10))).measure(0),
            backend="piquasso-fock",
            cutoff=24,
        )


def test_partial_cdf_error_is_not_silently_ignored(monkeypatch):
    import photographiq.fock_measurements as module

    calls = 0

    def inaccurate_quad(*args, **kwargs):
        nonlocal calls
        calls += 1
        return (1.0, 0.0) if calls == 1 else (0.0, 1e-3)

    monkeypatch.setattr(module, "quad", inaccurate_quad)
    with pytest.raises(ArithmeticError, match="partial-CDF"):
        homodyne_projection({(0,): 1}, 0, 8, 0, np.random.default_rng(1))
