import numpy as np
import piquasso as pq
import pytest

import photographiq as pg
from photographiq.backends.fock import PiquassoFockBackend
from tests.fock.reference import cubic, displacement, operators, squeeze


def run(commands, cutoff=24, **kwargs):
    return pg.simulate(
        pg.Pattern().extend(commands), backend="piquasso-fock", cutoff=cutoff, **kwargs
    )


@pytest.mark.parametrize("gamma", [0.2, 0.65])
def test_cubic_independent_and_raw_native(gamma):
    cutoff = 36
    source = pg.FockInput((np.sqrt(0.6), 1j * np.sqrt(0.4)))
    result = run([pg.Prepare(0, state=source), pg.CubicPhase(0, gamma)], cutoff)
    initial = np.zeros(cutoff, complex)
    initial[:2] = source.amplitudes
    expected = cubic(cutoff, gamma) @ initial
    np.testing.assert_allclose(result.state.state_vector, expected, atol=2e-13)
    with pq.Program() as program:
        pq.Q() | pq.NumberState((0,)) * source.amplitudes[0]
        pq.Q() | pq.NumberState((1,)) * source.amplitudes[1]
        pq.Q(0) | pq.CubicPhase(gamma)
    raw = pq.PureFockSimulator(d=1, config=pq.Config(cutoff=cutoff, hbar=2)).execute(program).state
    np.testing.assert_allclose(result.state.state_vector, raw.state_vector, atol=1e-13)
    assert abs(np.vdot(initial, expected)) ** 2 < 0.99


def test_kerr_exact_nontrivial_and_native():
    source = pg.FockInput(tuple(np.ones(5, complex) / np.sqrt(5)))
    kappa = 0.71
    output = run([pg.Prepare(0, state=source), pg.Kerr(0, kappa)], 10).state
    expected = np.array(source.amplitudes) * np.exp(1j * kappa * np.arange(5) ** 2)
    np.testing.assert_allclose(output.state_vector[:5], expected, atol=1e-14)
    with pq.Program() as program:
        for n, a in enumerate(source.amplitudes):
            pq.Q() | pq.NumberState((n,)) * a
        pq.Q(0) | pq.Kerr(kappa)
    raw = pq.PureFockSimulator(d=1, config=pq.Config(cutoff=10)).execute(program).state
    np.testing.assert_allclose(output.state_vector, raw.state_vector)


def test_gaussian_fock_gates_against_independent_operators():
    alpha, r = 0.2 + 0.3j, 0.18
    output = run(
        [pg.Prepare(0, 0), pg.Squeeze(0, r), pg.Displace(0, q=2 * alpha.real, p=2 * alpha.imag)], 28
    ).state
    vacuum = np.eye(28, dtype=complex)[:, 0]
    expected = displacement(28, alpha) @ squeeze(28, r) @ vacuum
    assert abs(np.vdot(expected, output.state_vector)) ** 2 > 1 - 1e-12


def test_ideal_ladders_sqrt_factors_and_failures():
    source = pg.FockInput((0, np.sqrt(0.4), 0, np.sqrt(0.6)))
    initial = np.zeros(9, complex)
    initial[:4] = source.amplitudes
    a, ad, *_ = operators(9)
    for command, matrix in ((pg.PhotonAdd(0), ad), (pg.PhotonSubtract(0), a)):
        out = run([pg.Prepare(0, state=source), command], 9).state
        vector = matrix @ initial
        expected_norm = np.vdot(vector, vector).real
        np.testing.assert_allclose(out.state_vector, vector / np.sqrt(expected_norm))
        assert np.isclose(out.diagnostics[-1]["ladder_norm_squared"], expected_norm)
    with pytest.raises(ValueError, match="zero norm"):
        run([pg.Prepare(0, 0), pg.PhotonSubtract(0)])
    with pytest.raises(ValueError, match="cutoff"):
        run([pg.Prepare(0, state=pg.FockInput.number(4)), pg.PhotonAdd(0)], 5)
    assert np.allclose(
        source.photon_subtracted().amplitudes, (np.sqrt(0.4 / 2.2), 0, np.sqrt(1.8 / 2.2))
    )


def test_physical_heralding_probability_and_nonlinear_signal():
    theta = 0.31
    p = pg.non_gaussian.photon_subtraction(theta)
    p.commands.pop()
    p.append(pg.Kerr("in", 0.25 * (-1) ** pg.Outcome("count"))).append(pg.Output(("in",)))
    result = pg.simulate(
        p,
        backend="piquasso-fock",
        cutoff=8,
        inputs={"in": pg.FockInput.number(3)},
        measurement_outcomes={"count": 1},
    )
    assert np.isclose(np.exp(result.log_likelihood), 3 * np.sin(theta) ** 2 * np.cos(theta) ** 4)
    assert np.isclose(result.state.photon_number("in"), 2)
    assert result.measurement_statistics["count"]["kind"] == "probability"
    assert p.dependencies().has_edge(2, 3)
    restored = pg.Pattern.from_json(p.to_json())
    assert restored.inspect() == p.inspect()
    with pytest.raises(ValueError, match="zero probability"):
        pg.simulate(
            p,
            backend="piquasso-fock",
            cutoff=8,
            inputs={"in": pg.FockInput.number(0)},
            measurement_outcomes={"count": 1},
        )


def test_homodyne_entangled_conditional_state_and_angle():
    resource = pg.FockSuperposition.from_mapping({(0, 0): np.sqrt(0.4), (1, 1): np.sqrt(0.6)})
    theta, x = 0.6, 0.7
    p = pg.Pattern().append(pg.PrepareResource((0, 1), resource)).measure(0, pg.Homodyne(theta))
    result = pg.simulate(p, backend="piquasso-fock", cutoff=8, measurement_outcomes={0: x})
    expected = np.zeros(8, complex)
    expected[:2] = [np.sqrt(0.4), np.sqrt(0.6) * x * np.exp(-1j * theta)]
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(result.state.state_vector, expected, atol=1e-13)
    density = np.exp(-x * x / 2) / np.sqrt(2 * np.pi) * (0.4 + 0.6 * x * x)
    assert np.isclose(np.exp(result.log_likelihood), density)
    assert result.measurement_statistics[0]["kind"] == "density"


def test_homodyne_sampling_quantiles_seed_and_clean_failure():
    from scipy.special import ndtri

    p = pg.Pattern().append(pg.Prepare(0, 0)).measure(0)
    for seed in range(8):
        r = pg.simulate(p, backend="piquasso-fock", cutoff=6, seed=seed)
        assert abs(r.outcomes[0] - ndtri(np.random.default_rng(seed).random())) < 1e-8
    p = pg.Pattern().append(pg.Prepare(0, 0)).measure(0, pg.Homodyne(0, efficiency=0.9))
    backend = PiquassoFockBackend(8)
    with pytest.raises(NotImplementedError, match="noisy_homodyne"):
        pg.simulate(p, backend=backend)
    assert backend.nodes == ()


def test_raw_piquasso_count_statistics():
    theta = 0.4
    with pq.Program() as program:
        pq.Q() | pq.NumberState((2, 0))
        pq.Q(0, 1) | pq.Beamsplitter(theta)
        pq.Q(1) | pq.ParticleNumberMeasurement()
    raw = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=5, seed_sequence=41)).execute(
        program, shots=4000
    )
    samples = np.asarray(raw.samples)[:, 0]
    expected = 2 * np.sin(theta) ** 2
    assert abs(samples.mean() - expected) < 0.04
    pattern = pg.non_gaussian.photon_subtraction(theta)
    for k in range(3):
        result = pg.simulate(
            pattern,
            backend="piquasso-fock",
            cutoff=5,
            inputs={"in": pg.FockInput.number(2)},
            measurement_outcomes={"count": k},
        )
        empirical = np.mean(samples == k)
        assert abs(np.exp(result.log_likelihood) - empirical) < 0.025
