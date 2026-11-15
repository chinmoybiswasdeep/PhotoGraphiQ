"""Fock-state boson sampling against an independent creation-operator expansion and Piquasso."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
import pytest
from scipy import stats

from photographiq.sampling import (
    collision_probabilities,
    haar_unitary,
    output_distribution,
    output_patterns,
    output_probability,
    sample,
    transition_amplitude,
)


def _expanded_amplitudes(u: np.ndarray, inputs) -> dict[tuple[int, ...], complex]:
    """prod_j (sum_k U_kj a_k^dag)^{s_j} / sqrt(s_j!) |0>, expanded monomial by monomial."""
    m = len(u)
    poly: dict[tuple[int, ...], complex] = {(0,) * m: 1.0}
    for j, s in enumerate(inputs):
        for _ in range(s):
            nxt: dict[tuple[int, ...], complex] = defaultdict(complex)
            for mono, c in poly.items():
                for k in range(m):
                    key = list(mono)
                    key[k] += 1
                    nxt[tuple(key)] += c * u[k, j]
            poly = nxt
    norm = math.sqrt(math.prod(math.factorial(s) for s in inputs))
    # (a^dag)^t |0> = sqrt(t!) |t>
    return {
        t: c * math.sqrt(math.prod(math.factorial(x) for x in t)) / norm for t, c in poly.items()
    }


@pytest.mark.parametrize("inputs", [(1, 1, 0, 0), (1, 1, 1, 0), (2, 1, 0, 0), (0, 3, 0, 1)])
def test_amplitudes_match_creation_operator_expansion(inputs):
    u = haar_unitary(4, rng=sum(inputs) * 7 + inputs[0])
    reference = _expanded_amplitudes(u, inputs)
    for t, amplitude in reference.items():
        assert abs(transition_amplitude(u, inputs, t) - amplitude) < 1e-12


def test_distribution_normalization_and_hong_ou_mandel():
    u = haar_unitary(5, rng=9)
    dist = output_distribution(u, (1, 1, 1, 0, 0))
    assert len(dist) == math.comb(3 + 5 - 1, 3)
    assert math.fsum(dist.values()) == pytest.approx(1.0, abs=1e-13)
    split = collision_probabilities(dist)
    assert split["collision_free"] + split["collision"] == pytest.approx(1.0, abs=1e-13)
    # HOM: a 50:50 beamsplitter never outputs (1, 1).
    bs = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
    assert output_probability(bs, (1, 1), (1, 1)) < 1e-30
    assert output_probability(bs, (1, 1), (2, 0)) == pytest.approx(0.5, abs=1e-15)
    assert output_probability(u, (1, 1, 1, 0, 0), (1, 1, 0, 0, 0)) == 0.0  # photon number


def test_output_patterns_enumeration():
    patterns = output_patterns(3, 4)
    assert len(patterns) == len(set(patterns)) == math.comb(6, 2)
    assert all(sum(p) == 4 and len(p) == 3 for p in patterns)


def test_matches_piquasso_fock_simulation():
    pq = pytest.importorskip("piquasso")
    u = haar_unitary(3, rng=4)
    inputs = (2, 1, 0)
    with pq.Program() as program:
        pq.Q() | pq.NumberState(list(inputs))
        pq.Q(0, 1, 2) | pq.Interferometer(u)
    state = pq.PureFockSimulator(d=3, config=pq.Config(cutoff=4, hbar=2.0)).execute(program).state
    reference = state.fock_probabilities_map
    for t, p in output_distribution(u, inputs).items():
        assert p == pytest.approx(reference[t], abs=1e-12)


def _chi_square_pvalue(samples: np.ndarray, dist: dict) -> float:
    counts = Counter(map(tuple, samples))
    observed = np.array([counts.get(t, 0) for t in dist])
    expected = np.array(list(dist.values())) * len(samples)
    keep = expected >= 5  # pool rare outcomes to keep the chi-square approximation valid
    obs = np.r_[observed[keep], observed[~keep].sum()]
    exp = np.r_[expected[keep], expected[~keep].sum()]
    if exp[-1] == 0:
        obs, exp = obs[:-1], exp[:-1]
    return stats.chisquare(obs, exp).pvalue


@pytest.mark.parametrize(
    ("inputs", "method"),
    [((1, 1, 1, 0, 0), "clifford"), ((1, 0, 1, 0, 1), "auto"), ((2, 1, 0, 0), "enumerate")],
)
def test_exact_samplers_reproduce_distribution(inputs, method):
    u = haar_unitary(len(inputs), rng=17)
    dist = output_distribution(u, inputs)
    samples = sample(u, inputs, 8000, rng=123, method=method)
    assert samples.shape == (8000, len(inputs))
    assert (samples.sum(1) == sum(inputs)).all()
    assert _chi_square_pvalue(samples, dist) > 1e-3


def test_chi_square_check_rejects_distinguishable_particles():
    """Negative control: the sampler test above has power against a wrong distribution."""
    u = haar_unitary(5, rng=17)
    inputs = (1, 1, 1, 0, 0)
    rng = np.random.default_rng(5)
    routes = [rng.choice(5, size=8000, p=np.abs(u[:, j]) ** 2) for j in range(3)]
    samples = np.zeros((8000, 5), dtype=np.int64)
    for route in routes:
        np.add.at(samples, (np.arange(8000), route), 1)
    assert _chi_square_pvalue(samples, output_distribution(u, inputs)) < 1e-6


def test_clifford_rejects_collision_inputs():
    with pytest.raises(ValueError):
        sample(np.eye(2), (2, 0), 1, method="clifford")
