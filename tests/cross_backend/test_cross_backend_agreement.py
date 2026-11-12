"""Cross-backend agreement over every AVAILABLE registered backend.

References: the NumPy ``gaussian`` backend for Gaussian patterns (its output is
converted to the Fock basis with Piquasso's Gaussian-state ``density_matrix``, a code
path independent of gate-by-gate Fock evolution), and closed forms where they exist
(cat photon counting, single-mode homodyne densities).

Fock tolerances come from the observed total-photon-cutoff convergence of the
Fock backends against those references (max abs element error). For ``graph`` the
native and Piquasso mixed engines truncate the CZ differently (native CZ elements are
exact projections, closer to the reference); at cutoff 12 their homodyne-density errors
are 3.5e-6 (native) and 5.5e-7 (Piquasso), at cutoff 16 5.7e-8 and 1.8e-7, so the case
runs at 16 with tolerances set from the larger of the two observed errors:

==============  ======  ========  ========  ========  ========
case            cutoff  rho err   (c-4)     density   tolerance (rho, density)
==============  ======  ========  ========  ========  ========
wire            16      1.5e-4    9.3e-4    2.5e-7    5e-4, 1e-6
graph (mixed)   16      2.8e-6    5.5e-5    1.8e-7    2e-5, 1e-6
cat counting    16      1.2e-8    7.1e-6    3e-17     1e-7, 1e-12
homodyne 1-mode 20      --        --        1.5e-8    1e-7 (c=12: 1.7e-5)
==============  ======  ========  ========  ========  ========

Gaussian-representation backends must agree with the reference to float roundoff.
Backends lacking a required feature are skipped with the missing feature named;
``test_expected_coverage`` pins which backends must run each case.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import photographiq as pg
from photographiq import registry
from photographiq.capabilities import required_features

ROUNDOFF = 1e-10  # Gaussian moments: a few dozen float64 matrix products of O(1) entries


def wire():
    pattern = pg.Pattern(inputs=(0,)).extend(
        [
            pg.Prepare(1, 0.3),
            pg.Squeeze(0, 0.1),
            pg.Rotate(0, 0.3),
            pg.Entangle(0, 1, 0.5),
            pg.Measure(0, pg.Homodyne(np.pi / 2), key="m"),
        ]
    )
    return pattern, {0: pg.GaussianInput.coherent(0.2 + 0.1j)}, {"m": 0.25}


def graph():
    # Lazily prepared CZ chain a-b-c: at most two live modes keeps mixed Fock cheap.
    pattern = pg.Pattern(inputs=()).extend(
        [
            pg.Prepare("a", 0.2),
            pg.Prepare("b", 0.2),
            pg.Entangle("a", "b", 0.4),
            pg.Loss("b", 0.8, 0.05),
            pg.Measure("a", pg.Homodyne(0.3, efficiency=0.9), key="m"),
            pg.Prepare("c", 0.2),
            pg.Entangle("b", "c", 0.4),
        ]
    )
    return pattern, {}, {"m": 0.1}


GAUSSIAN_CASES = {
    "wire": (wire, 16, 5e-4, 1e-6),
    "graph": (graph, 16, 2e-5, 1e-6),
}

EXPECTED = {
    # Backends that must cover each case whenever they are available.
    "wire": {
        "gaussian",
        "piquasso",
        "piquasso-fock",
        "piquasso-mixed-fock",
        "native-gaussian",
        "native-fock",
        "native-mixed-fock",
    },
    "graph": {
        "gaussian",
        "piquasso",
        "piquasso-mixed-fock",
        "native-gaussian",
        "native-mixed-fock",
    },
}


def _run(name, pattern, inputs, outcomes, cutoff):
    info = registry.BACKENDS[name]
    reason = info.unavailable_reason()
    if reason is not None:
        pytest.skip(f"{name}: {reason}")
    missing = registry.missing_features(name, required_features(pattern, inputs, None, outcomes))
    if missing:
        pytest.skip(f"{name} lacks {sorted(missing)}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # retained-norm diagnostics
        return pg.simulate(
            pattern,
            backend=name,
            inputs=inputs,
            measurement_outcomes=outcomes,
            cutoff=cutoff if info.uses_cutoff else None,
            seed=1,
        )


def _gaussian_density_matrix(state, cutoff):
    """Fock density matrix of a GaussianState and its occupation basis (via Piquasso)."""
    pq = pytest.importorskip("piquasso")
    native = pq.GaussianState(
        d=len(state.nodes),
        connector=pq.NumpyConnector(),
        config=pq.Config(hbar=2.0, cutoff=cutoff),
    )
    native.xpxp_mean_vector = state.mean
    native.xpxp_covariance_matrix = 2 * state.covariance
    basis = [tuple(int(n) for n in b) for b in native.fock_probabilities_map]
    return {b: i for i, b in enumerate(basis)}, np.asarray(native.density_matrix)


def test_gaussian_to_fock_reference_matches_coherent_closed_form():
    alpha = 0.2 + 0.1j
    state = pg.GaussianInput.coherent(alpha).state("x")
    index, rho = _gaussian_density_matrix(state, 12)
    n = np.arange(12)
    from scipy.special import gammaln

    psi = np.exp(-(abs(alpha) ** 2) / 2 + n * np.log(alpha) - gammaln(n + 1) / 2)
    order = [index[(k,)] for k in n]
    np.testing.assert_allclose(rho[np.ix_(order, order)], np.outer(psi, psi.conj()), atol=1e-14)


@pytest.mark.parametrize("case", sorted(GAUSSIAN_CASES))
def test_expected_coverage(case):
    pattern, inputs, outcomes = GAUSSIAN_CASES[case][0]()
    features = required_features(pattern, inputs, None, outcomes)
    for name in registry.available_backends():
        covered = not registry.missing_features(name, features)
        assert covered == (name in EXPECTED[case]), name


@pytest.mark.parametrize("name", list(registry.BACKENDS))
@pytest.mark.parametrize("case", sorted(GAUSSIAN_CASES))
def test_gaussian_pattern_agreement(case, name):
    build, cutoff, rho_tol, density_tol = GAUSSIAN_CASES[case]
    pattern, inputs, outcomes = build()
    reference = pg.simulate(
        pattern, backend="gaussian", inputs=inputs, measurement_outcomes=outcomes
    )
    result = _run(name, pattern, inputs, outcomes, cutoff)
    density = result.measurement_statistics["m"]["value"]
    expected_density = reference.measurement_statistics["m"]["value"]
    assert result.measurement_statistics["m"]["postselected"]
    if registry.BACKENDS[name].representation == "gaussian":
        assert result.state.nodes == reference.state.nodes
        np.testing.assert_allclose(result.state.mean, reference.state.mean, atol=ROUNDOFF)
        np.testing.assert_allclose(
            result.state.covariance, reference.state.covariance, atol=ROUNDOFF
        )
        assert density == pytest.approx(expected_density, abs=ROUNDOFF)
        return
    index, rho = _gaussian_density_matrix(reference.state, 3 * cutoff)
    order = [index[tuple(int(n) for n in b)] for b in result.state.basis]
    np.testing.assert_allclose(result.state.density_matrix, rho[np.ix_(order, order)], atol=rho_tol)
    assert density == pytest.approx(expected_density, abs=density_tol)


def test_wire_cutoff_convergence_justifies_tolerance():
    pattern, inputs, outcomes = wire()
    reference = pg.simulate(
        pattern, backend="gaussian", inputs=inputs, measurement_outcomes=outcomes
    )
    index, rho = _gaussian_density_matrix(reference.state, 48)
    errors = []
    for cutoff in (8, 12, 16):
        result = _run("piquasso-fock", pattern, inputs, outcomes, cutoff)
        order = [index[tuple(int(n) for n in b)] for b in result.state.basis]
        errors.append(np.abs(result.state.density_matrix - rho[np.ix_(order, order)]).max())
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < GAUSSIAN_CASES["wire"][2] / 2


def cat_counting():
    pattern = pg.Pattern(inputs=()).extend(
        [
            pg.Prepare("a", 0.0, pg.CatResource(0.8)),
            pg.Prepare("b", 0.0),
            pg.BeamSplitter("a", "b", 0.4),
            pg.Measure("b", pg.PhotonNumber(), key="n"),
        ]
    )
    return pattern, {}, {"n": 1}


@pytest.mark.parametrize("name", list(registry.BACKENDS))
def test_cat_photon_counting_closed_form(name):
    """Even cat |a>+|-a> on BS(theta) with vacuum; count one photon in b.

    |a,0> -> |a c, a s>, so <1|_b gives (a s) e^{-|a s|^2/2} (|a c> - |-a c>):
    an odd cat of amplitude a cos(theta) with probability
    N^2 e^{-|as|^2} |as|^2 2 (1 - e^{-2|ac|^2}), N^2 = 1 / (2 (1 + e^{-2|a|^2})).
    """
    pattern, inputs, outcomes = cat_counting()
    result = _run(name, pattern, inputs, outcomes, 16)
    alpha, theta = 0.8, 0.4
    ac, as_ = alpha * np.cos(theta), alpha * np.sin(theta)
    norm2 = 1 / (2 * (1 + np.exp(-2 * alpha**2)))
    probability = norm2 * np.exp(-(as_**2)) * as_**2 * 2 * (1 - np.exp(-2 * ac**2))
    assert result.measurement_statistics["n"]["value"] == pytest.approx(probability, abs=1e-12)
    exact = np.asarray(pg.FockInput.cat(ac, 80, -1).amplitudes)
    order = [int(b[0]) for b in result.state.basis]
    psi = exact[order]
    np.testing.assert_allclose(result.state.density_matrix, np.outer(psi, psi.conj()), atol=1e-7)


@pytest.mark.parametrize("efficiency", [1.0, 0.8])
@pytest.mark.parametrize("name", list(registry.BACKENDS))
def test_homodyne_postselection_density_closed_form(name, efficiency):
    """Density of x = n.r + noise, n=(cos t, sin t): N(x; n.mu, n^T V n + (1-eta)/eta)."""
    angle = 0.7
    squeezed = pg.GaussianInput.squeezed(0.2, 0.4)
    mean = np.array([0.3, -0.2])
    source = pg.GaussianInput(tuple(mean), squeezed.covariance)
    pattern = pg.Pattern(inputs=("x",)).extend(
        [pg.Measure("x", pg.Homodyne(angle, efficiency=efficiency), key="m")]
    )
    direction = np.array([np.cos(angle), np.sin(angle)])
    mu = direction @ mean
    variance = (
        direction @ np.asarray(squeezed.covariance) @ direction + (1 - efficiency) / efficiency
    )
    for outcome in (-1.0, 0.2, 1.5):
        result = _run(name, pattern, {"x": source}, {"m": outcome}, 20)
        expected = np.exp(-((outcome - mu) ** 2) / (2 * variance)) / np.sqrt(2 * np.pi * variance)
        assert result.measurement_statistics["m"]["value"] == pytest.approx(expected, abs=1e-7)
