"""CV kernels (closed-form Gaussian overlaps) and non-Gaussian Fock programs vs references."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import photographiq as pg
from photographiqml.cv import (
    CVCluster,
    CVFeatureMap,
    CVKernel,
    CVRegressor,
    NonGaussianProgram,
    layered_program,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "validation"))
from cv_fock_reference import FockReference  # noqa: E402

pytest.importorskip("photographiq._native")


# ---------------------------------------------------------------- kernels
def test_coherent_overlap_kernel_is_the_gaussian_rbf():
    X = np.random.default_rng(0).uniform(-1, 1, (7, 2))
    kernel = CVKernel(CVFeatureMap(2, scale=1.3))
    gram = kernel.gram_matrix(X)
    distance = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.testing.assert_allclose(gram, np.exp(-(1.3**2) * distance / 4), atol=1e-13)
    np.testing.assert_allclose(kernel(X[:3], X), gram[:3], atol=1e-14)


def test_unitary_invariance_and_finite_squeezing_dependence():
    X = np.random.default_rng(1).uniform(-1, 1, (6, 2))
    fmap = CVFeatureMap(2, scale=0.9)
    program = layered_program(2, 1)
    params = {n: 0.3 for n in program.parameters}
    base = CVKernel(fmap).gram_matrix(X)
    ideal = CVKernel(fmap, program, params).gram_matrix(X)
    np.testing.assert_allclose(ideal, base, atol=1e-12)  # Tr[U rho U^+ U sigma U^+] = Tr[rho sigma]
    previous = None
    for r in (0.5, 1.0, 2.0):  # mixed outputs: purity < 1, kernel approaches the ideal one
        physical = CVKernel(fmap, program, params, mode="physical", squeezing=r)
        diagnostics = physical.diagnostics(X)
        assert diagnostics["symmetry_error"] < 1e-14
        assert diagnostics["minimum_eigenvalue"] > -1e-12
        assert diagnostics["max_diagonal"] < 1
        error = np.abs(physical.gram_matrix(X) - base).max()
        assert previous is None or error < previous
        previous = error


def test_fidelity_versus_hilbert_schmidt_for_mixed_states():
    X = np.array([[0.2, -0.4], [0.5, 0.1]])
    fmap = CVFeatureMap(2, scale=1.0)
    pure_hs = CVKernel(fmap, kind="overlap").gram_matrix(X)
    pure_f = CVKernel(fmap, kind="fidelity").gram_matrix(X)
    np.testing.assert_allclose(pure_hs, pure_f, atol=1e-12)  # identical when a state is pure
    cluster = CVCluster(2, 3, squeezing=0.3)
    params = dict.fromkeys(cluster.parameters, 0.2)
    hs = CVKernel(fmap, cluster, params, kind="overlap", mode="physical").gram_matrix(X)
    fid = CVKernel(fmap, cluster, params, kind="fidelity", mode="physical").gram_matrix(X)
    norm = CVKernel(fmap, cluster, params, kind="normalized", mode="physical").gram_matrix(X)
    np.testing.assert_allclose(np.diag(fid), 1, atol=1e-9)
    np.testing.assert_allclose(np.diag(norm), 1, atol=1e-12)
    assert np.all(np.diag(hs) < 0.9)  # purity, not 1
    assert not np.allclose(hs, fid)


# ---------------------------------------------------------------- non-Gaussian programs
def reference_moments(gates, d0, cutoff=90):
    ref = FockReference(cutoff)
    psi = ref.displace(*d0) @ ref.vacuum()
    for kind, value in gates:
        psi = getattr(ref, kind)(value) @ psi
    return ref.moments(psi)


@pytest.mark.parametrize(
    "gates",
    [
        [("rotate", 0.7), ("squeeze", 0.3)],
        [("kerr", 0.15), ("rotate", 0.4)],
        [("cubic", 0.08), ("squeeze", -0.2), ("kerr", 0.05)],
    ],
)
def test_fock_engine_matches_the_matrix_exponential_reference(gates):
    program = NonGaussianProgram(1)
    method = {"rotate": "rotate", "squeeze": "squeeze", "kerr": "kerr", "cubic": "cubic_phase"}
    for kind, value in gates:
        getattr(program, method[kind])(0, value)
    fmap = CVFeatureMap(1, scale=1.0)
    x = np.array([0.9])
    expected = reference_moments(gates, (0.9, 0.0))
    for cutoff in (40, 60):
        means, variances, norm = program.fock_moments({}, x, fmap, cutoff, (0,))
        np.testing.assert_allclose(means, expected[:2], atol=2e-6)
        np.testing.assert_allclose(variances, expected[2:4], atol=2e-5)
        assert norm == pytest.approx(1, abs=1e-6)


def test_cutoff_convergence_and_retained_norm_reporting():
    program = NonGaussianProgram(1).kerr(0, 0.2).squeeze(0, 0.4)
    fmap = CVFeatureMap(1, scale=2.0)
    values = [program.fock_moments({}, np.array([1.0]), fmap, c, (0,)) for c in (16, 20, 32, 48)]
    errors = [abs(v[0][0] - values[-1][0][0]) for v in values[:-1]]
    assert errors[0] > errors[1] > errors[2] and errors[2] < 1e-6
    assert 1 - 1e-3 < values[0][2] < 1 - 1e-6  # truncation is visible in the retained norm
    assert values[-1][2] > 1 - 1e-12


def test_gaussian_only_program_agrees_between_fock_and_moment_execution():
    program = NonGaussianProgram(2).rotate(0, 0.3).beamsplitter(0, 1, 0.7).squeeze(1, 0.2)
    fmap = CVFeatureMap(2, scale=0.8)
    x = np.array([0.5, -0.3])
    X, t, _, _ = program.channel({}, x)
    d, _ = fmap.moments(x)
    means, _, _ = program.fock_moments({}, x, fmap, 30, (0, 1))
    np.testing.assert_allclose(means, X @ d + t, atol=1e-8)
    with pytest.raises(NotImplementedError):
        NonGaussianProgram(1).kerr(0, 0.1).channel({}, x[:1])
    with pytest.raises(NotImplementedError):  # capability rejection on a Gaussian engine
        pg.simulate(NonGaussianProgram(1).kerr(0, 0.1).pattern({}), backend="gaussian")


def test_non_gaussian_regressor_expresses_a_nonlinear_map_gaussian_cannot():
    X = np.linspace(-1, 1, 9)[:, None]
    y = X[:, 0] ** 2
    gaussian = CVRegressor(layered_program(1, 1), CVFeatureMap(1, scale=1.0), epochs=40).fit(X, y)
    # Kerr, rotation and squeezing commute with parity, so <q> would be odd in the input
    # amplitude; a trainable displacement before the Kerr gate breaks that symmetry.
    program = NonGaussianProgram(1).displace(0, "b", 0.0).kerr(0, "k").rotate(0, "r")
    kerr = CVRegressor(
        program,
        CVFeatureMap(1, scale=1.0),
        cutoff=40,
        epochs=40,
        learning_rate=0.1,
        init_scale=0.3,
        seed=1,
    ).fit(X, y)
    assert kerr.history.settings["gradient"] == "finite-difference"
    assert kerr.score(X, y)["mse"] < 0.5 * gaussian.score(X, y)["mse"]
    assert min(kerr.retained_norms) > 1 - 1e-6
    with pytest.raises(ValueError):
        CVRegressor(program, CVFeatureMap(1), cutoff=24, gradient="complex-step")
    with pytest.raises(ValueError):
        CVRegressor(program, CVFeatureMap(1))  # cutoff required


def test_cubic_injection_lowering_tracks_its_outcome_record():
    program = NonGaussianProgram(1).cubic_phase(0, 0.1)
    fmap = CVFeatureMap(1, scale=0.5)
    x = np.array([0.4])
    pattern = program.pattern({}, x, cubic="injection", injection_squeezing=0.3)
    kinds = [type(c).__name__ for c in pattern.commands]
    assert (
        kinds.count("Measure") == 1
        and "QuadraticPhase" in kinds
        and pattern.ancilla_keys == ("m0",)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        channel = program.injection_channel(
            {}, x, fmap, 40, injection_squeezing=0.3, outcomes=np.linspace(-7, 7, 57)
        )
    assert channel["outcome_probability"] == pytest.approx(1, abs=0.02)
    rho = channel["density_matrix"]
    assert np.trace(rho).real == pytest.approx(1)
    assert np.linalg.eigvalsh((rho + rho.conj().T) / 2).min() > -1e-9
