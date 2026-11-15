"""Gaussian boson sampling: closed forms, an independent Fock-space construction, Piquasso.

The Fock reference evolves a two-mode state vector with dense matrix exponentials of
truncated ladder-operator generators in the PhotoGraphiQ gate conventions (per-mode
cutoff 36), and applies pure loss *after* all gates as binomial thinning of the
photon-number distribution (exact: loss commutes with photon counting). Its truncation
error is checked against per-mode cutoff 30 (observed 3e-15 for the probabilities
compared; cutoff 24 still differs by 4e-12). ``expm_multiply`` was rejected because its
~1e-12 accuracy floor exceeds the comparison tolerance.

Closed-form tolerance: with the Glynn-type power-trace hafnian, observed absolute errors
of N-photon probabilities on the single-mode squeezed state (r = 1, method ``"mixed"``,
``2N``-dimensional hafnians) are <= 2.2e-16 for N <= 11 and 2.9e-15 at N = 16; the
tests use 1e-14 for N <= 11. (The earlier subset/eigenvalue variant reached 1.3e-12 at
N = 11 and was replaced for that reason.)
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections import Counter
from typing import cast

import numpy as np
import pytest
from scipy import stats
from scipy.linalg import expm

from photographiq import native
from photographiq.sampling import (
    ThresholdPrecisionWarning,
    apply_loss,
    clements_decomposition,
    gbs_matrices,
    haar_unitary,
    pnr_distribution,
    pnr_probability,
    sample_pnr,
    sample_threshold,
    squeezed_interferometer_state,
    threshold_distribution,
    threshold_probability,
)
from photographiq.states import GaussianState


def _native_available() -> bool:
    try:
        native.require("sampling")
    except ImportError:
        return False
    return True


@pytest.fixture(
    params=[
        "python",
        pytest.param(
            "native",
            marks=pytest.mark.skipif(not _native_available(), reason="native sampling not built"),
        ),
    ]
)
def backend(request):
    return request.param


# ------------------------------------------------------------------ closed-form states


@pytest.mark.parametrize("r", [0.3, 1.0, -0.7])
def test_single_mode_squeezed_vacuum_closed_form(backend, r):
    state = squeezed_interferometer_state([r], np.eye(1))
    t = math.tanh(abs(r))
    for n in range(12):
        exact = (
            0.0
            if n % 2
            else t**n * math.factorial(n) / (math.cosh(r) * 2**n * math.factorial(n // 2) ** 2)
        )
        for method in ("pure", "mixed"):
            got = pnr_probability(state, [n], method=method, backend=backend)
            assert got == pytest.approx(exact, abs=1e-14, rel=1e-11)


def test_two_mode_squeezed_vacuum_closed_form(backend):
    """S(r) (+) S(-r) followed by a 50:50 beamsplitter is the TMSV of squeezing r."""
    r = 0.8
    bs = np.array([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2)
    state = squeezed_interferometer_state([r, -r], bs)
    for n0, n1 in itertools.product(range(6), repeat=2):
        exact = math.tanh(r) ** (2 * n0) / math.cosh(r) ** 2 if n0 == n1 else 0.0
        for method in ("pure", "mixed"):
            got = pnr_probability(state, [n0, n1], method=method, backend=backend)
            assert got == pytest.approx(exact, abs=1e-14, rel=1e-11)
    # Threshold: P(no click) = 1/cosh^2 r and only correlated clicks occur.
    p00 = 1 / math.cosh(r) ** 2
    assert threshold_probability(state, [0, 0], backend=backend) == pytest.approx(p00, rel=1e-12)
    assert threshold_probability(state, [1, 1], backend=backend) == pytest.approx(1 - p00)
    assert abs(threshold_probability(state, [1, 0], backend=backend)) < 1e-14


def test_coherent_state_is_poissonian(backend):
    alpha = np.array([0.6 - 0.3j, 0.2j])
    mean = np.ravel(np.column_stack([2 * alpha.real, 2 * alpha.imag]))
    state = GaussianState(mean, np.eye(4), (0, 1))
    for n in itertools.product(range(4), repeat=2):
        exact = math.prod(
            math.exp(-(abs(a) ** 2)) * abs(a) ** (2 * k) / math.factorial(k)
            for a, k in zip(alpha, n)
        )
        assert pnr_probability(state, n, backend=backend) == pytest.approx(exact, rel=1e-12)
    click = [1 - math.exp(-(abs(a) ** 2)) for a in alpha]
    assert threshold_probability(state, [1, 1], backend=backend) == pytest.approx(
        click[0] * click[1], rel=1e-12
    )


# ------------------------------------------------------------ GBS matrix construction


def test_gbs_matrices_structure():
    state = squeezed_interferometer_state(
        [0.4, 0.3, 0.2], haar_unitary(3, rng=1), [0.9, 0.8, 0.7], [0.1, 0.2j, -0.3]
    )
    g = gbs_matrices(state)
    q_inv = np.linalg.inv(g.sigma_q)
    p0 = np.exp(-0.5 * (g.beta.conj() @ q_inv @ g.beta).real) / np.sqrt(
        np.linalg.det(g.sigma_q).real
    )
    assert g.vacuum_probability == pytest.approx(p0, rel=1e-13)
    assert np.allclose(g.a, g.a.T, atol=1e-14)
    assert np.allclose(g.sigma_q, g.sigma_q.conj().T, atol=1e-14)
    assert not g.pure
    pure = gbs_matrices(squeezed_interferometer_state([0.4, 0.3], haar_unitary(2, rng=2)))
    assert pure.pure and np.allclose(gbs_matrices(GaussianState(np.zeros(2), np.eye(2), (0,))).a, 0)
    with pytest.raises(ValueError):
        pnr_probability(state, [1, 1, 0], method="pure")


# ------------------------------------------------------------- Fock-space reference


def _fock_reference(squeezing, unitary, displacement, cutoff: int) -> np.ndarray:
    """|psi(n0, n1)|^2 after Squeeze, Clements interferometer, Displace (2 modes)."""
    a = np.diag(np.sqrt(np.arange(1, cutoff)), 1).astype(complex)
    n = a.T @ a
    psi = np.zeros((cutoff, cutoff), dtype=complex)  # psi[n0, n1]
    psi[0, 0] = 1.0

    def local(mode, generator):
        u = expm(generator)
        return u @ psi if mode == 0 else psi @ u.T

    for j, r in enumerate(squeezing):  # S(r): q -> e^{-r} q
        psi = local(j, r / 2 * (a @ a - a.T @ a.T))
    a0, a1 = np.kron(a, np.eye(cutoff)), np.kron(np.eye(cutoff), a)
    for op in clements_decomposition(unitary).operations():
        if op[0] == "rotate":  # exp(i phi n): a -> e^{i phi} a
            psi = local(op[1], 1j * op[2] * n)
        else:  # (u, v) = (0, 1): a_u -> c a_u - s a_v, a_v -> s a_u + c a_v
            theta = op[3]
            psi = (expm(theta * (a0 @ a1.T - a0.T @ a1)) @ psi.ravel()).reshape(psi.shape)
    for j, alpha in enumerate(displacement):
        psi = local(j, alpha * a.T - np.conj(alpha) * a)
    return np.abs(psi) ** 2


def _thinned(probabilities: np.ndarray, eta) -> np.ndarray:
    """Pure loss on photon-number distributions: binomial thinning per mode."""
    out = probabilities
    for axis, e in enumerate(eta):
        n = np.arange(probabilities.shape[axis])
        kernel = stats.binom.pmf(n[:, None], n[None, :], e)  # kernel[k_out, k_in]
        out = np.moveaxis(np.tensordot(kernel, out, axes=([1], [axis])), 0, axis)
    return out


_FOCK_CASE = ([0.5, -0.3], haar_unitary(2, rng=8), [0.4 - 0.2j, 0.3j])


@pytest.fixture(scope="module")
def fock_reference():
    fine, coarse = (_fock_reference(*_FOCK_CASE, cutoff) for cutoff in (36, 30))
    assert np.abs(fine[:5, :5] - coarse[:5, :5]).max() < 1e-14  # reference converged
    return fine


def test_pnr_matches_independent_fock_construction(backend, fock_reference):
    squeezing, unitary, displacement = _FOCK_CASE
    eta = [0.85, 0.6]
    fock = fock_reference
    lossy_fock = _thinned(fock, eta)
    pure = squeezed_interferometer_state(squeezing, unitary, displacement=displacement)
    lossy = apply_loss(pure, eta)  # loss after displacement, as in the reference
    for n in itertools.product(range(5), repeat=2):
        assert pnr_probability(pure, n, backend=backend) == pytest.approx(fock[n], abs=1e-13)
        assert pnr_probability(pure, n, method="mixed", backend=backend) == pytest.approx(
            fock[n], abs=1e-13
        )
        assert pnr_probability(lossy, n, backend=backend) == pytest.approx(lossy_fock[n], abs=1e-13)
    # Threshold probabilities equal sums of PNR probabilities over the click support.
    for clicks in itertools.product((0, 1), repeat=2):
        mask = np.ones_like(lossy_fock, dtype=bool)
        for axis, c in enumerate(clicks):
            levels = np.arange(lossy_fock.shape[axis])
            sel = (levels > 0) if c else (levels == 0)
            mask &= np.expand_dims(sel, 1 - axis)
        expected = lossy_fock[mask].sum()
        got = threshold_probability(lossy, clicks, backend=backend)
        assert got == pytest.approx(expected, abs=1e-12)


# ----------------------------------------------------------------- Piquasso reference


def _piquasso_program(pq, squeezing, unitary, eta, displacement):
    m = len(squeezing)
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        for i, r in enumerate(squeezing):
            pq.Q(i) | pq.Squeezing(r=r)
        pq.Q(*range(m)) | pq.Interferometer(unitary)
        for i, e in enumerate(eta):
            if e != 1:
                pq.Q(i) | pq.Attenuator(theta=np.arccos(np.sqrt(e)))
        for i, a in enumerate(displacement):
            pq.Q(i) | pq.Displacement(r=abs(a), phi=np.angle(a))
    return program


@pytest.mark.parametrize("displaced", [False, True])
def test_matches_piquasso_gaussian_simulation(backend, displaced):
    pq = pytest.importorskip("piquasso")
    squeezing, unitary, eta = [0.4, 0.2, 0.5], haar_unitary(3, rng=1), [0.9, 0.7, 0.8]
    displacement = [0.3 + 0.1j, -0.2j, 0.1] if displaced else [0.0, 0.0, 0.0]
    program = _piquasso_program(pq, squeezing, unitary, eta, displacement)
    ref = pq.GaussianSimulator(d=3, config=pq.Config(hbar=2.0)).execute(program).state
    state = squeezed_interferometer_state(squeezing, unitary, eta, displacement)
    assert np.abs(ref.xpxp_covariance_matrix / 2 - state.covariance).max() < 1e-14
    assert np.abs(ref.xpxp_mean_vector - state.mean).max() < 1e-14
    for n in itertools.product(range(3), repeat=3):
        expected = ref.get_particle_detection_probability(n)
        assert pnr_probability(state, n, backend=backend) == pytest.approx(expected, abs=1e-13)
    for c in itertools.product((0, 1), repeat=3):
        expected = ref.get_threshold_detection_probability(c)
        assert threshold_probability(state, c, backend=backend) == pytest.approx(
            expected, abs=1e-13
        )


def test_matches_piquasso_pure_fock_simulation(backend):
    """Lossless, displaced 3-mode state against Piquasso's Fock-space (non-hafnian) simulator."""
    pq = pytest.importorskip("piquasso")
    squeezing, unitary, displacement = [0.3, 0.2, 0.25], haar_unitary(3, rng=5), [0.2, 0.1j, 0]
    program = _piquasso_program(pq, squeezing, unitary, [1, 1, 1], displacement)
    state = squeezed_interferometer_state(squeezing, unitary, displacement=displacement)
    results = {}
    for cutoff in (12, 14):
        config = pq.Config(hbar=2.0, cutoff=cutoff)
        results[cutoff] = pq.PureFockSimulator(d=3, config=config).execute(program).state
    low = [n for n in itertools.product(range(4), repeat=3) if sum(n) <= 4]
    fine, coarse = results[14].fock_probabilities_map, results[12].fock_probabilities_map
    truncation = max(abs(fine[n] - coarse[n]) for n in low)
    assert truncation < 1e-11  # observed 1.7e-12: Piquasso's truncated displacement converged
    for n in low:
        assert pnr_probability(state, n, backend=backend) == pytest.approx(
            fine[n], abs=10 * truncation + 1e-13
        )


# ------------------------------------------------------------- normalization, loss


def test_pnr_normalization_and_truncated_mass(backend):
    state = squeezed_interferometer_state(
        [0.3, 0.25, 0.2], haar_unitary(3, rng=3), [0.95, 0.9, 0.8], [0.1, 0, 0.2j]
    )
    mean_photons = sum(state.photon_number(i) for i in state.nodes)
    previous = None
    for cutoff in (4, 7, 10):
        dist = pnr_distribution(state, cutoff, backend=backend)
        assert len(dist.probabilities) == math.comb(cutoff - 1 + 3, 3)
        assert all(p > -1e-15 for p in dist.probabilities.values())
        assert 0 <= dist.truncated_mass <= mean_photons / cutoff  # Markov bound
        if previous is not None:
            assert dist.truncated_mass < previous
        previous = dist.truncated_mass
    assert dist.truncated_mass < 1e-5  # observed 2.1e-6 at cutoff 10
    partial_mean = sum(sum(n) * p for n, p in dist.probabilities.items())
    assert partial_mean == pytest.approx(mean_photons, abs=1e-4)


def test_threshold_normalization_and_vacuum_inclusion_exclusion(backend):
    """P(S) = sum_{Y subset S} (-1)^{|Y|} P(vacuum on Y and on the complement of S)."""
    state = squeezed_interferometer_state(
        [0.6, -0.4, 0.5, 0.3], haar_unitary(4, rng=6), [0.9, 0.5, 0.8, 0.7], [0.2, 0, 0.1j, 0]
    )
    dist = threshold_distribution(state, backend=backend)
    assert math.fsum(dist.values()) == pytest.approx(1.0, abs=1e-13)
    vacuum = GaussianState(np.zeros(8), np.eye(8), state.nodes)

    def p_vacuum(modes):
        if not modes:
            return 1.0
        return state.reduced(modes).overlap(vacuum.reduced(modes))

    for clicks, p in dist.items():
        on = [i for i, c in enumerate(clicks) if c]
        off = [i for i, c in enumerate(clicks) if not c]
        expected = math.fsum(
            (-1) ** len(y) * p_vacuum(tuple(sorted(off + list(y))))
            for k in range(len(on) + 1)
            for y in itertools.combinations(on, k)
        )
        assert p == pytest.approx(expected, abs=1e-13)


def test_loss_is_binomial_thinning(backend):
    unitary = haar_unitary(2, rng=12)
    lossless = squeezed_interferometer_state([0.5, 0.4], unitary, displacement=[0.3, -0.2j])
    eta = [0.7, 0.45]
    lossy = apply_loss(lossless, eta)
    k = 16
    ideal = np.zeros((k, k))
    for n in itertools.product(range(k), repeat=2):
        if sum(n) < k:
            ideal[n] = pnr_probability(lossless, n, backend=backend)
    tail = 1 - ideal.sum()
    thinned = _thinned(ideal, eta)
    for n in itertools.product(range(4), repeat=2):
        assert pnr_probability(lossy, n, backend=backend) == pytest.approx(
            thinned[n], abs=tail + 1e-14
        )
    with pytest.raises(ValueError):
        apply_loss(lossless, 1.2)


# --------------------------------------------------------------------------- samplers


def test_pnr_sampler_reports_truncated_mass():
    state = squeezed_interferometer_state([0.9, 0.7], haar_unitary(2, rng=2), [0.9, 0.9])
    result = sample_pnr(state, 20000, cutoff=6, rng=4)
    dist = pnr_distribution(state, 6)
    assert result.truncated_mass == pytest.approx(dist.truncated_mass)
    assert result.truncated_mass > 0.01  # this configuration has visible truncated mass
    overflow = result.overflow
    assert stats.binomtest(overflow, 20000, dist.truncated_mass).pvalue > 1e-3
    counts = Counter(map(tuple, result.samples[result.samples[:, 0] >= 0]))
    keys = [n for n, p in dist.probabilities.items() if p * 20000 >= 5]
    observed = [counts.get(n, 0) for n in keys]
    expected = [dist.probabilities[n] * 20000 for n in keys]
    rest_obs = 20000 - sum(observed)
    rest_exp = 20000 - sum(expected)
    assert stats.chisquare(observed + [rest_obs], expected + [rest_exp]).pvalue > 1e-3


def test_threshold_sampler_matches_distribution():
    state = squeezed_interferometer_state([0.8, 0.5, 0.6], haar_unitary(3, rng=9), 0.8)
    dist = threshold_distribution(state)
    samples = sample_threshold(state, 20000, rng=1)
    counts = Counter(map(tuple, samples))
    observed = [counts.get(c, 0) for c in dist]
    expected = [p * 20000 for p in dist.values()]
    assert stats.chisquare(observed, expected).pvalue > 1e-3


def test_auto_method_uses_the_exact_mixed_formula_for_nearly_pure_states():
    # Review finding: an absolute 1e-10 purity test routed a state with loss 1 - 1e-11 to
    # the pure formula, silently dropping the off-diagonal block of A.
    from photographiq.sampling.gbs import gbs_matrices, pnr_probability
    from photographiq.sampling.gbs import squeezed_interferometer_state as build
    from photographiq.sampling.unitaries import haar_unitary

    unitary = haar_unitary(2, 4)
    nearly_pure = build([0.6, 0.3], unitary, transmissivity=1 - 1e-11)
    assert not gbs_matrices(nearly_pure).pure
    for pattern in [(2, 1), (1, 1), (3, 0)]:
        auto = pnr_probability(nearly_pure, pattern, backend="python")
        mixed = pnr_probability(nearly_pure, pattern, method="mixed", backend="python")
        assert auto == mixed
    pure = build([0.6, 0.3], unitary)
    assert gbs_matrices(pure).pure


# ----------------------------------------------------- threshold rounding diagnostics


def test_threshold_error_estimate_bounds_multiprecision_reference(backend):
    """16 modes, 8 clicks, r = 0.2: P ~ 1e-9 from an alternating sum of O(1) terms.

    The 30-digit evaluation is the reference; the double-precision result must lie within
    its own error estimate, and the estimate (2.0e-5) must trigger the warning. The upper
    limit only caps how pessimistic the bound may be: it also covers dark-mode conditioning
    and eigensolver rounding, which the earlier, tighter estimate (2.6e-6) missed (it was
    exceeded on small-squeezing cases; see the multiprecision bound test below).
    """
    state = squeezed_interferometer_state(np.full(16, 0.2), haar_unitary(16, rng=3), 0.9)
    clicks = [1] * 8 + [0] * 8
    ref, ref_error = threshold_probability(state, clicks, precision=30, return_error=True)
    assert ref == pytest.approx(1.1329737713467e-09, rel=1e-12) and ref_error < 1e-15
    with pytest.warns(ThresholdPrecisionWarning, match="precision="):
        value, error = threshold_probability(state, clicks, backend=backend, return_error=True)
    assert 1e-6 < error < 3e-5
    assert abs(value - ref) <= error * ref


def test_threshold_multiprecision_matches_closed_forms():
    r = 0.8
    bs = np.array([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2)
    tmsv = squeezed_interferometer_state([r, -r], bs)
    p00 = 1 / math.cosh(r) ** 2
    assert threshold_probability(tmsv, [0, 0], precision=30) == pytest.approx(p00, rel=1e-15)
    assert threshold_probability(tmsv, [1, 1], precision=30) == pytest.approx(1 - p00, rel=1e-15)
    alpha = np.array([0.6 - 0.3j, 0.2j])
    mean = np.ravel(np.column_stack([2 * alpha.real, 2 * alpha.imag]))
    coherent = GaussianState(mean, np.eye(4), (0, 1))
    expected = math.prod(1 - math.exp(-(abs(a) ** 2)) for a in alpha)
    assert threshold_probability(coherent, [1, 1], precision=30) == pytest.approx(
        expected, rel=1e-15
    )
    with pytest.raises(ValueError, match="digits"):
        threshold_probability(coherent, [1, 1], precision=8)


def _auto(*args, **kwargs):
    from photographiq.sampling.gbs import ThresholdInfo

    return cast(
        tuple[float, ThresholdInfo],
        threshold_probability(*args, precision="auto", return_info=True, **kwargs),
    )


def test_threshold_auto_precision_stays_in_double_when_accurate(backend):
    state = squeezed_interferometer_state([0.5] * 6, haar_unitary(6, rng=1), 0.8)
    clicks = [1, 0, 1, 1, 0, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("error", ThresholdPrecisionWarning)
        value, info = _auto(state, clicks, backend=backend)
    double, error = _with_error(state, clicks, backend=backend)
    assert (info.path, info.digits) == ("double", None)
    assert value == double and info.error == info.double_error == error < 1e-6
    # An exact zero (B = 0, vacuum mode clicking alone) needs no mpmath either.
    vac = squeezed_interferometer_state([0.5, 0.0], np.eye(2))
    with pytest.warns(ThresholdPrecisionWarning, match="consistent with zero"):
        value, info = _auto(vac, [0, 1], backend=backend)
    assert value == 0.0 and info.path == "double"


def test_threshold_auto_precision_switches_to_mpmath(backend):
    """16 modes, 8 clicks, r = 0.2: the double bound (2e-5) exceeds rtol, so auto recomputes."""
    state = squeezed_interferometer_state(np.full(16, 0.2), haar_unitary(16, rng=3), 0.9)
    clicks = [1] * 8 + [0] * 8
    ref = cast(float, threshold_probability(state, clicks, precision=30))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ThresholdPrecisionWarning)
        value, info = _auto(state, clicks, backend=backend)
    assert info.path == "mpmath" and info.digits is not None and info.digits >= 20
    assert info.double_error is not None and info.double_error > 1e-6 > info.error
    assert abs(value - ref) <= 1e-6 * ref
    # A looser tolerance keeps the double result (and warns only above rtol).
    with warnings.catch_warnings():
        warnings.simplefilter("error", ThresholdPrecisionWarning)
        value, info = _auto(state, clicks, backend=backend, rtol=1e-4)
    assert info.path == "double" and abs(value - ref) <= info.error * ref
    with pytest.raises(ValueError, match="rtol"):
        _auto(state, clicks, rtol=0.0)
    with pytest.raises(ValueError, match="not both"):
        threshold_probability(state, clicks, return_error=True, return_info=True)


def test_threshold_precision_without_mpmath_names_the_extra(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "mpmath", None)  # import mpmath raises ImportError
    state = squeezed_interferometer_state([0.5] * 2, _BS)
    for precision in ("auto", 30):
        with pytest.raises(ImportError, match=r"photographiq\[precision\]"):
            threshold_probability(state, [1, 1], precision=precision)
    assert threshold_probability(state, [1, 1]) > 0  # double precision needs no mpmath


# ------------------------------------------- threshold rounding bound, sampler clipping


def _kernels_for(backend):
    from photographiq.sampling.linalg import _kernels

    return _kernels(backend)


def _with_error(*args, **kwargs) -> tuple[float, float]:
    return cast(tuple[float, float], threshold_probability(*args, return_error=True, **kwargs))


_BS = np.array([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2)
_U5 = np.eye(5, dtype=complex)
_U5[np.ix_([0, 1, 3, 4], [0, 1, 3, 4])] = haar_unitary(4, rng=7)
_BOUND_CASES = {
    # Squeezing 0.05: the earlier estimate eps sum|w_T| (5.2e-17) was below the native error
    # (4.2e-16) because eigenvalue contributions cancel inside each exponent.
    "small-squeezing": (
        squeezed_interferometer_state([0.05] * 10, haar_unitary(10, rng=4), 0.95),
        [1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    ),
    "displaced": (
        squeezed_interferometer_state(
            [0.5] * 6, haar_unitary(6, rng=1), 0.8, list(0.3 * np.exp(2j * np.arange(6)))
        ),
        [1, 0, 1, 1, 0, 0],
    ),
    "displaced-8": (
        squeezed_interferometer_state(
            [0.3] * 8, haar_unitary(8, rng=9), 0.9, list(0.3 * np.exp(1j * np.arange(8)))
        ),
        [1, 1, 0, 1, 0, 1, 0, 0],
    ),
    "large-squeezing": (
        squeezed_interferometer_state([1.2] * 6, haar_unitary(6, rng=10), 0.7),
        [1, 0, 0, 1, 1, 1],
    ),
    "lossless-displaced": (
        squeezed_interferometer_state(
            [0.6] * 8, haar_unitary(8, rng=11), 1.0, [0.2, 0, 0.1, 0, 0, 0.3j, 0, 0]
        ),
        [0, 1, 1, 0, 1, 0, 1, 1],
    ),
    "no-dark-modes": (
        apply_loss(
            squeezed_interferometer_state(
                [0.4] * 4, haar_unitary(4, rng=12), 1.0, [0.5, 0.1, -0.2j, 0]
            ),
            0.6,
        ),
        [1, 1, 1, 1],
    ),
    # Zero-probability patterns. In the TMSV the error (1.6e-16, from conditioning) is 6x the
    # exact value for the stored covariance; the earlier estimate claimed 2e-16 relative.
    "zero-tmsv": (squeezed_interferometer_state([0.8, -0.8], _BS), [1, 0]),
    "zero-product": (
        apply_loss(squeezed_interferometer_state([0.5, 0.0], np.eye(2)), 0.9),
        [1, 1],
    ),
    "zero-vacuum-mode": (
        squeezed_interferometer_state([0.4, 0.3, 0.0, 0.5, 0.6], _U5, 0.85, [0.2, 0, 0, 0.1j, 0]),
        [1, 0, 1, 1, 0],
    ),
    # Few clicks at tiny squeezing: without the factor max(1, 16 / sqrt(2^k - 1)) the error
    # was 1.6x (python, k = 1), 1.4x (python, k = 2) and 2.8x (native, k = 3) the bound.
    "few-clicks-1": (squeezed_interferometer_state([0.006], haar_unitary(1, rng=0), 0.99), [1]),
    "few-clicks-2": (
        squeezed_interferometer_state([0.0018, 0.0042], haar_unitary(2, rng=11)),
        [1, 1],
    ),
    "few-clicks-3": (
        squeezed_interferometer_state([0.0006, 0.001, 0.0014], haar_unitary(3, rng=6)),
        [1, 1, 1],
    ),
}


@pytest.mark.parametrize("name", list(_BOUND_CASES))
def test_threshold_rounding_bound_holds_against_multiprecision(backend, name):
    """|double - 30-digit reference| <= B (absolute bound incl. dark-mode conditioning)."""
    from photographiq.sampling.gbs import _threshold_double

    state, clicks = _BOUND_CASES[name]
    g = gbs_matrices(state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ThresholdPrecisionWarning)
        ref = cast(float, threshold_probability(g, clicks, precision=30))
        value, error = _with_error(g, clicks, backend=backend)
    same, bound, _ = _threshold_double(g, np.asarray(clicks), _kernels_for(backend))
    assert same == value
    assert error == (bound / abs(value) if value else math.inf)  # relative = B / |value|
    assert abs(value - ref) <= bound


def test_threshold_flags_negative_and_zero_results(backend):
    product = apply_loss(squeezed_interferometer_state([0.5, 0.0], np.eye(2)), 0.9)
    # Mode 1 has V = 0.9 + 0.1 rounded below 1: the exact result for the stored covariance
    # is -4.9e-17, far beyond rounding, so the warning says the input is not physical.
    with pytest.warns(ThresholdPrecisionWarning, match="negative beyond"):
        value = cast(float, threshold_probability(product, [0, 1], backend=backend))
    assert value < 0
    with pytest.warns(ThresholdPrecisionWarning, match="consistent with zero within"):
        value, error = _with_error(product, [1, 1], backend=backend)
    assert value == 0.0 and error == math.inf
    # A negative result inside the bound (-2.5e-17 vs 1.3e-16) is flagged too; the earlier
    # estimate (relative 1e-15) let it pass silently.
    tmsv = squeezed_interferometer_state([0.3, -0.3], _BS)
    with pytest.warns(ThresholdPrecisionWarning, match="consistent with zero within"):
        value, error = _with_error(tmsv, [1, 0], backend=backend)
    assert value < 0 and error > 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", ThresholdPrecisionWarning)
        threshold_probability(tmsv, [1, 1], backend=backend)  # ordinary results stay quiet


def _vacuum_sum_value(excess, shift):
    from photographiq.sampling.gbs import _vacuum_sum

    return _vacuum_sum(excess, shift)[0]


def test_conditional_click_sensitivity_bounds():
    """|dF| <= (1+sqrt3)/8 ||V_c|| Tr|dV_c| and <= ||V_c||^{1/2} ||de|| / 2 (module doc).

    F is the all-click probability of a physical Gaussian state (V_c = I + 2E, mean e);
    derivatives by central differences in random directions.
    """
    rng = np.random.default_rng(5)
    h = 1e-6
    for trial in range(12):
        k = 1 + trial % 3
        state = squeezed_interferometer_state(
            rng.uniform(0.1, 0.8, k).tolist(),
            haar_unitary(k, rng=trial),
            rng.uniform(0.5, 1.0, k).tolist(),
            (0.4 * (rng.normal(size=k) + 1j * rng.normal(size=k))).tolist(),
        )
        excess, mean = (state.covariance - np.eye(2 * k)) / 2, state.mean
        vc = np.linalg.eigvalsh(state.covariance)[-1]
        n = rng.normal(size=(2 * k, 2 * k))
        n = (n + n.T) / 2
        up = _vacuum_sum_value(excess + h * n / 2, mean)
        df = (up - _vacuum_sum_value(excess - h * n / 2, mean)) / (2 * h)
        assert abs(df) <= (1 + math.sqrt(3)) / 8 * vc * np.abs(np.linalg.eigvalsh(n)).sum()
        de = rng.normal(size=2 * k)
        up = _vacuum_sum_value(excess, mean + h * de)
        df = (up - _vacuum_sum_value(excess, mean - h * de)) / (2 * h)
        assert abs(df) <= 0.5 * math.sqrt(vc) * np.linalg.norm(de)
    # Not vacuous: at the vacuum, dF/dV_qq = 1/4 against the bound (1 + sqrt3)/8 = 0.34.
    up = _vacuum_sum_value(np.diag([h / 2, 0.0]), np.zeros(2))
    df = (up - _vacuum_sum_value(np.diag([-h / 2, 0.0]), np.zeros(2))) / (2 * h)
    assert df == pytest.approx(0.25, rel=1e-6)


def test_python_vacuum_sum_raises_like_native_for_non_positive_blocks(backend):
    from photographiq.sampling.gbs import _vacuum_sum

    excess, shift = np.diag([-1.5, 0.1]), np.zeros(2)  # I + E has eigenvalue -0.5
    kernels = _kernels_for(backend)
    evaluate = _vacuum_sum if kernels is None else kernels.threshold_vacuum_sum
    with pytest.raises(ValueError, match="^Conditioned covariance block is not positive definite$"):
        evaluate(excess, shift)


def test_threshold_precision_validates_backend():
    state = squeezed_interferometer_state([0.3, -0.3], _BS)
    with pytest.raises(ValueError, match="Unknown backend"):
        threshold_probability(state, [1, 1], precision=20, backend="bogus")  # type: ignore[arg-type]


def test_pnr_sampler_clips_rounding_level_negatives():
    # Nearly pure TMSV: the exact P(2, 5) is 0, the mixed-formula hafnian gives -2.3e-18.
    state = squeezed_interferometer_state([0.8, -0.8], _BS, 1 - 1e-13)
    dist = pnr_distribution(state, 8)
    assert min(dist.probabilities.values()) < 0
    result = sample_pnr(state, 2000, cutoff=8, rng=3)
    negative = {n for n, p in dist.probabilities.items() if p < 0}
    assert not negative & set(map(tuple, result.samples))


def test_samplers_raise_for_negatives_beyond_rounding(monkeypatch):
    from photographiq.sampling import gbs

    weights = gbs._sampling_weights(np.array([0.7, 0.3, -1e-17]))
    assert weights[2] == 0.0 and weights.sum() == pytest.approx(1.0, abs=1e-15)
    with pytest.raises(ValueError, match="negative beyond rounding"):
        gbs._sampling_weights(np.array([0.7, 0.3, -1e-9]))
    state = squeezed_interferometer_state([0.3, -0.3], _BS)
    bad = {(0, 0): 0.9, (0, 1): -1e-6, (1, 0): 0.0, (1, 1): 0.1}
    monkeypatch.setattr(gbs, "threshold_distribution", lambda *a, **k: bad)
    with pytest.raises(ValueError, match="negative beyond rounding"):
        sample_threshold(state, 10, rng=0)
