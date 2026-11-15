"""Independent validation of the Fock engines (native-fock, native-mixed-fock; Piquasso second).

References are built in the test from first principles (NumPy/SciPy/mpmath only):

* Gates: ``scipy.linalg.expm`` of the generator (hbar = 2, q = a + a^dag) on an ENLARGED
  product space with D levels PER MODE (D >= 26 here, against cutoff 10), applied to the
  input and then projected onto the total-photon subspace n_1 + ... + n_m < cutoff. A
  single gate on a state supported inside that subspace is P U psi; the engines apply
  P U P and renormalise, so the reference state is P U psi / ||P U psi|| and the reported
  retained norm must equal ||P U psi||^2 (truncation effect accounted for exactly).
  Convergence of the reference in D is asserted (D = 40 vs 56 for single-mode gates,
  D = 26 vs 32 for two-mode gates; projected results differ by < 1e-14).
* Cubic phase: exact projected elements <m|exp(i gamma q^3/6)|n> by spectrally convergent
  trapezoidal integration of Hermite functions (two grids agree to the 2e-12 rounding floor); the default
  ``truncated_generator`` definition is checked against expm of the cutoff x cutoff
  truncated q^3 and shown to converge to the exact gate with the cutoff.
* States: closed-form amplitudes evaluated with mpmath (30 digits).
* Measurements, partial traces, loss: direct projection / explicit tensor contraction /
  beam-splitter dilation with a thermal environment mode.

Tolerances: single-gate comparisons are bounded by float64 rounding of expm and of the
engines' Bargmann recurrences (~1e-14 for these sizes); 1e-12 leaves > 30x headroom over
the measured maxima in docs/validation/independent-validation-matrix.md. Piquasso is a
secondary check: its CZ (GaussianTransform) is an Euler decomposition truncated after every
factor, so it differs from P U P at truncation level (2e-5 at cutoff 10; tolerance 1e-4).
"""

from __future__ import annotations

import itertools
import os
import warnings
from functools import lru_cache
from math import comb

import mpmath as mp
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.special import eval_hermite, gammaln

import photographiq as pg
from photographiq import registry

NATIVE = ["native-fock", "native-mixed-fock"]
PIQUASSO = ["piquasso-fock", "piquasso-mixed-fock"]
CUTOFF = 10
MEASURED: dict = {}  # (operation, backend) -> (max error, tolerance, test); see the matrix doc


def _record(operation, backend, error, tolerance):
    key = (operation, backend)
    test = os.environ.get("PYTEST_CURRENT_TEST", "").split("[")[0].split(" ")[0]
    previous = MEASURED.get(key, (0.0, tolerance, test))
    MEASURED[key] = (max(previous[0], float(error)), tolerance, previous[2])
    assert error <= tolerance, f"{operation} on {backend}: error {error:.3e} > {tolerance:.1e}"


def _backend(name, cutoff=CUTOFF, **kwargs):
    if name.startswith("piquasso"):
        pytest.importorskip("piquasso")
    reason = registry.BACKENDS[name].unavailable_reason()
    if reason is not None:
        pytest.skip(reason)
    cls = registry.BACKENDS[name].load()
    engine = cls(cutoff, **kwargs)
    engine.reset(3)
    return engine


def _mixed(name):
    return "mixed" in name


# ----- dense first-principles operators --------------------------------------------------
def ladders(modes, dim):
    a = np.diag(np.sqrt(np.arange(1, dim)), 1).astype(complex)
    out = []
    for k in range(modes):
        ops = [np.eye(dim)] * modes
        ops[k] = a
        full = ops[0]
        for op in ops[1:]:
            full = np.kron(full, op)
        out.append(full)
    return out


def index(occupation, dim):
    return int(sum(n * dim ** (len(occupation) - 1 - k) for k, n in enumerate(occupation)))


def total_basis(modes, cutoff):
    return {b for b in itertools.product(range(cutoff), repeat=modes) if sum(b) < cutoff}


def dense_vector(amplitudes, modes, dim):
    v = np.zeros(dim**modes, complex)
    for occ, amp in amplitudes.items():
        v[index(occ, dim)] = amp
    return v


def project(vector_or_rho, basis, dim):
    idx = [index(b, dim) for b in basis]
    x = np.asarray(vector_or_rho)
    return x[idx] if x.ndim == 1 else x[np.ix_(idx, idx)]


@lru_cache(maxsize=64)
def unitary(gate, value, dim):
    """expm of the gate generator with ``dim`` levels per mode, two modes in order (0, 1).

    Single-mode generators are exponentiated on one mode and tensored with the identity.
    """
    a = np.diag(np.sqrt(np.arange(1, dim)), 1).astype(complex)
    ad, one = a.conj().T, np.eye(dim)
    n, q = ad @ a, a + ad
    if gate == "displace":
        alpha = complex(value[0], value[1]) / 2
        single = (1, alpha * ad - np.conj(alpha) * a)
    elif gate == "rotate":
        single = (1, 1j * value * n)
    elif gate == "squeeze":
        single = (0, value * (a @ a - ad @ ad) / 2)
    elif gate == "kerr":
        single = (0, 1j * value * n @ n)
    elif gate == "quadratic_phase":
        single = (1, 1j * value * q @ q / 4)
    else:
        single = None
    if single is not None:
        mode, g = single
        u = expm(g)
        return np.kron(u, one) if mode == 0 else np.kron(one, u)
    au, av = np.kron(a, one), np.kron(one, a)
    if gate == "beamsplitter":  # q_u' = c q_u - s q_v
        g = value * (av.conj().T @ au - au.conj().T @ av)
    elif gate == "entangle":  # exp(i g q_u q_v / 2)
        g = 1j * value * (au + au.conj().T) @ (av + av.conj().T) / 2
    elif gate == "cross_kerr":
        g = 1j * value * (au.conj().T @ au) @ (av.conj().T @ av)
    else:
        raise KeyError(gate)
    return expm(g)


# Levels per mode of the enlarged reference, and the larger size used to prove convergence.
DIMS = {"single": (40, 56), "two": (26, 32)}
SINGLE_MODE = {"displace", "rotate", "squeeze", "kerr", "quadratic_phase"}


CALLS = {
    "displace": (lambda e: e.displace(1, 0.5, -0.3), (0.5, -0.3)),
    "rotate": (lambda e: e.rotate(1, 0.8), 0.8),
    "squeeze": (lambda e: e.squeeze(0, 0.3), 0.3),
    "beamsplitter": (lambda e: e.beamsplitter(0, 1, 0.55), 0.55),
    "entangle": (lambda e: e.entangle(0, 1, 0.4), 0.4),
    "kerr": (lambda e: e.kerr(0, 0.35), 0.35),
    "cross_kerr": (lambda e: e.cross_kerr(0, 1, 0.6), 0.6),
    "quadratic_phase": (lambda e: e.quadratic_phase(1, 0.3), 0.3),
}
NUMBER_CONSERVING = {"rotate", "beamsplitter", "kerr", "cross_kerr"}


def random_superposition(rng, modes, max_total=3):
    support = [b for b in itertools.product(range(max_total + 1), repeat=modes)]
    support = [b for b in support if sum(b) <= max_total]
    amps = rng.normal(size=len(support)) + 1j * rng.normal(size=len(support))
    amps /= np.linalg.norm(amps)
    return dict(zip(support, amps, strict=True))


def _prepare(engine, name, amplitudes, mixed_partner=None):
    """Pure input for pure engines; rank-2 mixture (amplitudes, partner) for mixed ones."""
    modes = len(next(iter(amplitudes)))
    if not _mixed(name):
        engine.prepare_resource(tuple(range(modes)), pg.FockSuperposition.from_mapping(amplitudes))
        return [(1.0, amplitudes)]
    basis = sorted(set(amplitudes) | set(mixed_partner))
    v1 = np.array([amplitudes.get(b, 0) for b in basis])
    v2 = np.array([mixed_partner.get(b, 0) for b in basis])
    rho = 0.7 * np.outer(v1, v1.conj()) + 0.3 * np.outer(v2, v2.conj())
    engine.prepare_resource(tuple(range(modes)), pg.FockDensityMatrix(rho, tuple(basis)))
    return [(0.7, amplitudes), (0.3, mixed_partner)]


def _reference(mixture, gate, value, modes, dim, basis):
    u = unitary(gate, value, dim)
    rho = np.zeros((len(basis), len(basis)), complex)
    for w, amps in mixture:
        out = project(u @ dense_vector(amps, modes, dim), basis, dim)
        rho += w * np.outer(out, out.conj())
    return rho


def _state_matrix(state, name):
    if _mixed(name):
        return state.density_matrix
    v = state.state_vector
    return np.outer(v, v.conj())


# ----- truncation convention ----------------------------------------------------------------
@pytest.mark.parametrize("name", NATIVE)
def test_truncation_is_by_total_photon_number(name):
    for modes, cutoff in [(1, 7), (2, 6), (3, 5)]:
        engine = _backend(name, cutoff)
        for k in range(modes):
            engine.prepare(k, state=pg.FockInput.number(0))
        basis = engine.get_state().basis
        assert set(map(tuple, basis)) == total_basis(modes, cutoff)
        assert len(basis) == comb(modes + cutoff - 1, modes)


# ----- states --------------------------------------------------------------------------------
def _coherent_mp(alpha, cutoff):
    mp.mp.dps = 30
    a = mp.mpc(alpha.real, alpha.imag)
    return [mp.exp(-(abs(a) ** 2) / 2) * a**n / mp.sqrt(mp.factorial(n)) for n in range(cutoff)]


def _squeezed_mp(r, theta, cutoff):
    mp.mp.dps = 30
    z = -mp.exp(2j * mp.mpf(theta)) * mp.tanh(r)
    out = [mp.mpc(0)] * cutoff
    for k in range((cutoff + 1) // 2):
        if 2 * k < cutoff:
            fact = mp.sqrt(mp.factorial(2 * k)) / (2**k * mp.factorial(k))
            out[2 * k] = z**k * fact / mp.sqrt(mp.cosh(r))
    return out


@pytest.mark.parametrize("name", NATIVE + PIQUASSO)
def test_number_coherent_squeezed_states_analytic(name):
    cutoff = 14
    tolerance = 1e-13 if name in NATIVE else 1e-10
    alpha, r, theta = 1.1 - 0.7j, 0.6, 0.35
    cases = {
        "number": (pg.FockInput.number(5), [0] * 5 + [1] + [0] * (cutoff - 6)),
        "coherent": (pg.GaussianInput.coherent(alpha), _coherent_mp(alpha, cutoff)),
        "squeezed": (pg.GaussianInput.squeezed(r, theta), _squeezed_mp(r, theta, cutoff)),
        # Default resource prepare(node, r): p-squeezed vacuum = R(pi/2) S(r)|0>.
        "resource p-squeezed": (r, _squeezed_mp(r, np.pi / 2, cutoff)),
    }
    for label, (prep, exact) in cases.items():
        exact = np.array([complex(x) for x in exact])
        engine = _backend(name, cutoff)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if isinstance(prep, float):
                engine.prepare(0, prep)
            else:
                engine.prepare(0, state=prep)
        state = engine.get_state()
        retained = float(np.vdot(exact, exact).real)  # analytic omitted weight
        exact /= np.sqrt(retained)
        ordered = np.array([exact[b[0]] for b in state.basis])
        rho = _state_matrix(state, name)
        error = np.abs(rho - np.outer(ordered, ordered.conj())).max()
        _record(f"state:{label}", name, error, tolerance)
        first = engine.retained_norms[0]  # the single-mode preparation (before tensoring)
        _record(f"retained:{label}", name, abs(first - retained), tolerance)
        if label == "coherent":  # omitted weight = regularised upper incomplete gamma
            q = float(mp.gammainc(cutoff, abs(alpha) ** 2, regularized=True))
            _record("retained:coherent = Q(c, |a|^2)", name, abs(first - q), tolerance)


@pytest.mark.parametrize("name", NATIVE)
def test_displaced_squeezed_and_mixed_gaussian_inputs(name):
    cutoff, big = 16, 90
    r, theta, beta, nth = 0.45, 0.8, 0.6 + 0.3j, 0.3
    a = ladders(1, big)[0]
    u = (
        expm(beta * a.conj().T - np.conj(beta) * a)
        @ np.diag(np.exp(1j * theta * np.arange(big)))
        @ expm(r * (a @ a - a.T @ a.T) / 2)
    )
    weights = nth ** np.arange(60) / (1 + nth) ** np.arange(1, 61) if _mixed(name) else [1.0]
    rho = sum(w * np.outer(u[:, k], u[:, k].conj()) for k, w in enumerate(weights))
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    cov = (2 * nth + 1 if _mixed(name) else 1) * rot @ np.diag(np.exp([-2 * r, 2 * r])) @ rot.T
    prep = pg.GaussianInput((2 * beta.real, 2 * beta.imag), tuple(map(tuple, cov)))
    engine = _backend(name, cutoff)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        engine.prepare(0, state=prep)
    state = engine.get_state()
    expected = rho[:cutoff, :cutoff]
    retained = np.trace(expected).real
    order = [b[0] for b in state.basis]
    expected = expected[np.ix_(order, order)] / retained
    label = "state:thermal displaced squeezed" if _mixed(name) else "state:displaced squeezed"
    _record(label, name, np.abs(_state_matrix(state, name) - expected).max(), 1e-12)
    reported = engine.retained_norms[0]
    _record(label.replace("state", "retained"), name, abs(reported - retained), 1e-12)


# ----- gates against expm on the enlarged space ------------------------------------------------
GATE_TOLERANCE = {"native": 1e-12, "piquasso": 1e-10}
PIQUASSO_TRUNCATED = {"entangle": 1e-4}  # Euler factors truncated (measured 2.0e-5)


@pytest.mark.parametrize("gate", list(CALLS))
@pytest.mark.parametrize("name", NATIVE + PIQUASSO)
def test_gate_against_enlarged_expm(name, gate):
    if name in PIQUASSO and gate == "cross_kerr":
        pytest.skip("Piquasso Fock backends have no cross-Kerr method")
    rng = np.random.default_rng(list(CALLS).index(gate))
    psi, partner = random_superposition(rng, 2), random_superposition(rng, 2)
    engine = _backend(name)
    mixture = _prepare(engine, name, psi, mixed_partner=partner)
    call, value = CALLS[gate]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        call(engine)
    state = engine.get_state()
    basis = [tuple(int(n) for n in b) for b in state.basis]
    dim, dim_check = DIMS["single" if gate in SINGLE_MODE else "two"]
    ref = _reference(mixture, gate, value, 2, dim, basis)
    ref_check = _reference(mixture, gate, value, 2, dim_check, basis)
    _record(
        f"reference convergence D={dim} vs {dim_check}:{gate}",
        "expm",
        np.abs(ref - ref_check).max(),
        1e-14,
    )
    retained = np.trace(ref).real
    ref /= retained
    kind = "native" if name in NATIVE else "piquasso"
    tolerance = GATE_TOLERANCE[kind]
    if kind == "piquasso":
        tolerance = PIQUASSO_TRUNCATED.get(gate, tolerance)
    rho = _state_matrix(state, name)
    _record(f"gate:{gate}", name, np.abs(rho - ref).max(), tolerance)
    reported = engine.diagnostics[-1]["retained_norm"]
    _record(f"retained:{gate}", name, abs(reported - retained), tolerance)
    if gate in NUMBER_CONSERVING:
        assert abs(retained - 1) < 1e-14 and abs(reported - 1) < 1e-13
    else:
        assert retained < 1 - 1e-12  # the truncated gate is a strict contraction here
    trace = np.trace(rho).real
    hermiticity = np.abs(rho - rho.conj().T).max()
    min_eig = np.linalg.eigvalsh((rho + rho.conj().T) / 2).min()
    _record(
        "physicality:trace/Hermitian/PSD", name, max(abs(trace - 1), hermiticity, -min_eig), 1e-13
    )


@pytest.mark.parametrize("name", NATIVE)
def test_truncated_gate_converges_to_untruncated_gate(name):
    """Displacement + CZ: the cutoff-c state approaches the untruncated U psi as c grows."""
    rng = np.random.default_rng(0)
    psi = random_superposition(rng, 2, max_total=2)
    big = DIMS["two"][1]
    exact = unitary("displace", (0.5, -0.3), big) @ dense_vector(psi, 2, big)
    exact = unitary("entangle", 0.4, big) @ exact
    errors = []
    for cutoff in (8, 12, 16, 20):
        engine = _backend(name, cutoff)
        _prepare(engine, name, psi, mixed_partner=psi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            engine.displace(1, 0.5, -0.3)
            engine.entangle(0, 1, 0.4)
        state = engine.get_state()
        basis = [tuple(int(n) for n in b) for b in state.basis]
        ref = project(exact, basis, big)
        fid = np.vdot(ref, _state_matrix(state, name) @ ref).real
        errors.append(1 - fid)
    assert all(b < a for a, b in zip(errors, errors[1:], strict=False))
    _record("convergence: 1-F at cutoff 20 (D+CZ)", name, errors[-1], 1e-9)


def test_number_conserving_gates_preserve_inner_products():
    """Pure engine: truncated rotation/BS/Kerr/cross-Kerr are exactly unitary per sector."""
    name = "native-fock"
    rng = np.random.default_rng(12)
    states = [random_superposition(rng, 2, max_total=9) for _ in range(2)]
    outputs = []
    for amps in states:
        engine = _backend(name)
        with warnings.catch_warnings():  # support reaches the cutoff on purpose
            warnings.simplefilter("ignore", RuntimeWarning)
            _prepare(engine, name, amps)
            engine.rotate(0, 0.4)
            engine.beamsplitter(0, 1, 1.1)
            engine.kerr(1, 0.9)
            engine.cross_kerr(0, 1, -0.7)
        assert all(abs(n - 1) < 1e-13 for n in engine.retained_norms[1:])
        outputs.append(engine.get_state().state_vector)
    basis = [tuple(b) for b in engine.get_state().basis]
    inputs = [np.array([amps.get(b, 0) for b in basis]) for amps in states]
    error = abs(np.vdot(*outputs) - np.vdot(*inputs))
    _record("unitarity: <phi1|phi2> = <psi1|psi2>", name, error, 1e-13)


# ----- cubic phase --------------------------------------------------------------------------
def hermite_functions(x, cutoff):
    """psi_n(x) for hbar = 2: (2 pi)^-1/4 (2^n n!)^-1/2 H_n(x / sqrt 2) exp(-x^2/4)."""
    n = np.arange(cutoff)[:, None]
    log_norm = -0.25 * np.log(2 * np.pi) - 0.5 * (n * np.log(2) + gammaln(n + 1))
    return np.exp(log_norm) * eval_hermite(n, x[None, :] / np.sqrt(2)) * np.exp(-(x**2) / 4)


def cubic_projected(gamma, cutoff, points):
    x = np.linspace(-22, 22, points)
    psi = hermite_functions(x, cutoff)
    weights = np.full(points, x[1] - x[0])
    return (psi * weights * np.exp(1j * gamma * x**3 / 6)) @ psi.T


def test_cubic_reference_quadrature_converges():
    a, b = cubic_projected(0.4, 24, 6001), cubic_projected(0.4, 24, 12001)
    assert np.abs(a - b).max() < 1e-11  # rounding floor of H_n(x) e^{-x^2/4} ~ 2e-12
    gram = cubic_projected(0.0, 24, 6001)
    np.testing.assert_allclose(gram, np.eye(24), atol=1e-13)  # Hermite functions orthonormal


@pytest.mark.parametrize("name", NATIVE)
def test_cubic_phase_both_definitions(name):
    gamma, cutoff = 0.2, 12
    rng = np.random.default_rng(4)
    psi = random_superposition(rng, 1)
    vector = np.array([psi.get((n,), 0) for n in range(cutoff)])
    exact_proj = cubic_projected(gamma, cutoff, 6001) @ vector  # P V psi
    q = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    q = q + q.T
    truncated = expm(1j * gamma * q @ q @ q / 6) @ vector
    for method, ref, label in (
        ("projected", exact_proj, "cubic (projected = P V P)"),
        ("truncated_generator", truncated, "cubic (expm of truncated q^3)"),
    ):
        engine = _backend(name, cutoff, cubic_phase=method, norm_tolerance=0.1)
        _prepare(engine, name, psi, mixed_partner=psi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            engine.cubic_phase(0, gamma)
        state = engine.get_state()
        order = [b[0] for b in state.basis]
        retained = np.vdot(ref, ref).real
        expected = ref[order] / np.sqrt(retained)
        error = np.abs(_state_matrix(state, name) - np.outer(expected, expected.conj())).max()
        _record(label, name, error, 1e-11)
        _record(f"retained:{label}", name, abs(engine.retained_norms[-1] - retained), 1e-11)
    # Both definitions converge to the exact gate: infidelity vs P V psi decreases with cutoff.
    errors = {}
    for method in ("projected", "truncated_generator"):
        for cutoff in (12, 20, 28):
            engine = _backend(name, cutoff, cubic_phase=method, norm_tolerance=0.1)
            _prepare(engine, name, psi, mixed_partner=psi)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                engine.cubic_phase(0, gamma)
            exact = cubic_projected(gamma, 40, 6001) @ np.r_[vector, np.zeros(28)]
            state = engine.get_state()
            ref = exact[[b[0] for b in state.basis]]
            fid = np.vdot(ref, _state_matrix(state, name) @ ref).real / np.vdot(exact, exact).real
            errors.setdefault(method, []).append(1 - fid)
    for method, values in errors.items():
        assert values[0] > values[1] > values[2], (method, values)
        _record(f"convergence: cubic {method} 1-F at cutoff 28", name, values[-1], 1e-6)


# ----- measurements and partial traces -------------------------------------------------------
@pytest.mark.parametrize("name", NATIVE)
def test_photon_counting_probability_and_conditional_state(name):
    rng = np.random.default_rng(8)
    psi, partner = random_superposition(rng, 3), random_superposition(rng, 3)
    worst_p = worst_state = 0.0
    for outcome in (0, 1, 2):
        engine = _backend(name, 6)
        mixture = _prepare(engine, name, psi, mixed_partner=partner)
        engine.measure(1, pg.PhotonNumber(), outcome=outcome)
        state = engine.get_state()
        basis = [tuple(b) for b in state.basis]
        rho = np.zeros((len(basis), len(basis)), complex)
        for w, amps in mixture:
            v = np.array([amps.get((b[0], outcome, b[1]), 0) for b in basis])
            rho += w * np.outer(v, v.conj())
        probability = np.trace(rho).real
        worst_p = max(worst_p, abs(engine.last_measurement["value"] - probability))
        error = np.abs(_state_matrix(state, name) - rho / probability).max()
        worst_state = max(worst_state, error)
    _record("photon counting: probability", name, worst_p, 1e-14)
    _record("photon counting: conditional state", name, worst_state, 1e-13)


@pytest.mark.parametrize("name", NATIVE)
def test_partial_trace_by_explicit_contraction(name):
    rng = np.random.default_rng(13)
    psi, partner = random_superposition(rng, 3), random_superposition(rng, 3)
    cutoff = 6
    engine = _backend(name, cutoff)
    mixture = _prepare(engine, name, psi, mixed_partner=partner)
    tensor = np.zeros((cutoff,) * 6, complex)
    for w, amps in mixture:
        v = np.zeros((cutoff,) * 3, complex)
        for occ, amp in amps.items():
            v[occ] = amp
        tensor += w * np.einsum("abc,def->abcdef", v, v.conj())
    worst = 0.0
    for keep, contraction in [
        ((2, 0), "abcdbf->cafd"),
        ((1,), "abcaec->be"),
        ((0, 1), "abcdec->abde"),
    ]:
        reduced = np.einsum(contraction, tensor)
        state = engine.get_state(nodes=keep)
        basis = [tuple(b) for b in state.basis]
        flat = reduced.reshape((cutoff ** len(keep),) * 2)
        idx = [index(b, cutoff) for b in basis]
        worst = max(worst, np.abs(state.density_matrix - flat[np.ix_(idx, idx)]).max())
    _record("partial trace (incl. reordering)", name, worst, 1e-14)


@pytest.mark.parametrize("name", NATIVE + PIQUASSO)
def test_fock_homodyne_density_convention(name):
    """Coherent |alpha>: q cos + p sin ~ N(2 Re(alpha e^{-i theta}), 1 + (1-eta)/eta + noise)."""
    alpha, cutoff = 0.7 + 0.4j, 30
    detectors = [(1.0, 0.0)] + ([(0.6, 0.0), (0.6, 0.2)] if _mixed(name) else [])
    worst = worst_quadrature = 0.0
    for eta, noise in detectors:
        for theta, m in ((0.0, 1.1), (0.9, -0.4), (2.2, 0.3)):
            engine = _backend(name, cutoff)
            engine.prepare(0, state=pg.GaussianInput.coherent(alpha))
            mu = 2 * (alpha * np.exp(-1j * theta)).real
            mean, variance = engine.get_state().quadrature(0, theta)
            worst_quadrature = max(worst_quadrature, abs(mean - mu), abs(variance - 1))
            engine.measure(0, pg.Homodyne(theta, eta, noise), theta, outcome=m)
            var = 1 + (1 - eta) / eta + noise
            exact = np.exp(-((m - mu) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)
            worst = max(worst, abs(engine.last_measurement["value"] / exact - 1))
    _record("homodyne density (Fock)", name, worst, 1e-9)
    _record("quadrature q cos + p sin (Fock)", name, worst_quadrature, 1e-12)


# ----- loss as a beam-splitter dilation ------------------------------------------------------
@pytest.mark.parametrize("nth", [0.0, 0.15])
def test_loss_matches_beam_splitter_dilation(nth):
    name = "native-mixed-fock"
    eta, cutoff, env_levels = 0.65, 8, 16
    rng = np.random.default_rng(21)
    psi, partner = random_superposition(rng, 2), random_superposition(rng, 2)
    engine = _backend(name, cutoff)
    mixture = _prepare(engine, name, psi, mixed_partner=partner)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        engine.loss(0, eta, nth)
    state = engine.get_state()
    basis = [tuple(b) for b in state.basis]
    # Beam splitter (system mode 0, environment) on dim^2; environment in a thermal state.
    dim = 3 + env_levels + 1  # BS conserves n_0 + n_env <= 3 + env_levels
    u = unitary("beamsplitter", float(np.arccos(np.sqrt(eta))), dim)
    u = u.reshape(dim, dim, dim, dim)  # (out sys, out env, in sys, in env)
    env = nth ** np.arange(env_levels) / (1 + nth) ** np.arange(1, env_levels + 1)
    rho = np.zeros((len(basis), len(basis)), complex)
    for w, amps in mixture:
        v = np.zeros((dim, cutoff), complex)  # (sys mode 0, spectator mode 1)
        for (n0, n1), amp in amps.items():
            v[n0, n1] = amp
        for k, pk in enumerate(env if nth else [1.0]):
            out = np.einsum("aeb,bs->aes", u[..., k], v)  # (sys 0, env, spectator)
            rows = np.array([out[b[0], :, b[1]] for b in basis])  # env index left open
            rho += w * pk * rows @ rows.conj().T  # trace over the environment
    retained = np.trace(rho).real
    label = "channel:thermal loss (dilation)" if nth else "channel:loss (dilation)"
    _record(label, name, np.abs(state.density_matrix - rho / retained).max(), 1e-12)
    _record(f"retained:{label}", name, abs(engine.retained_norms[-1] - retained), 1e-12)
    if nth == 0:
        assert abs(retained - 1) < 1e-14  # pure loss never leaves the total-photon subspace


@pytest.mark.parametrize("name", NATIVE)
def test_long_sequence_physicality(name):
    engine = _backend(name, 12, norm_tolerance=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        engine.prepare(0, state=pg.GaussianInput.coherent(0.4))
        engine.prepare(1, state=pg.FockInput.number(1))
        engine.prepare(2, 0.2)
        for _ in range(3):
            engine.beamsplitter(0, 1, 0.7)
            engine.entangle(1, 2, 0.3)
            engine.kerr(0, 0.5)
            engine.cubic_phase(2, 0.1)
            engine.squeeze(1, 0.1)
            engine.displace(2, 0.1, 0.2)
            if _mixed(name):
                engine.loss(1, 0.9, 0.05)
    rho = _state_matrix(engine.get_state(), name)
    trace = np.trace(rho).real
    hermiticity = np.abs(rho - rho.conj().T).max()
    min_eig = np.linalg.eigvalsh((rho + rho.conj().T) / 2).min()
    _record(
        "physicality after 21-gate sequence",
        name,
        max(abs(trace - 1), hermiticity, -min_eig),
        1e-12,
    )
    reported = [d["retained_norm"] for d in engine.diagnostics]
    assert reported == engine.retained_norms and all(0 < x <= 1 + 1e-12 for x in reported)
