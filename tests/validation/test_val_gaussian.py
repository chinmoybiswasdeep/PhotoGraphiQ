"""Independent validation of the Gaussian engines (gaussian, native-gaussian, piquasso).

Every reference here is built in the test from first principles with NumPy/SciPy only:

* Symplectic matrices come from the quadratic generator of each gate, U = exp(-i H) with
  H = x^T M x / 2 and [x_j, x_k] = 2i Omega_jk (hbar = 2), whose Heisenberg action is
  S = expm(2 Omega M). That map itself is checked against the Fock-space Heisenberg action
  of the same generators (``test_hamiltonian_map_matches_fock_heisenberg_action``).
* Gaussian moments of vacuum/coherent/squeezed/thermal states are evaluated from dense
  Fock-space operators on a large single-mode space (q = a + a^dag, p = -i(a - a^dag)).
* Homodyne/heterodyne conditioning is derived from a physical dilation (loss beam splitter
  with a vacuum ancilla for inefficiency; 50:50 beam splitter plus two ideal homodynes for
  heterodyne) and the PRECISION-matrix form of Gaussian conditioning, not the Schur-complement
  formula the backends implement.
* No PhotoGraphiQ function is used on the reference side (only to run the engines and, where
  stated, to read back its own diagnostics, which are the quantities under test).

Tolerances: float64 roundoff of a few dozen products of O(1)-O(10) matrices is ~1e-14; the
asserted bounds (1e-11 for moments, 1e-10 relative for densities) leave >= 15x headroom over
the measured errors listed in docs/validation/independent-validation-matrix.md (the largest,
6.5e-13, is the precision-matrix reference itself on a covariance with condition ~4e3). Fock-derived
references are truncated at dimension 160, where omitted weight is < 1e-30.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from scipy.linalg import block_diag, expm

import photographiq as pg
from photographiq import registry
from photographiq.states import GaussianState

BACKENDS = ["gaussian", "native-gaussian", "piquasso"]
MEASURED: dict = {}  # (operation, backend) -> (max error, tolerance, test); see the matrix doc


def _record(operation, backend, error, tolerance):
    key = (operation, backend)
    test = os.environ.get("PYTEST_CURRENT_TEST", "").split("[")[0].split(" ")[0]
    previous = MEASURED.get(key, (0.0, tolerance, test))
    MEASURED[key] = (max(previous[0], float(error)), tolerance, previous[2])
    assert error <= tolerance, f"{operation} on {backend}: error {error:.3e} > {tolerance:.1e}"


def _backend(name):
    if name == "piquasso":
        pytest.importorskip("piquasso")
    if registry.BACKENDS[name].unavailable_reason() is not None:
        pytest.skip(registry.BACKENDS[name].unavailable_reason())
    engine = registry.create_backend(name)
    engine.reset(7)
    return engine


# ----- first-principles phase-space algebra --------------------------------------------
def omega(n):
    out = np.zeros((2 * n, 2 * n))
    for k in range(n):
        out[2 * k, 2 * k + 1], out[2 * k + 1, 2 * k] = 1.0, -1.0
    return out


def from_generator(m):
    """Heisenberg map of U = exp(-i x^T M x / 2) with [x_j, x_k] = 2i Omega_jk."""
    m = np.asarray(m, dtype=float)
    return expm(2 * omega(len(m) // 2) @ m)


def gen_rotation(phi):  # U = exp(i phi n), n = (q^2 + p^2 - 2) / 4
    return -phi / 2 * np.eye(2)


def gen_squeeze(r):  # U = exp(r (a^2 - a^dag^2) / 2) = exp(i r (qp + pq) / 4)
    return np.array([[0.0, -r / 2], [-r / 2, 0.0]])


def gen_quadratic(s):  # U = exp(i s q^2 / 4)
    return np.diag([-s / 2, 0.0])


def gen_cz(g):  # U = exp(i g q_u q_v / 2)
    m = np.zeros((4, 4))
    m[0, 2] = m[2, 0] = -g / 2
    return m


def gen_beamsplitter(theta):  # U = exp(theta (a_v^dag a_u - a_u^dag a_v))
    m = np.zeros((4, 4))
    m[1, 2] = m[2, 1] = -theta / 2  # -theta/2 (q_v p_u + p_u q_v)
    m[0, 3] = m[3, 0] = theta / 2  # +theta/2 (p_v q_u + q_u p_v)
    return m


def embed(matrix, modes, n):
    idx = [j for k in modes for j in (2 * k, 2 * k + 1)]
    out = np.eye(2 * n)
    out[np.ix_(idx, idx)] = matrix
    return out


def random_state(rng, n, nus=None):
    """Random mixed Gaussian state V = S diag(nu) S^T, S from a random quadratic generator."""
    h = rng.normal(size=(2 * n, 2 * n))
    s = from_generator(0.2 * (h + h.T))
    nus = rng.uniform(1.0, 2.0, size=n) if nus is None else np.asarray(nus, dtype=float)
    cov = s @ np.diag(np.repeat(nus, 2)) @ s.T
    return rng.normal(scale=0.7, size=2 * n), (cov + cov.T) / 2, nus


def loss_xy(eta, nth):
    return np.sqrt(eta) * np.eye(2), (1 - eta) * (2 * nth + 1) * np.eye(2)


def condition_precision(mean, cov, cross, outcome_mean, outcome_cov, outcome):
    """Gaussian conditioning of x on y via the joint PRECISION matrix (not the Schur form).

    ``cross`` = Cov(x, y). Returns conditional mean, covariance and the density of y.
    """
    joint = np.block([[cov, cross], [cross.T, outcome_cov]])
    k = len(mean)
    lam = np.linalg.inv(joint)
    cond_cov = np.linalg.inv(lam[:k, :k])
    shift = np.atleast_1d(outcome) - outcome_mean
    cond_mean = mean - cond_cov @ lam[:k, k:] @ shift
    density = np.exp(-0.5 * shift @ np.linalg.solve(outcome_cov, shift)) / np.sqrt(
        np.linalg.det(2 * np.pi * outcome_cov)
    )
    return cond_mean, (cond_cov + cond_cov.T) / 2, float(density)


# ----- dense Fock-space operators (independent of the package) --------------------------
def ladder(dim):
    return np.diag(np.sqrt(np.arange(1, dim)), 1).astype(complex)


def fock_moments(rho, a):
    q, p = a + a.conj().T, -1j * (a - a.conj().T)
    ops = [q, p]
    mean = np.array([np.trace(rho @ x).real for x in ops])
    second = np.array([[np.trace(rho @ (x @ y + y @ x)).real / 2 for y in ops] for x in ops])
    return mean, second - np.outer(mean, mean)


# ----- conventions ------------------------------------------------------------------------
def test_conventions_hbar2_commutator_and_vacuum():
    dim = 40
    a = ladder(dim)
    q, p = a + a.conj().T, -1j * (a - a.conj().T)
    # [q, p] = 2i on the block unaffected by truncation.
    np.testing.assert_allclose((q @ p - p @ q)[: dim - 1, : dim - 1], 2j * np.eye(dim - 1))
    vac = np.zeros((dim, dim), complex)
    vac[0, 0] = 1
    mean, cov = fock_moments(vac, a)
    np.testing.assert_allclose(cov, np.eye(2), atol=1e-15)  # vacuum covariance I
    np.testing.assert_array_equal(pg.gaussian.omega(3), omega(3))  # interleaved (q, p)


def test_hamiltonian_map_matches_fock_heisenberg_action():
    """S = expm(2 Omega M) equals <U^dag x U> moments of the Fock generators (dim 30/mode)."""
    dim = 30
    a1 = ladder(dim)
    alpha = 0.3 - 0.2j
    coh = expm(alpha * a1.conj().T - np.conj(alpha) * a1)[:, 0]
    one = np.eye(dim)
    au, av = np.kron(a1, one), np.kron(one, a1)
    xs = [au + au.conj().T, -1j * (au - au.conj().T), av + av.conj().T, -1j * (av - av.conj().T)]
    psi = np.kron(coh, expm(0.25 * a1.conj().T - 0.25 * a1)[:, 0])
    cases = {
        "cz": (gen_cz(0.7), expm(1j * 0.7 * xs[0] @ xs[2] / 2)),
        "bs": (gen_beamsplitter(0.4), expm(0.4 * (av.conj().T @ au - au.conj().T @ av))),
        "rot+sq+P": (
            block_diag(gen_rotation(0.3) + gen_squeeze(0.2) + gen_quadratic(0.5), np.zeros((2, 2))),
            expm(
                0.3j * au.conj().T @ au
                + 0.2 * (au @ au - au.conj().T @ au.conj().T) / 2
                + 0.5j * xs[0] @ xs[0] / 4
            ),
        ),
    }
    for m, u in cases.values():
        out = u @ psi
        mean_in = np.array([np.vdot(psi, x @ psi).real for x in xs])
        mean_out = np.array([np.vdot(out, x @ out).real for x in xs])
        np.testing.assert_allclose(mean_out, from_generator(m) @ mean_in, atol=1e-10)


# ----- states -------------------------------------------------------------------------
@pytest.mark.parametrize("name", BACKENDS)
def test_vacuum_coherent_squeezed_thermal_states(name):
    dim = 160
    a = ladder(dim)
    alpha, r, theta, nbar = 0.6 - 0.45j, 0.55, 0.4, 0.7
    coherent = expm(alpha * a.conj().T - np.conj(alpha) * a)[:, 0]
    squeezed = np.exp(1j * theta * np.arange(dim)) * expm(r * (a @ a - a.T @ a.T) / 2)[:, 0]
    weights = nbar ** np.arange(dim) / (1 + nbar) ** np.arange(1, dim + 1)
    references = {
        "vacuum": (pg.GaussianInput(), np.outer(np.eye(dim)[0], np.eye(dim)[0])),
        "coherent": (pg.GaussianInput.coherent(alpha), np.outer(coherent, coherent.conj())),
        "squeezed": (pg.GaussianInput.squeezed(r, theta), np.outer(squeezed, squeezed.conj())),
        "thermal": (
            pg.GaussianInput(covariance=tuple(map(tuple, (2 * nbar + 1) * np.eye(2)))),
            np.diag(weights).astype(complex),
        ),
    }
    for label, (prep, rho) in references.items():
        engine = _backend(name)
        engine.prepare("m", state=prep)
        state = engine.get_state()
        mean, cov = fock_moments(rho, a)
        error = max(np.abs(state.mean - mean).max(), np.abs(state.covariance - cov).max())
        _record(f"state:{label}", name, error, 1e-11)
        purity = np.trace(rho @ rho).real
        _record(f"purity:{label}", name, abs(state.purity() - purity), 1e-12)
    # Default resource: momentum-squeezed vacuum, V = diag(e^{2r}, e^{-2r}).
    engine = _backend(name)
    engine.prepare("m", r)
    error = np.abs(engine.get_state().covariance - np.diag(np.exp([2 * r, -2 * r]))).max()
    _record("state:resource p-squeezed", name, error, 1e-12)


def test_thermal_entropy_and_native_diagnostics():
    from photographiq.backends import native_gaussian as ng

    nbar, dim = 0.7, 400
    p = nbar ** np.arange(dim) / (1 + nbar) ** np.arange(1, dim + 1)
    entropy = -np.sum(p * np.log(p))
    state = GaussianState(np.zeros(2), (2 * nbar + 1) * np.eye(2), ("m",))
    for label, value in (("numpy", state.entropy()), ("native", ng.entropy(state))):
        _record("entropy:thermal", label, abs(value - entropy), 1e-12)
    _record("purity:thermal", "native-fn", abs(ng.purity(state) - np.sum(p**2)), 1e-12)


# ----- Gaussian unitaries ----------------------------------------------------------------
GATES = {
    "rotation": (lambda e: e.rotate(1, 0.7), (1,), gen_rotation(0.7)),
    "squeeze": (lambda e: e.squeeze(0, -0.45), (0,), gen_squeeze(-0.45)),
    "quadratic_phase": (lambda e: e.quadratic_phase(2, 0.6), (2,), gen_quadratic(0.6)),
    "beamsplitter": (lambda e: e.beamsplitter(0, 2, 0.6), (0, 2), gen_beamsplitter(0.6)),
    "cz": (lambda e: e.entangle(2, 1, -0.8), (2, 1), gen_cz(-0.8)),
}


@pytest.mark.parametrize("gate", list(GATES))
@pytest.mark.parametrize("name", BACKENDS)
def test_gaussian_gates_against_generator_symplectic(name, gate):
    rng = np.random.default_rng(11)
    mean, cov, _ = random_state(rng, 3)
    apply, modes, m = GATES[gate]
    local = from_generator(m)
    np.testing.assert_allclose(local @ omega(len(modes)) @ local.T, omega(len(modes)), atol=1e-14)
    s = embed(local, modes, 3)
    engine = _backend(name)
    engine.set_state(GaussianState(mean, cov, (0, 1, 2)))
    apply(engine)
    out = engine.get_state()
    error = max(np.abs(out.mean - s @ mean).max(), np.abs(out.covariance - s @ cov @ s.T).max())
    _record(f"gate:{gate}", name, error, 1e-11)


@pytest.mark.parametrize("name", BACKENDS)
def test_displacement_shifts_mean_only(name):
    rng = np.random.default_rng(3)
    mean, cov, _ = random_state(rng, 2)
    engine = _backend(name)
    engine.set_state(GaussianState(mean, cov, ("a", "b")))
    engine.displace("b", 0.37, -1.2)
    out = engine.get_state()
    error = max(
        np.abs(out.mean - (mean + [0, 0, 0.37, -1.2])).max(), np.abs(out.covariance - cov).max()
    )
    _record("gate:displacement", name, error, 1e-12)


# ----- channels -------------------------------------------------------------------------
@pytest.mark.parametrize("eta,nth", [(0.63, 0.0), (0.4, 0.35), (1.0, 0.5), (0.0, 0.2)])
@pytest.mark.parametrize("name", BACKENDS)
def test_loss_channel_against_xy_and_beam_splitter_dilation(name, eta, nth):
    rng = np.random.default_rng(5)
    mean, cov, _ = random_state(rng, 2)
    x, y = loss_xy(eta, nth)
    xf, yf = block_diag(np.eye(2), x), block_diag(np.zeros((2, 2)), y)
    analytic = (xf @ mean, xf @ cov @ xf.T + yf)
    # Dilation: mode 1 meets a thermal environment on a beam splitter cos^2(theta) = eta.
    big = block_diag(cov, (2 * nth + 1) * np.eye(2))
    bs = embed(from_generator(gen_beamsplitter(np.arccos(np.sqrt(eta)))), (1, 2), 3)
    dil_mean, dil_cov = (bs @ np.r_[mean, 0, 0])[:4], (bs @ big @ bs.T)[:4, :4]
    np.testing.assert_allclose(dil_cov, analytic[1], atol=1e-13)
    np.testing.assert_allclose(dil_mean, analytic[0], atol=1e-14)
    engine = _backend(name)
    engine.set_state(GaussianState(mean, cov, ("a", "b")))
    engine.loss("b", eta, nth)
    out = engine.get_state()
    error = max(np.abs(out.mean - analytic[0]).max(), np.abs(out.covariance - analytic[1]).max())
    _record("channel:thermal-loss" if nth else "channel:loss", name, error, 1e-11)


# ----- measurements ----------------------------------------------------------------------
def _homodyne_reference(mean, cov, mode, theta, eta, noise, outcome):
    """Rotate by -theta, mix with a vacuum ancilla (cos^2 = eta), read q, rescale by 1/sqrt(eta)."""
    n = len(mean) // 2
    s = embed(from_generator(gen_rotation(-theta)), (mode,), n + 1)
    s = embed(from_generator(gen_beamsplitter(np.arccos(np.sqrt(eta)))), (mode, n), n + 1) @ s
    big_mean, big_cov = s @ np.r_[mean, 0, 0], s @ block_diag(cov, np.eye(2)) @ s.T
    read = np.zeros((1, 2 * n + 2))
    read[0, 2 * mode] = 1 / np.sqrt(eta)
    rest = [j for j in range(2 * n) if j // 2 != mode]
    return condition_precision(
        big_mean[rest],
        big_cov[np.ix_(rest, rest)],
        big_cov[rest] @ read.T,
        read @ big_mean,
        read @ big_cov @ read.T + noise,
        outcome,
    )


@pytest.mark.parametrize(
    "theta,eta,noise,outcome", [(0.0, 1.0, 0.0, 0.8), (1.1, 1.0, 0.0, -1.3), (2.3, 0.7, 0.15, 0.4)]
)
@pytest.mark.parametrize("name", BACKENDS)
def test_homodyne_conditioning_and_density(name, theta, eta, noise, outcome):
    rng = np.random.default_rng(17)
    mean, cov, _ = random_state(rng, 3)
    ref_mean, ref_cov, ref_density = _homodyne_reference(mean, cov, 1, theta, eta, noise, outcome)
    engine = _backend(name)
    engine.set_state(GaussianState(mean, cov, ("a", "b", "c")))
    value = engine.measure("b", pg.Homodyne(theta, eta, noise), theta, outcome=outcome)
    out = engine.get_state()
    assert value == outcome and out.nodes == ("a", "c")
    label = "homodyne" if eta == 1 and noise == 0 else "homodyne:inefficient+noise"
    error = max(np.abs(out.mean - ref_mean).max(), np.abs(out.covariance - ref_cov).max())
    _record(f"{label}:conditional", name, error, 1e-11)
    density = engine.last_measurement["value"]
    _record(f"{label}:density", name, abs(density / ref_density - 1), 1e-10)


def _heterodyne_reference(mean, cov, mode, outcome):
    """50:50 beam splitter with vacuum ancilla, ideal q and p homodynes, outcomes x sqrt(2)."""
    n = len(mean) // 2
    s = embed(from_generator(gen_beamsplitter(np.pi / 4)), (mode, n), n + 1)
    big_mean, big_cov = s @ np.r_[mean, 0, 0], s @ block_diag(cov, np.eye(2)) @ s.T
    read = np.zeros((2, 2 * n + 2))
    read[0, 2 * mode] = read[1, 2 * n + 1] = np.sqrt(2)
    rest = [j for j in range(2 * n) if j // 2 != mode]
    return condition_precision(
        big_mean[rest],
        big_cov[np.ix_(rest, rest)],
        big_cov[rest] @ read.T,
        read @ big_mean,
        read @ big_cov @ read.T,
        outcome,
    )


@pytest.mark.parametrize("name", BACKENDS)
def test_heterodyne_conditioning_and_density(name):
    rng = np.random.default_rng(23)
    mean, cov, _ = random_state(rng, 3)
    outcome = (0.7, -1.1)
    ref_mean, ref_cov, ref_density = _heterodyne_reference(mean, cov, 0, outcome)
    engine = _backend(name)
    engine.set_state(GaussianState(mean, cov, ("a", "b", "c")))
    engine.measure("a", pg.Heterodyne(), outcome=outcome)
    out = engine.get_state()
    error = max(np.abs(out.mean - ref_mean).max(), np.abs(out.covariance - ref_cov).max())
    _record("heterodyne:conditional", name, error, 1e-11)
    density = engine.last_measurement["value"]
    _record("heterodyne:density", name, abs(density / ref_density - 1), 1e-10)


@pytest.mark.parametrize("name", BACKENDS)
def test_heterodyne_density_is_husimi_function(name):
    """Single mode: density(q, p) = |<alpha|psi>|^2 / (4 pi), alpha = (q + i p)/2 (Fock dim 120)."""
    dim, r, theta, beta = 120, 0.4, 0.9, 0.3 + 0.5j
    a = ladder(dim)
    psi = expm(beta * a.conj().T - np.conj(beta) * a) @ (
        np.exp(1j * theta * np.arange(dim)) * expm(r * (a @ a - a.T @ a.T) / 2)[:, 0]
    )
    engine = _backend(name)
    engine.prepare("m", state=pg.GaussianInput.squeezed(r, theta))
    engine.displace("m", 2 * beta.real, 2 * beta.imag)
    worst = 0.0
    for q, p in [(0.1, 0.2), (1.4, -0.6), (-2.0, 2.5)]:
        alpha = complex(q, p) / 2
        coh = expm(alpha * a.conj().T - np.conj(alpha) * a)[:, 0]
        husimi = abs(np.vdot(coh, psi)) ** 2 / (4 * np.pi)
        trial = registry.create_backend(name)
        trial.set_state(engine.get_state())
        trial.measure("m", pg.Heterodyne(), outcome=(q, p))
        worst = max(worst, abs(trial.last_measurement["value"] / husimi - 1))
    _record("heterodyne:husimi density", name, worst, 1e-10)


# ----- adaptive feed-forward -------------------------------------------------------------
def _wire_pattern(r):
    first = np.arctan2(1, 0.3)
    second = pg.expressions.atan2(1.0, 0.2 + 0.5 * pg.Outcome("a"))
    return pg.Pattern(inputs=("L",)).extend(
        [
            pg.Prepare("C1", r),
            pg.Entangle("L", "C1", 1.0),
            pg.Measure("L", pg.Homodyne(first), "a"),
            pg.Displace("C1", q=-pg.Outcome("a") / np.sin(first)),
            pg.Prepare("C2", r),
            pg.Entangle("C1", "C2", 1.0),
            pg.Measure("C1", pg.Homodyne(second), "b"),
            pg.Displace("C2", q=-pg.Outcome("b") / pg.sin(second)),
            pg.Output(("C2",)),
        ]
    )


def _wire_step(mean, cov, r, theta, outcome):
    """One teleportation step by hand: append p-squeezed C, CZ, homodyne L at theta, shift."""
    mean = np.r_[mean, 0, 0]
    cov = block_diag(cov, np.diag(np.exp([2 * r, -2 * r])))
    s = from_generator(gen_cz(1.0))
    mean, cov = s @ mean, s @ cov @ s.T
    m, c, density = _homodyne_reference(mean, cov, 0, theta, 1.0, 0.0, outcome)
    return m + [-outcome / np.sin(theta), 0], c, density


@pytest.mark.parametrize("name", BACKENDS)
def test_adaptive_wire_against_hand_teleportation(name):
    r, alpha = 0.6, 0.3 - 0.2j
    mean0, cov0 = np.array([2 * alpha.real, 2 * alpha.imag]), np.eye(2)
    inputs = {"L": pg.GaussianInput.coherent(alpha)}
    for outcomes in ({"a": 0.4, "b": -0.7}, None):
        result = pg.simulate(
            _wire_pattern(r),
            backend=name,
            inputs=inputs,
            measurement_outcomes=outcomes,
            seed=4,
        )
        a, b = result.outcomes["a"], result.outcomes["b"]
        m1, c1, d1 = _wire_step(mean0, cov0, r, np.arctan2(1, 0.3), a)
        m2, c2, d2 = _wire_step(m1, c1, r, np.arctan2(1, 0.2 + 0.5 * a), b)
        out = result.state
        error = max(np.abs(out.mean - m2).max(), np.abs(out.covariance - c2).max())
        label = "postselected" if outcomes else "sampled"
        _record(f"feed-forward wire:{label}", name, error, 1e-11)
        _record(
            f"feed-forward wire:{label} likelihood",
            name,
            abs(result.log_likelihood - np.log(d1 * d2)),
            1e-10,
        )


@pytest.mark.parametrize("name", BACKENDS)
def test_wire_step_ensemble_is_analytic_teleportation_map(name):
    """E_m[conditional state] = T(k) rho T(k)^T + diag(0, e^{-2r}), T(k) = [[-k, -1], [1, 0]]."""
    rng = np.random.default_rng(9)
    mean0, cov0, _ = random_state(rng, 1)
    r, k = 0.5, 0.3
    theta = np.arctan2(1, k)
    t = np.array([[-k, -1.0], [1.0, 0.0]])
    states, expected_m = [], None
    for outcome in (-0.9, 1.6):
        engine = _backend(name)
        engine.set_state(GaussianState(mean0, cov0, ("L",)))
        engine.prepare("C", r)
        engine.entangle("L", "C", 1.0)
        engine.measure("L", pg.Homodyne(theta), theta, outcome=outcome)
        engine.displace("C", -outcome / np.sin(theta), 0.0)
        states.append((outcome, engine.get_state()))
    (m1, s1), (m2, s2) = states
    gain = (s1.mean - s2.mean) / (m1 - m2)
    # Outcome variance by hand: m = q_L cos + (p_L + q_C) sin.
    h = np.array([np.cos(theta), np.sin(theta)])
    variance = h @ cov0 @ h + np.sin(theta) ** 2 * np.exp(2 * r)
    expected_m = h @ mean0
    ensemble = s1.covariance + variance * np.outer(gain, gain)
    channel = t @ cov0 @ t.T + np.diag([0.0, np.exp(-2 * r)])
    mean_at_expected = s1.mean + gain * (expected_m - m1)
    error = max(np.abs(ensemble - channel).max(), np.abs(mean_at_expected - t @ mean0).max())
    _record("feed-forward wire:ensemble = T rho T^T + N", name, error, 1e-11)


# ----- multimode random circuits and mixed states ------------------------------------------
@pytest.mark.parametrize("name", BACKENDS)
def test_random_multimode_circuit_physicality(name):
    rng = np.random.default_rng(29)
    n = 5
    mean, cov, _ = random_state(rng, n)
    engine = _backend(name)
    engine.set_state(GaussianState(mean, cov, tuple(range(n))))
    for _ in range(40):
        kind = rng.integers(6)
        u, v = (int(x) for x in rng.choice(n, 2, replace=False))
        value = float(rng.normal(scale=0.6))
        if kind == 0:
            engine.rotate(u, value)
            s = embed(from_generator(gen_rotation(value)), (u,), n)
        elif kind == 1:
            engine.squeeze(u, value / 2)
            s = embed(from_generator(gen_squeeze(value / 2)), (u,), n)
        elif kind == 2:
            engine.beamsplitter(u, v, value)
            s = embed(from_generator(gen_beamsplitter(value)), (u, v), n)
        elif kind == 3:
            engine.entangle(u, v, value)
            s = embed(from_generator(gen_cz(value)), (u, v), n)
        elif kind == 4:
            engine.displace(u, value, -value)
            mean[2 * u : 2 * u + 2] += [value, -value]
            continue
        else:
            eta, nth = float(rng.uniform(0.5, 1)), float(rng.uniform(0, 0.3))
            engine.loss(u, eta, nth)
            x, y = loss_xy(eta, nth)
            xf = embed(x, (u,), n)
            yf = np.zeros((2 * n, 2 * n))
            yf[2 * u : 2 * u + 2, 2 * u : 2 * u + 2] = y
            mean, cov = xf @ mean, xf @ cov @ xf.T + yf
            continue
        mean, cov = s @ mean, s @ cov @ s.T
    out = engine.get_state()
    scale = np.abs(cov).max()
    error = max(np.abs(out.mean - mean).max(), np.abs(out.covariance - cov).max() / scale)
    _record("random 5-mode circuit (rel.)", name, error, 1e-11)
    physical = np.linalg.eigvalsh(out.covariance + 1j * omega(n)).min()
    nus = np.sort(np.abs(np.linalg.eigvals(1j * omega(n) @ out.covariance)))[::2]
    assert physical > -1e-9 and nus.min() > 1 - 1e-9
    np.testing.assert_array_equal(out.covariance, out.covariance.T)


def _g(nu):
    a, b = (nu + 1) / 2, (nu - 1) / 2
    return a * np.log(a) - (b * np.log(b) if b > 0 else 0.0)


@pytest.mark.parametrize("name", BACKENDS)
def test_mixed_state_spectral_diagnostics(name):
    from photographiq.backends import native_gaussian as ng

    rng = np.random.default_rng(31)
    nus = np.array([1.0, 1.7, 3.2])
    mean, cov, _ = random_state(rng, 3, nus)
    engine = _backend(name)
    engine.set_state(GaussianState(mean, cov, (0, 1, 2)))
    engine.beamsplitter(0, 2, 0.4)  # passive: spectrum invariant
    engine.entangle(1, 2, 0.9)
    state = engine.get_state()
    diagnostics = {
        "symplectic eigenvalues": (state.symplectic_eigenvalues(), np.sort(nus)),
        "purity": (state.purity(), 1 / np.prod(nus)),
        "entropy": (state.entropy(), sum(_g(x) for x in nus)),
    }
    if name == "native-gaussian":
        diagnostics.update(
            {
                "symplectic eigenvalues": (ng.symplectic_eigenvalues(state), np.sort(nus)),
                "purity": (ng.purity(state), 1 / np.prod(nus)),
                "entropy": (ng.entropy(state), sum(_g(x) for x in nus)),
            }
        )
    for label, (value, expected) in diagnostics.items():
        _record(f"mixed state:{label}", name, np.abs(np.asarray(value) - expected).max(), 1e-10)


@pytest.mark.parametrize("eta", [1.0, 0.6])
@pytest.mark.parametrize("name", BACKENDS)
def test_tmsv_log_negativity(name, eta):
    from photographiq.backends import native_gaussian as ng

    r = 0.7
    engine = _backend(name)
    engine.prepare("A", state=pg.GaussianInput.squeezed(r))
    engine.prepare("B", state=pg.GaussianInput.squeezed(r, np.pi / 2))
    engine.beamsplitter("A", "B", np.pi / 4)
    engine.loss("A", eta)
    engine.loss("B", eta)
    state = engine.get_state()
    a, c = eta * np.cosh(2 * r) + 1 - eta, eta * np.sinh(2 * r)
    expected = max(0.0, -np.log2(a - c))  # = 2r / ln 2 at eta = 1
    values = [state.log_negativity(("A",))]
    if name == "native-gaussian":
        values.append(ng.log_negativity(state, ("A",)))
    error = max(abs(v - expected) for v in values)
    _record("log-negativity TMSV" + ("" if eta == 1 else " (lossy)"), name, error, 1e-11)


# ----- consistency of the shared state containers ----------------------------------------
@pytest.mark.parametrize("name", BACKENDS)
def test_quadrature_convention_q_cos_plus_p_sin(name):
    engine = _backend(name)
    engine.prepare("m", state=pg.GaussianInput.coherent(0.4 + 0.9j))
    for angle in (0.0, 0.6, np.pi / 2):
        value, variance = engine.get_state().quadrature("m", angle)
        expected = 0.8 * np.cos(angle) + 1.8 * np.sin(angle)
        _record("quadrature q cos + p sin", name, abs(value - expected) + abs(variance - 1), 1e-14)
