"""Independent single-mode oracle for one finite-squeezing CV-MBQC wire step.

The step is Prepare(new, r), Entangle(old, new), Measure(old, Homodyne(theta)) with
theta = atan2(1, k), then Displace(new, q=-h/sin(theta)). Conditioned on outcome h it
maps the input psi to K(h) psi / ||K(h) psi|| with (hbar=2, m = h/sin(theta), V = e^{2r})

    K(h) = M_V(x + m) A P(k),   M_V(z) = (2 pi V)^{-1/4} exp(-z^2 / (4V)),
    P(k) = exp(i k x^2 / 4),    A = exp(i pi n / 2),

and the outcome density in h is p(h) = ||K(h) psi||^2 / |sin(theta)|.

This is a conditional, non-unitary, outcome-dependent map: M_V is a weak position
measurement whose width is set by the finite resource squeezing. Only as r -> infinity
does it approach the deterministic Fourier-shear gate F P(k) (F = A). The oracle is not
library API; PhotoGraphiQ simulates the step natively on the two-mode Fock space.

Derivation (PhotoGraphiQ conventions). The resource wavefunction is s(y) = M_V(y) since
Prepare squeezes p (Var q = V). CZ is exp(i x y / 2). The homodyne value h of
q cos(theta) + p sin(theta) = sin(theta) (p + k q) is the eigenvalue m = h / sin(theta)
of P(k)^dag p P(k), and |h> = |p = m> / sqrt|sin(theta)|, which gives the Jacobian.
Projecting, <p=m| P(k) psi (x) s(y) e^{ixy/2} = s(y) chi~(m - y) with chi = P(k) psi and
chi~(p) = (4 pi)^{-1/2} int e^{-ipx/2} chi(x) dx. Displace(q=-m) shifts y -> y + m,
leaving M_V(y + m) chi~(-y), and chi~(-y) = (A chi)(y).

Adapted from photonic-qrc-eoc (https://github.com/chinmoybiswasdeep/photonic-qrc-eoc,
branch CV, commit 493f4c3f186ee314796964d6ddb6b2749171b7bb), MIT License, Copyright (c)
2026 Chinmoy Biswas; notebook 2c_adaptive_nongaussian_mbqrc_prl_preflight_colab, cells 17
and 23 (K_theta(m) and p(h)). The quadrature below is re-derived, not the notebook's DVR.
"""

import math

import numpy as np
import pytest

import photographiq as pg

X = np.linspace(-16, 16, 1601)
DX = X[1] - X[0]
LEVELS = 90


def _hermite_functions(levels):
    """hbar=2 Hermite functions psi_n(x) from q psi_n = sqrt(n+1) psi_{n+1} + sqrt(n) psi_{n-1}."""
    h = np.empty((levels, X.size))
    h[0] = (2 * np.pi) ** -0.25 * np.exp(-X * X / 4)
    h[1] = X * h[0]
    for n in range(2, levels):
        h[n] = (X * h[n - 1] - np.sqrt(n - 1) * h[n - 2]) / np.sqrt(n)
    return h


HERMITE = _hermite_functions(LEVELS)


def _coherent(alpha):
    """Analytic <x|alpha> in hbar=2 with the Fock phase convention alpha^n / sqrt(n!)."""
    return (2 * np.pi) ** -0.25 * np.exp(
        -X * X / 4 + alpha * X - alpha**2 / 2 - abs(alpha) ** 2 / 2
    )


def _coherent_amplitudes(alpha, levels):
    n = np.arange(levels)
    amplitudes = alpha**n / np.sqrt([math.factorial(i) for i in n]) * np.exp(-(abs(alpha) ** 2) / 2)
    return pg.FockInput(tuple(amplitudes / np.linalg.norm(amplitudes)))


def _kraus(psi, k, h, r, *, ideal=False):
    """Return (K psi on the y grid, p(h)); ideal=True drops M_V, giving A P(k) psi."""
    theta = math.atan2(1, k)
    m, variance = h / math.sin(theta), math.exp(2 * r)
    chi = np.exp(1j * k * X * X / 4) * psi
    rotated = np.exp(0.5j * np.outer(X, X)) @ chi * DX / np.sqrt(4 * np.pi)  # chi~(-y)
    if ideal:
        return rotated, None
    out = (2 * np.pi * variance) ** -0.25 * np.exp(-((X + m) ** 2) / (4 * variance)) * rotated
    return out, np.sum(abs(out) ** 2) * DX / abs(math.sin(theta))


def _fock_fidelity(wavefunction, state):
    """|<ref|state>|^2 with the reference projected on 90 levels and normed on the grid."""
    reference = HERMITE @ wavefunction * DX
    vector = np.zeros(LEVELS, complex)
    vector[[b[0] for b in state.basis]] = state.state_vector
    norm = np.sum(abs(wavefunction) ** 2) * DX
    return abs(np.vdot(reference, vector)) ** 2 / norm / np.vdot(vector, vector).real


CASES = {
    "fock1": (pg.FockInput.number(1), HERMITE[1], 0.4, 0.3, 0.3),
    "coherent": (_coherent_amplitudes(0.5 - 0.3j, 18), _coherent(0.5 - 0.3j), -0.9, -0.7, 0.2),
    "even-cat": (pg.FockInput.cat(0.8, 16), _coherent(0.8) + _coherent(-0.8), 0.0, 0.5, 0.25),
}


def test_oracle_grid_is_converged():
    for _, psi, k, h, r in CASES.values():
        out, _ = _kraus(psi / np.sqrt(np.sum(abs(psi) ** 2) * DX), k, h, r)
        reference = HERMITE @ out * DX
        assert 1 - np.vdot(reference, reference).real / (np.sum(abs(out) ** 2) * DX) < 1e-12


# Observed (total-photon cutoff 20/30/40/50/60): 1-F ~ 5e-6, 3e-8, 2e-10, 2e-12, 3e-14 and
# relative density error ~ 7e-5, 8e-7, 5e-9, 6e-11, 1e-13 (cat floors near 1e-11 from its
# 16-level input). Tolerances below sit about two decades above the observed values.
@pytest.mark.filterwarnings("ignore:GaussianTransform. retained Fock norm")
@pytest.mark.parametrize("name", CASES)
def test_native_wire_step_matches_finite_squeezing_kraus(name):
    state, psi, k, h, r = CASES[name]
    psi = psi / np.sqrt(np.sum(abs(psi) ** 2) * DX)
    expected, density = _kraus(psi, k, h, r)
    infidelities, density_errors = [], []
    for cutoff in (30, 45, 60):
        result = pg.simulate(
            pg.protocols.wire((k,), squeezing=r),
            backend="piquasso-fock",
            cutoff=cutoff,
            inputs={0: state},
            measurement_outcomes={0: h},
        )
        assert result.state.nodes == (1,)
        infidelities.append(1 - _fock_fidelity(expected, result.state))
        density_errors.append(abs(math.exp(result.log_likelihood) / density - 1))
    assert infidelities[0] > infidelities[1] and density_errors[0] > density_errors[1]
    assert infidelities[0] < 1e-6 and density_errors[0] < 1e-4
    assert infidelities[-1] < 1e-11 and density_errors[-1] < 1e-9
    # Finite r is not the ideal gate: the conditional output is measurably off A P(k) psi.
    ideal, _ = _kraus(psi, k, h, r, ideal=True)
    assert 1 - _fock_fidelity(ideal, result.state) > 1e-3
