"""Gate references from direct quadrature integrals with independent Hermites."""

import numpy as np
import pytest
from scipy.integrate import simpson

import photographiq as pg
from photographiq.backends.fock import PiquassoFockBackend
from tests.gkp.convergence.test_modular_reference import reference_density
from tests.v03_reference import hermites


@pytest.mark.parametrize("gate", ["H", "S"])
@pytest.mark.parametrize("coefficients", [(1, 0), (0, 1), (1, 1), (1, -1)])
def test_finite_gate_against_quadrature_reference(gate, coefficients):
    errors = []
    for cutoff in (24, 48, 80):
        code = pg.GKPCode(0.55, 0.5, cutoff, 6, 4097)
        source = code.encode(*coefficients)
        p = pg.Pattern(inputs=(0,)).append(code.logical_gate(0, gate))
        result = pg.simulate(
            p, inputs={0: source}, backend=PiquassoFockBackend(cutoff, norm_tolerance=0.05)
        ).state
        # Evaluate input Fock polynomial independently, on a resolved real line.
        q = np.linspace(-25, 25, 2001)
        h = hermites(q, cutoff)
        if gate == "S":
            transformed = (h @ np.array(source.amplitudes)) * np.exp(0.25j * q * q)
            reference = simpson(h * transformed[:, None], x=q, axis=0)
            reference /= np.linalg.norm(reference)
        else:
            reference = np.array(source.amplitudes) * np.exp(0.5j * np.pi * np.arange(cutoff))
        # Piquasso's Gaussian decomposition chooses an unobservable global phase.
        phase = np.vdot(reference, result.state_vector)
        errors.append(np.linalg.norm(result.state_vector - reference * phase / abs(phase)))
        target_coefficients = (
            np.array([[1, 1], [1, -1]]) / np.sqrt(2) if gate == "H" else np.diag([1, 1j])
        ) @ coefficients
        target = code.encode(*target_coefficients)
        fidelity = abs(np.vdot(target.amplitudes, result.state_vector)) ** 2
        leakage = code.leakage(result.state_vector)
        assert 0 <= leakage <= 1 and 0.4 < fidelity <= 1 + 1e-12
    assert errors[-1] < 2e-8


def test_ideal_lattice_phase_and_cz_identity():
    for m in range(-6, 7):
        q = m * np.sqrt(2 * np.pi)
        assert np.exp(0.25j * q * q) == pytest.approx(1 if m % 2 == 0 else 1j)
        for n in range(-6, 7):
            assert np.exp(0.5j * q * n * np.sqrt(2 * np.pi)) == pytest.approx((-1) ** (m * n))


def test_independent_fourier_density_normalization():
    from scipy.integrate import quad

    code = pg.GKPCode(0.55, 0.5)
    for basis in ("X", "Z"):
        assert quad(reference_density(code, (1, 1j), basis), -30, 30, limit=300)[
            0
        ] == pytest.approx(1, abs=1e-9)
