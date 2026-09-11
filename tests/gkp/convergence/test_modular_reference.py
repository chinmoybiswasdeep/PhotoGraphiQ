"""Independent continuous-comb Fourier reference, not Fock overlap normalization."""

import numpy as np
import pytest
from scipy.integrate import quad

import photographiq as pg


def reference_density(code, coefficients, basis):
    width, envelope = code.peak_width, code.envelope
    centers = [
        np.sqrt(2 * np.pi) * (2 * np.arange(-code.peaks, code.peaks + 1) + b) for b in (0, 1)
    ]
    weights = [np.exp(-(envelope**2) * c * c / 4) for c in centers]

    # Closed integrals normalize individual codewords and their complex superposition.
    def inner(i, j):
        return (
            np.sqrt(2 * np.pi)
            * width
            * np.sum(
                weights[i][:, None]
                * weights[j][None, :]
                * np.exp(-((centers[i][:, None] - centers[j][None, :]) ** 2) / (8 * width**2))
            )
        )

    norms = np.sqrt([inner(0, 0), inner(1, 1)])
    amplitudes = np.array(coefficients, complex) / norms
    norm = sum(
        amplitudes[i].conjugate() * amplitudes[j] * inner(i, j) for i in (0, 1) for j in (0, 1)
    ).real

    def density(x):
        if basis == "Z":
            psi = sum(
                amplitudes[i]
                * np.sum(weights[i] * np.exp(-((x - centers[i]) ** 2) / (4 * width**2)))
                for i in (0, 1)
            )
        else:
            # hbar=2 Fourier kernel exp(-ipq/2)/sqrt(4*pi).
            psi = (
                width
                * np.exp(-(width**2) * x * x / 4)
                * sum(
                    amplitudes[i] * np.sum(weights[i] * np.exp(-0.5j * x * centers[i]))
                    for i in (0, 1)
                )
            )
        return abs(psi) ** 2 / norm

    return density


def reference_metrics(density):
    probabilities = np.zeros(2)
    residual_second = 0.0
    L = np.sqrt(2 * np.pi)
    for cell in range(-16, 17):
        lo, hi = (cell - 0.5) * L, (cell + 0.5) * L
        probabilities[cell % 2] += quad(density, lo, hi, epsabs=1e-11)[0]
        residual_second += quad(lambda x: (x - cell * L) ** 2 * density(x), lo, hi, epsabs=1e-11)[0]
    return probabilities, residual_second


@pytest.mark.parametrize("width,envelope", [(0.4, 0.4), (0.55, 0.5), (0.7, 0.35)])
@pytest.mark.parametrize(
    "basis,coefficients,correct",
    [("Z", (1, 0), 0), ("Z", (0, 1), 1), ("X", (1, 1), 0), ("X", (1, -1), 1)],
)
def test_physical_readout_converges_to_independent_continuous_reference(
    width, envelope, basis, coefficients, correct
):
    errors = []
    for cutoff in (24, 48, 112):
        code = pg.GKPCode(width, envelope, cutoff, 6, 4097)
        state = np.array(code.encode(*coefficients).amplitudes)
        effects = pg.modular_effects(cutoff, basis)
        probabilities = np.einsum("i,kij,j->k", state.conj(), effects, state).real
        expected, _ = reference_metrics(reference_density(code, coefficients, basis))
        errors.append(np.max(abs(probabilities - expected)))
        assert np.linalg.eigvalsh(effects[0]).min() > -1e-10
    assert errors[-1] < 3e-7
    assert errors[-1] < errors[0]
    assert 0.8 < probabilities[correct] < 1


def test_grid_and_peak_refinement_on_decoded_probabilities():
    effects = pg.modular_effects(48, "X")
    rows = []
    for peaks, grid in ((4, 2049), (4, 4097), (6, 4097)):
        code = pg.GKPCode(0.55, 0.45, 48, peaks, grid)
        state = np.array(code.minus().amplitudes)
        rows.append(np.einsum("i,kij,j->k", state.conj(), effects, state).real)
    np.testing.assert_allclose(rows[0], rows[1], atol=1e-10)
    np.testing.assert_allclose(rows[1], rows[2], atol=1e-10)
