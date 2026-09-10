"""Ideal conditional homodyne in a finite Fock expansion, hbar=2.

Piquasso does not return a conditional homodyne state. This adapter contracts
its amplitudes with Hermite wavefunctions. Sampling integrates the resulting
nonnegative density; tolerances control numerical inversion, not Hilbert cutoff.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def wavefunctions(x: float, cutoff: int) -> np.ndarray:
    values = np.empty(cutoff)
    values[0] = (2 * np.pi) ** (-0.25) * np.exp(-x * x / 4)
    if cutoff > 1:
        values[1] = x * values[0]
    for n in range(2, cutoff):
        values[n] = (x * values[n - 1] - np.sqrt(n - 1) * values[n - 2]) / np.sqrt(n)
    return values


def homodyne_projection(amplitudes, mode, cutoff, angle, rng, outcome=None):
    groups = {}
    for basis, amplitude in amplitudes.items():
        survivor = basis[:mode] + basis[mode + 1 :]
        if survivor not in groups:
            groups[survivor] = np.zeros(cutoff, dtype=complex)
        groups[survivor][basis[mode]] = amplitude
    basis = tuple(groups)
    matrix = np.array(list(groups.values())) * np.exp(-1j * angle * np.arange(cutoff))

    def vector(x):
        return matrix @ wavefunctions(x, cutoff)

    def density(x):
        v = vector(x)
        return float(np.vdot(v, v).real)

    if outcome is None:
        # Beyond the classical turning point, add a Gaussian-tail margin.
        bound = 2 * np.sqrt(cutoff) + 10
        mass, error = quad(density, -bound, bound, epsabs=1e-10, epsrel=1e-10, limit=300)
        if abs(mass - 1) > 1e-7 or error > 1e-7:
            raise ArithmeticError("Homodyne integration did not resolve the normalized density")
        target = float(rng.random()) * mass

        def residual(x):
            mass_to_x, error_to_x = quad(density, -bound, x, epsabs=1e-10, epsrel=1e-10, limit=300)
            if not np.isfinite(mass_to_x) or not np.isfinite(error_to_x) or error_to_x > 1e-7:
                raise ArithmeticError("Homodyne partial-CDF integration failed its error budget")
            return mass_to_x - target

        outcome = brentq(residual, -bound, bound, xtol=1e-10)
    if not isinstance(outcome, (int, float, np.integer, np.floating)) or not np.isfinite(outcome):
        raise ValueError("Homodyne outcome must be a finite real scalar")
    outcome = float(outcome)
    probability_density = density(outcome)
    if probability_density <= 1e-300:
        raise ValueError("Homodyne postselection has zero numerical density")
    projected = vector(outcome) / np.sqrt(probability_density)
    return outcome, dict(zip(basis, projected, strict=True)), probability_density
