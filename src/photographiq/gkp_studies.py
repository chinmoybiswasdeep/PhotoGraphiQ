"""Numerical studies and calibrated ensemble decoding for finite GKP resources."""

from dataclasses import dataclass, replace

import numpy as np

from .encoded import BaseGKPDecoder, GKPCode, NearestCellDecoder, modular_effects
from .fock_measurements import wavefunctions


class SoftDecisionDecoder(BaseGKPDecoder):
    """Bayesian discrimination of a specified equal-prior preparation ensemble.

    Uses the actual Fock homodyne densities of zero/one (Z) or plus/minus (X).
    This posterior concerns the preparation label, not arbitrary entangled
    logical amplitudes or the probability of a nearest-cell decision being right.
    """

    def __init__(self, code, basis="Z", prior=(0.5, 0.5)):
        if basis not in ("X", "Z"):
            raise NotImplementedError("Soft decoder supports X/Z ensembles")
        if (
            len(prior) != 2
            or not np.isfinite(prior).all()
            or min(prior) <= 0
            or not np.isclose(sum(prior), 1, atol=1e-12, rtol=0)
        ):
            raise ValueError("Supply two positive normalized prior probabilities")
        self.code, self.basis, self.prior = code, basis, tuple(prior)
        self.vectors = np.array(
            [
                s.amplitudes
                for s in (
                    (code.zero(), code.one()) if basis == "Z" else (code.plus(), code.minus())
                )
            ]
        )

    def decode(self, outcome):
        result = NearestCellDecoder().decode(outcome)
        bra = wavefunctions(outcome, self.code.cutoff) * np.exp(
            -1j * (0 if self.basis == "Z" else np.pi / 2) * np.arange(self.code.cutoff)
        )
        likelihood = abs(self.vectors @ bra) ** 2 * np.array(self.prior)
        if likelihood.sum() <= 1e-300:
            raise ValueError("Outcome outside numerical likelihood support")
        probabilities = likelihood / likelihood.sum()
        bit = int(np.argmax(probabilities))
        return replace(
            result,
            bit=bit,
            decoder="finite-fock-ensemble-bayes",
            probabilities=tuple(probabilities),
            confidence=float(probabilities[bit]),
        )


@dataclass
class GKPMeasurementStudy:
    axis: str
    rows: list[dict]


def measurement_convergence(code, values, *, axis="cutoff", basis="Z", coefficients=(1, 0)):
    """Vary one parameter independently; report physical metrics, never auto-certify.

    Width/envelope sweeps change the physical resource, not numerical resolution.
    Residual moments integrate each cell separately to avoid grid parity bias.
    """
    from scipy.integrate import quad

    from .gkp import SPACING
    from .pattern import Pattern
    from .simulator import simulate

    if axis not in ("cutoff", "grid_points", "peaks", "peak_width", "envelope"):
        raise ValueError("Unknown convergence axis")
    values = tuple(values)
    if len(values) < 2 or any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError("Provide at least two increasing values")
    rows = []
    previous = None
    for value in values:
        current: GKPCode = replace(code, **{axis: value})
        source = current.encode(*coefficients)
        v = np.array(source.amplitudes)
        effects = modular_effects(current.cutoff, basis)
        probabilities = np.einsum("i,kij,j->k", v.conj(), effects, v).real
        phase = np.exp(-1j * (0 if basis == "Z" else np.pi / 2) * np.arange(current.cutoff))

        def density(x):
            return abs(np.dot(wavefunctions(x, current.cutoff) * phase, v)) ** 2

        count = int(np.ceil((2 * np.sqrt(current.cutoff) + 12) / SPACING))
        moments = [
            sum(
                quad(
                    lambda x: (x - cell * SPACING) ** order * density(x),
                    (cell - 0.5) * SPACING,
                    (cell + 0.5) * SPACING,
                    epsabs=1e-10,
                )[0]
                for cell in range(-count, count + 1)
            )
            for order in (1, 2)
        ]
        state = simulate(
            Pattern(inputs=(0,)), inputs={0: source}, backend="piquasso-fock", cutoff=current.cutoff
        ).state
        fidelity = None
        if previous is not None:
            length = max(len(previous), len(v))
            fidelity = float(
                abs(
                    np.vdot(
                        np.pad(previous, (0, length - len(previous))),
                        np.pad(v, (0, length - len(v))),
                    )
                )
                ** 2
            )
        rows.append(
            {
                axis: value,
                "cutoff": current.cutoff,
                "probabilities": probabilities.tolist(),
                "residual_mean": moments[0],
                "residual_second_moment": moments[1],
                "code_subspace_leakage": current.leakage(v),
                "fidelity_to_previous": fidelity,
                "stabilizers": {
                    k: [z.real, z.imag]
                    for k, z in current.diagnostics(state)["stabilizers"].items()
                },
            }
        )
        previous = v
    return GKPMeasurementStudy(axis, rows)
