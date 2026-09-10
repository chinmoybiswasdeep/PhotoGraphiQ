"""Finite-energy resources constructed at the execution cutoff."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CubicPhaseResource:
    """CP(gamma) S(-r)|0>, with CP=exp(i gamma q^3/6), [q,p]=2i."""

    gamma: float
    squeezing: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.gamma) or not np.isfinite(self.squeezing) or self.squeezing < 0:
            raise ValueError("Resource parameters must be finite; squeezing >= 0")


@dataclass(frozen=True)
class CatResource:
    """Normalized finite-Fock projection of |alpha> + parity |-alpha>."""

    alpha: complex
    parity: int = 1

    def __post_init__(self):
        if not np.isfinite(self.alpha) or self.parity not in (-1, 1):
            raise ValueError("Invalid cat parameters")
        if self.alpha == 0 and self.parity == -1:
            raise ValueError("Odd cat at alpha=0 has zero norm")
