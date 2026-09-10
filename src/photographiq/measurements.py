"""Measurement descriptions. All angles are radians."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Homodyne:
    """Destructive q*cos(angle)+p*sin(angle) measurement with incident-quadrature calibration.

    Args:
        angle (float): Quadrature or gate angle in radians; expressions allowed where documented.
        efficiency (float): Detector efficiency in (0,1].
        noise (float): Nonnegative added calibrated quadrature variance.

    Raises:
        ValueError: Require 0 < efficiency <= 1 and finite noise >= 0.
    """

    angle: Any = 0.0
    efficiency: float = 1.0
    noise: float = 0.0

    def __post_init__(self):
        if not 0 < self.efficiency <= 1 or not np.isfinite(self.noise) or self.noise < 0:
            raise ValueError("Require 0 < efficiency <= 1 and finite noise >= 0")

    @classmethod
    def q(cls):
        """Construct ideal position-quadrature homodyne at zero angle."""
        return cls(0.0)

    @classmethod
    def p(cls):
        """Construct ideal momentum-quadrature homodyne at pi/2."""
        return cls(np.pi / 2)


@dataclass(frozen=True)
class Heterodyne:
    """Returns (q,p) phase-space coordinates; vacuum outcome covariance 2I."""


@dataclass(frozen=True)
class Generaldyne:
    """Gaussian phase-space measurement with a physical statistical seed covariance.

    Args:
        covariance (array-like): Statistical covariance, with vacuum covariance I.
    """

    covariance: Any

    def __post_init__(self):
        from .states import GaussianState

        GaussianState(np.zeros(2), self.covariance, (0,))


@dataclass(frozen=True)
class PhotonNumber:
    """Destructive photon counting, requiring a Fock conditional-state backend."""
