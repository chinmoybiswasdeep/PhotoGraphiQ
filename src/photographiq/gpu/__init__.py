"""Optional CUDA acceleration of batched Gaussian and Fock workloads through CuPy.

Importing this package does not import CuPy. Kernels default to ``device="gpu"`` and raise if
CuPy or a CUDA device is unavailable; ``device="cpu"`` runs the same algorithm in NumPy.
Every result carries an :class:`Execution` record naming the device that produced it.
See ``docs/gpu.md``.
"""

from .device import INSTALL_HINT, Execution, available, device_info, require
from .fock import FockBatch, apply_operator, apply_phase
from .gaussian import (
    BatchValues,
    GaussianBatch,
    HomodyneConditioning,
    HomodyneMoments,
    apply_symplectic,
    condition_homodyne,
    fidelity,
    homodyne_log_density,
    homodyne_moments,
    purity,
    symplectic_eigenvalues,
)

__all__ = [
    "INSTALL_HINT",
    "BatchValues",
    "Execution",
    "FockBatch",
    "GaussianBatch",
    "HomodyneConditioning",
    "HomodyneMoments",
    "apply_operator",
    "apply_phase",
    "apply_symplectic",
    "available",
    "condition_homodyne",
    "device_info",
    "fidelity",
    "homodyne_log_density",
    "homodyne_moments",
    "purity",
    "require",
    "symplectic_eigenvalues",
]
