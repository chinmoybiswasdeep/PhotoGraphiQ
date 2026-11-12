"""Compatibility shim: CUDA-Q lives in :mod:`photographiq.cudaq_dynamics` (an adapter).

CUDA-Q is not a pattern backend: the former ``cudaq-dynamics`` registry entry, whose
construction always raised, was removed. Use the Hamiltonian-level adapter instead.
"""

from ..cudaq_dynamics import cudaq_capabilities, cudaq_probe

__all__ = ["cudaq_capabilities", "cudaq_probe"]
