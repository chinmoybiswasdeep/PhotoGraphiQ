"""Measurement calculus, local Cliffords and ZX diagrams (pure Python, small exact models).

Submodules:

* :mod:`.qubit` -- DKP qubit measurement calculus: commands N/E/M/X/Z/S, ``QubitPattern``,
  standardization, signal shifting, flow-to-pattern, exact branch simulation/equivalence.
* :mod:`.clifford` -- 24-element single-qubit Clifford group, local complementation, LC
  equivalence of labelled graph states.
* :mod:`.zx` -- scalar-exact ZX diagrams, tensor semantics and a sound rewrite subset.
* :mod:`.cv` -- command-level view of CV patterns and ideal-limit Gaussian identity checks
  (no CV graphical calculus).
* :mod:`.gaussian_graph` -- Gaussian pure states as complex-weighted graphs Z = V + iU
  (Menicucci-Flammia-van Loock 2011): Gaussian unitaries and q/p homodyne graph rules.
"""

from . import clifford, cv, gaussian_graph, qubit, zx
from .clifford import Clifford, LCEquivalence, lc_equivalence, local_complement
from .gaussian_graph import GaussianGraph
from .qubit import (
    QubitPattern,
    equivalent,
    is_deterministic,
    pattern_from_flow,
    signal_shift,
    standardize,
)
from .zx import Diagram

__all__ = [
    "Clifford",
    "Diagram",
    "GaussianGraph",
    "LCEquivalence",
    "QubitPattern",
    "clifford",
    "cv",
    "equivalent",
    "gaussian_graph",
    "is_deterministic",
    "lc_equivalence",
    "local_complement",
    "pattern_from_flow",
    "qubit",
    "signal_shift",
    "standardize",
    "zx",
]
