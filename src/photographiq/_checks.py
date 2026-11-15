"""Small shared argument checks."""

from __future__ import annotations

from typing import Any

import numpy as np


def is_finite_real_scalar(value: object) -> bool:
    """True for a finite real scalar (Python or NumPy; complex with zero imaginary part counts).

    Same semantics as ``np.isscalar(v) and np.isreal(v) and np.isfinite(v)``. NumPy >= 2.5 stubs
    narrow ``np.isscalar`` to a union that ``np.isreal`` does not accept, hence the ``Any``.
    """
    if not np.isscalar(value):
        return False
    scalar: Any = value
    return bool(np.isreal(scalar) and np.isfinite(scalar))
