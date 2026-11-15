// Symplectic-form helpers in the interleaved (q0, p0, q1, p1, ...) ordering.
#pragma once

#include "photographiq/core/types.hpp"

namespace photographiq {

// Omega = (+)_modes [[0, 1], [-1, 0]].
RealMatrix symplectic_form(Index modes);

// max |S Omega S^T - Omega|, the symplecticity residual of a 2n x 2n matrix.
Real symplectic_residual(const RealMatrix& matrix);

}  // namespace photographiq
