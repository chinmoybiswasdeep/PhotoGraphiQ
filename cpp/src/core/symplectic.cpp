#include "photographiq/core/symplectic.hpp"

namespace photographiq {

RealMatrix symplectic_form(Index modes) {
  if (modes < 0) throw PhysicsError("Mode count must be nonnegative");
  RealMatrix omega = RealMatrix::Zero(2 * modes, 2 * modes);
  for (Index k = 0; k < modes; ++k) {
    omega(2 * k, 2 * k + 1) = 1.0;
    omega(2 * k + 1, 2 * k) = -1.0;
  }
  return omega;
}

Real symplectic_residual(const RealMatrix& matrix) {
  if (matrix.rows() != matrix.cols() || matrix.rows() % 2 != 0) {
    throw PhysicsError("A symplectic matrix must be square with even dimension");
  }
  const RealMatrix omega = symplectic_form(matrix.rows() / 2);
  return (matrix * omega * matrix.transpose() - omega).cwiseAbs().maxCoeff();
}

}  // namespace photographiq
