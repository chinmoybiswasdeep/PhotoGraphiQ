// Shared numerical types and the conventions every native engine follows.
//
// Conventions (identical to the Python package, see docs/getting-started/conventions.md):
//   * hbar = 2, so [q, p] = 2i, q = a + a^dagger, p = -i (a - a^dagger).
//   * Quadratures are interleaved per mode: (q0, p0, q1, p1, ...).
//   * Covariance matrices are statistical: V_ij = <{dx_i, dx_j}>/2, vacuum V = I.
//   * Symplectic form Omega = (+) [[0, 1], [-1, 0]]; S Omega S^T = Omega.
//   * Displacement vectors are quadrature means; a coherent state alpha has
//     mean (2 Re alpha, 2 Im alpha).
//   * Fock amplitudes are complex double; matrices are column-major Eigen types.
#pragma once

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

namespace photographiq {

using Real = double;
using Complex = std::complex<double>;
using Index = Eigen::Index;
using RealVector = Eigen::VectorXd;
using RealMatrix = Eigen::MatrixXd;
using ComplexVector = Eigen::VectorXcd;
using ComplexMatrix = Eigen::MatrixXcd;

inline constexpr Real kHbar = 2.0;

// Raised for physically invalid input (unphysical covariance, bad index, ...).
class PhysicsError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

}  // namespace photographiq
