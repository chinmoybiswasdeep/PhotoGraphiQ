#include "photographiq/gaussian/diagnostics.hpp"

#include <algorithm>
#include <cmath>

#include "photographiq/core/symplectic.hpp"

namespace photographiq::gaussian {

namespace {

Index square_modes(ConstCovRef cov) {
  if (cov.rows() != cov.cols() || cov.rows() % 2 != 0) {
    throw PhysicsError("Covariance must be square with even dimension");
  }
  return cov.rows() / 2;
}

// log|det| and sign of a general square matrix by partial-pivot LU.
std::pair<Real, Real> sign_log_det(const RealMatrix& matrix) {
  const Eigen::PartialPivLU<RealMatrix> lu(matrix);
  Real sign = static_cast<Real>(lu.permutationP().determinant());
  Real log_abs = 0.0;
  for (Index i = 0; i < matrix.rows(); ++i) {
    const Real u = lu.matrixLU()(i, i);
    sign *= u < 0.0 ? -1.0 : (u > 0.0 ? 1.0 : 0.0);
    log_abs += std::log(std::abs(u));
  }
  return {sign, log_abs};
}

Real xlogx(Real x) { return x > 0.0 ? x * std::log(x) : 0.0; }

// d^T S^-1 d and log det S of a positive definite S.
std::pair<Real, Real> gaussian_exponent(const RealVector& delta, const RealMatrix& sum) {
  const Eigen::LLT<RealMatrix> llt(sum);
  if (llt.info() != Eigen::Success) {
    throw PhysicsError("Sum of covariances must be positive definite");
  }
  const RealVector white = llt.matrixL().solve(delta);
  const RealMatrix lower = llt.matrixL();
  return {white.squaredNorm(), 2.0 * lower.diagonal().array().log().sum()};
}

void check_pair(ConstMeanRef mean1, ConstCovRef cov1, ConstMeanRef mean2, ConstCovRef cov2) {
  if (mode_count(mean1, cov1) != mode_count(mean2, cov2)) {
    throw PhysicsError("States must have the same number of modes");
  }
}

bool is_pure(ConstCovRef cov, Real tolerance) {
  const RealVector nu = symplectic_eigenvalues(cov);
  return ((nu.array() - 1.0).abs() < tolerance).all();
}

}  // namespace

RealVector symplectic_eigenvalues(ConstCovRef cov) {
  const Index n = square_modes(cov);
  if (n == 0) return RealVector(0);  // Eigen's eigensolvers are undefined on 0 x 0 input
  const RealMatrix omega = symplectic_form(n);
  RealVector values(2 * n);
  const Eigen::LLT<RealMatrix> llt(cov);
  if (llt.info() == Eigen::Success) {
    const RealMatrix lower = llt.matrixL();
    const RealMatrix antisymmetric = lower.transpose() * omega * lower;
    const ComplexMatrix hermitian = Complex(0.0, 1.0) * antisymmetric.cast<Complex>();
    const Eigen::SelfAdjointEigenSolver<ComplexMatrix> solver(hermitian, Eigen::EigenvaluesOnly);
    values = solver.eigenvalues().cwiseAbs();
  } else {  // not positive definite: unphysical, keep the general path as states.py does
    const Eigen::EigenSolver<RealMatrix> solver(omega * RealMatrix(cov), false);
    values = solver.eigenvalues().cwiseAbs();
  }
  std::sort(values.data(), values.data() + values.size());
  RealVector nu(n);
  for (Index k = 0; k < n; ++k) nu(k) = values(2 * k);
  return nu;
}

Real purity(ConstCovRef cov) {
  square_modes(cov);
  return std::exp(-0.5 * sign_log_det(cov).second);
}

Real entropy(ConstCovRef cov, Real tolerance) {
  const RealVector nu = symplectic_eigenvalues(cov);
  if (nu.size() > 0 && nu.minCoeff() < 1.0 - tolerance) {
    throw PhysicsError("Covariance has a symplectic eigenvalue below 1");
  }
  Real total = 0.0;
  for (Index k = 0; k < nu.size(); ++k) {
    const Real v = std::max(nu(k), 1.0);
    total += xlogx((v + 1.0) / 2.0) - xlogx((v - 1.0) / 2.0);
  }
  return total;
}

Real log_negativity(ConstCovRef cov, const ModeList& modes) {
  const Index n = square_modes(cov);
  quadrature_indices(modes, n);  // validates unique, in-range labels
  RealVector flip = RealVector::Ones(2 * n);
  for (const Index mode : modes) flip(2 * mode + 1) = -1.0;
  const RealMatrix transposed = flip.asDiagonal() * RealMatrix(cov) * flip.asDiagonal();
  const RealVector nu = symplectic_eigenvalues(transposed);
  Real total = 0.0;
  for (Index k = 0; k < n; ++k) total += std::max(0.0, -std::log2(nu(k)));
  return total;
}

Real overlap(ConstMeanRef mean1, ConstCovRef cov1, ConstMeanRef mean2, ConstCovRef cov2) {
  check_pair(mean1, cov1, mean2, cov2);
  const Index n = mean1.size() / 2;
  const auto [exponent, log_det] = gaussian_exponent(mean1 - mean2, cov1 + cov2);
  return std::exp(static_cast<Real>(n) * std::log(2.0) - 0.5 * exponent - 0.5 * log_det);
}

Real fidelity(ConstMeanRef mean1, ConstCovRef cov1, ConstMeanRef mean2, ConstCovRef cov2,
              Real pure_tolerance) {
  check_pair(mean1, cov1, mean2, cov2);
  if (is_pure(cov1, pure_tolerance) || is_pure(cov2, pure_tolerance)) {
    return overlap(mean1, cov1, mean2, cov2);
  }
  const Index n = mean1.size() / 2;
  const RealMatrix omega = symplectic_form(n);
  // BBP units: vacuum covariance I/2.
  const RealMatrix va = 0.5 * cov1;
  const RealMatrix vb = 0.5 * cov2;
  const RealMatrix sum = va + vb;
  const RealMatrix rhs = 0.25 * omega + vb * omega * va;
  const RealMatrix aux = omega.transpose() * sum.partialPivLu().solve(rhs);
  const Eigen::EigenSolver<RealMatrix> solver(aux * omega, false);
  Complex product(1.0, 0.0);
  for (Index j = 0; j < 2 * n; ++j) {
    const Complex mu = solver.eigenvalues()(j);
    product *= 1.0 + std::sqrt(1.0 + 1.0 / (4.0 * mu * mu));
  }
  const auto [sign_aux, log_aux] = sign_log_det(2.0 * aux);
  const auto [sign_sum, log_sum] = sign_log_det(sum);
  const Real ratio = product.real() * sign_aux * sign_sum * std::exp(log_aux - log_sum);
  if (!(ratio >= 0.0)) throw PhysicsError("Fidelity determinant ratio is not positive");
  const auto [exponent, unused] = gaussian_exponent(mean1 - mean2, cov1 + cov2);
  (void)unused;
  return std::sqrt(ratio) * std::exp(-0.5 * exponent);
}

}  // namespace photographiq::gaussian
