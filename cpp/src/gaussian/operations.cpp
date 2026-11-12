#include "photographiq/gaussian/operations.hpp"

#include <cmath>
#include <set>

#include "photographiq/core/symplectic.hpp"

namespace photographiq::gaussian {

namespace {

// V(:, P) is read column-wise (contiguous) and, because V is exactly symmetric, also
// stands for V(P, :)^T. W = V(:, P) X^T gives the new off-block entries; the (P, P)
// block becomes X W(P, :) + Y and is the only block that needs re-symmetrising.
void transform_local(MeanRef mean, CovRef cov, const std::vector<Index>& idx,
                     const RealMatrix& x, const RealMatrix* y) {
  const Index dim = cov.rows();
  const Index k = static_cast<Index>(idx.size());
  RealVector local(k);
  for (Index a = 0; a < k; ++a) local(a) = mean(idx[a]);
  const RealVector moved = x * local;
  for (Index a = 0; a < k; ++a) mean(idx[a]) = moved(a);

  RealMatrix columns(dim, k);
  for (Index a = 0; a < k; ++a) columns.col(a) = cov.col(idx[a]);
  const RealMatrix w = columns * x.transpose();
  RealMatrix block(k, k);
  for (Index a = 0; a < k; ++a) block.row(a) = w.row(idx[a]);
  RealMatrix inner = x * block;
  if (y != nullptr) inner += *y;
  const RealMatrix symmetric = 0.5 * (inner + inner.transpose());

  for (Index a = 0; a < k; ++a) {
    cov.col(idx[a]) = w.col(a);
    cov.row(idx[a]) = w.col(a).transpose();
  }
  for (Index a = 0; a < k; ++a) {
    for (Index b = 0; b < k; ++b) cov(idx[a], idx[b]) = symmetric(a, b);
  }
}

void check_mode(Index mode, Index modes) {
  if (mode < 0 || mode >= modes) throw PhysicsError("Mode index out of range");
}

}  // namespace

Index mode_count(ConstMeanRef mean, ConstCovRef cov) {
  const Index dim = mean.size();
  if (dim % 2 != 0 || cov.rows() != dim || cov.cols() != dim) {
    throw PhysicsError("Invalid Gaussian moment dimensions");
  }
  return dim / 2;
}

std::vector<Index> quadrature_indices(const ModeList& modes, Index num_modes) {
  std::set<Index> seen;
  std::vector<Index> indices;
  indices.reserve(2 * modes.size());
  for (const Index mode : modes) {
    check_mode(mode, num_modes);
    if (!seen.insert(mode).second) throw PhysicsError("Repeated gate modes");
    indices.push_back(2 * mode);
    indices.push_back(2 * mode + 1);
  }
  return indices;
}

bool is_symplectic(const RealMatrix& matrix, Real rtol, Real atol) {
  if (matrix.rows() != matrix.cols() || matrix.rows() % 2 != 0 || !matrix.allFinite()) {
    return false;
  }
  const RealMatrix omega = symplectic_form(matrix.rows() / 2);
  const RealMatrix residual = matrix * omega * matrix.transpose() - omega;
  return (residual.array().abs() <= atol + rtol * omega.array().abs()).all();
}

void apply_symplectic(MeanRef mean, CovRef cov, const ModeList& modes, const RealMatrix& matrix) {
  const std::vector<Index> idx = quadrature_indices(modes, mode_count(mean, cov));
  const Index k = static_cast<Index>(idx.size());
  if (matrix.rows() != k || matrix.cols() != k || !is_symplectic(matrix)) {
    throw PhysicsError("Transformation must be symplectic");
  }
  transform_local(mean, cov, idx, matrix, nullptr);
}

void apply_channel(MeanRef mean, CovRef cov, const ModeList& modes, const RealMatrix& x,
                   const RealMatrix& y, Real tolerance) {
  const std::vector<Index> idx = quadrature_indices(modes, mode_count(mean, cov));
  const Index k = static_cast<Index>(idx.size());
  if (x.rows() != k || x.cols() != k || y.rows() != k || y.cols() != k || !x.allFinite() ||
      !y.allFinite()) {
    throw PhysicsError("Channel matrices must be finite with shape (2k, 2k)");
  }
  if (k == 0) return;  // identity channel; maxCoeff/eigensolver are undefined on empty input
  const Real scale = 1.0 + y.cwiseAbs().maxCoeff() + x.cwiseAbs2().maxCoeff();
  if ((y - y.transpose()).cwiseAbs().maxCoeff() > tolerance * scale) {
    throw PhysicsError("Channel noise matrix Y must be symmetric");
  }
  const RealMatrix omega = symplectic_form(k / 2);
  const ComplexMatrix condition =
      y.cast<Complex>() + Complex(0.0, 1.0) * (omega - x * omega * x.transpose()).cast<Complex>();
  const Eigen::SelfAdjointEigenSolver<ComplexMatrix> solver(condition, Eigen::EigenvaluesOnly);
  if (solver.eigenvalues().minCoeff() < -tolerance * scale) {
    throw PhysicsError("Channel is not completely positive: Y + i Omega - i X Omega X^T < 0");
  }
  transform_local(mean, cov, idx, x, &y);
}

void apply_loss(MeanRef mean, CovRef cov, Index mode, Real transmissivity, Real thermal_photons) {
  if (!(transmissivity >= 0.0 && transmissivity <= 1.0) || !std::isfinite(thermal_photons) ||
      thermal_photons < 0.0) {
    throw PhysicsError("Invalid loss parameters");
  }
  const std::vector<Index> idx = quadrature_indices({mode}, mode_count(mean, cov));
  const RealMatrix x = std::sqrt(transmissivity) * RealMatrix::Identity(2, 2);
  const RealMatrix y =
      (1.0 - transmissivity) * (2.0 * thermal_photons + 1.0) * RealMatrix::Identity(2, 2);
  transform_local(mean, cov, idx, x, &y);
}

void displace(MeanRef mean, Index mode, Real q, Real p) {
  if (mean.size() % 2 != 0) throw PhysicsError("Invalid Gaussian moment dimensions");
  check_mode(mode, mean.size() / 2);
  if (!std::isfinite(q) || !std::isfinite(p)) throw PhysicsError("Nonfinite displacement");
  mean(2 * mode) += q;
  mean(2 * mode + 1) += p;
}

Moments append_mode(ConstMeanRef mean, ConstCovRef cov, const Eigen::Vector2d& mode_mean,
                    const Eigen::Matrix2d& mode_cov) {
  const Index dim = 2 * mode_count(mean, cov);
  if (!mode_mean.allFinite() || !mode_cov.allFinite()) {
    throw PhysicsError("Moments must be finite");
  }
  Moments out{RealVector(dim + 2), RealMatrix::Zero(dim + 2, dim + 2)};
  out.mean.head(dim) = mean;
  out.mean.tail<2>() = mode_mean;
  out.covariance.topLeftCorner(dim, dim) = cov;
  out.covariance.bottomRightCorner<2, 2>() = mode_cov;
  return out;
}

Moments remove_mode(ConstMeanRef mean, ConstCovRef cov, Index mode) {
  const Index modes = mode_count(mean, cov);
  check_mode(mode, modes);
  const Index dim = 2 * modes - 2;
  const Index first = 2 * mode;
  auto kept = [first](Index i) { return i < first ? i : i + 2; };
  Moments out{RealVector(dim), RealMatrix(dim, dim)};
  for (Index j = 0; j < dim; ++j) {
    out.mean(j) = mean(kept(j));
    for (Index i = 0; i < dim; ++i) out.covariance(i, j) = cov(kept(i), kept(j));
  }
  return out;
}

}  // namespace photographiq::gaussian
