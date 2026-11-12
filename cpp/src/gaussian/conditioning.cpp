#include "photographiq/gaussian/conditioning.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace photographiq::gaussian {

namespace {

void check_mode(Index mode, Index modes) {
  if (mode < 0 || mode >= modes) throw PhysicsError("Mode index out of range");
}

// Old index of the i-th quadrature that survives removal of the mode starting at `first`.
inline Index kept(Index i, Index first) { return i < first ? i : i + 2; }

}  // namespace

HomodyneMoments homodyne_moments(ConstMeanRef mean, ConstCovRef cov, Index mode, Real angle,
                                 Real efficiency, Real noise) {
  check_mode(mode, mode_count(mean, cov));
  if (!(efficiency > 0.0 && efficiency <= 1.0) || !std::isfinite(noise) || noise < 0.0) {
    throw PhysicsError("Require 0 < efficiency <= 1 and finite noise >= 0");
  }
  if (!std::isfinite(angle)) throw PhysicsError("Homodyne angle must be finite");
  const Real c = std::cos(angle);
  const Real s = std::sin(angle);
  const Index m = 2 * mode;
  // Same association as u @ B @ u in NumPy: (u^T B) u.
  const Real row0 = c * cov(m, m) + s * cov(m + 1, m);
  const Real row1 = c * cov(m, m + 1) + s * cov(m + 1, m + 1);
  const Real variance = row0 * c + row1 * s + (1.0 - efficiency) / efficiency + noise;
  if (!(variance > 0.0) || !std::isfinite(variance)) {
    throw PhysicsError("Singular homodyne variance");
  }
  return {c * mean(m) + s * mean(m + 1), variance};
}

Real condition_homodyne_into(ConstMeanRef mean, ConstCovRef cov, Index mode, Real angle,
                             Real efficiency, Real noise, Real outcome, MeanRef out_mean,
                             CovRef out_cov) {
  const HomodyneMoments moments = homodyne_moments(mean, cov, mode, angle, efficiency, noise);
  if (!std::isfinite(outcome)) {
    throw PhysicsError("Homodyne outcome must be a finite real scalar");
  }
  const Index dim = mean.size() - 2;
  if (out_mean.size() != dim || out_cov.rows() != dim || out_cov.cols() != dim) {
    throw PhysicsError("Output moments must have the post-measurement dimension");
  }
  const Index m = 2 * mode;
  const Real c = std::cos(angle);
  const Real s = std::sin(angle);
  RealVector cross(dim);
  for (Index i = 0; i < dim; ++i) cross(i) = cov(kept(i, m), m) * c + cov(kept(i, m), m + 1) * s;
  const Real shift = (outcome - moments.expected) / moments.variance;
  for (Index j = 0; j < dim; ++j) {
    out_mean(j) = mean(kept(j, m)) + cross(j) * shift;
    for (Index i = 0; i < dim; ++i) {
      out_cov(i, j) = cov(kept(i, m), kept(j, m)) - (cross(i) * cross(j)) / moments.variance;
    }
  }
  const Real sigma = std::sqrt(moments.variance);
  const Real z = (outcome - moments.expected) / sigma;
  return std::exp(-z * z / 2.0) / std::sqrt(2.0 * std::numbers::pi) / sigma;
}

Conditioned condition_homodyne(ConstMeanRef mean, ConstCovRef cov, Index mode, Real angle,
                               Real efficiency, Real noise, Real outcome) {
  const Index dim = std::max<Index>(mean.size() - 2, 0);
  Conditioned out{RealVector(dim), RealMatrix(dim, dim), 0.0};
  out.density = condition_homodyne_into(mean, cov, mode, angle, efficiency, noise, outcome,
                                        out.mean, out.covariance);
  return out;
}

PhaseSpaceMoments generaldyne_moments(ConstMeanRef mean, ConstCovRef cov, Index mode,
                                      const Eigen::Matrix2d& detector) {
  check_mode(mode, mode_count(mean, cov));
  if (!detector.allFinite() || std::abs(detector(0, 1) - detector(1, 0)) > 1e-10) {
    throw PhysicsError("Detector covariance must be finite and symmetric");
  }
  const Index m = 2 * mode;
  PhaseSpaceMoments out{mean.segment<2>(m), cov.block<2, 2>(m, m) + detector};
  if (Eigen::LLT<Eigen::Matrix2d>(out.covariance).info() != Eigen::Success) {
    throw PhysicsError("Phase-space outcome covariance must be positive definite");
  }
  return out;
}

Conditioned condition_generaldyne(ConstMeanRef mean, ConstCovRef cov, Index mode,
                                  const Eigen::Matrix2d& detector,
                                  const Eigen::Vector2d& outcome) {
  const PhaseSpaceMoments moments = generaldyne_moments(mean, cov, mode, detector);
  if (!outcome.allFinite()) {
    throw PhysicsError("Phase-space outcome must be a finite real pair (q, p)");
  }
  const Index dim = mean.size() - 2;
  const Index m = 2 * mode;
  RealMatrix cross_t(2, dim);  // C^T, C = V(remaining, measured)
  RealVector rest(dim);
  for (Index i = 0; i < dim; ++i) {
    cross_t(0, i) = cov(kept(i, m), m);
    cross_t(1, i) = cov(kept(i, m), m + 1);
    rest(i) = mean(kept(i, m));
  }
  // gain^T = (B + D)^-1 C^T, as numpy.linalg.solve(total, c.T).
  const RealMatrix gain_t = moments.covariance.partialPivLu().solve(cross_t);
  const Eigen::Vector2d residual = outcome - moments.expected;
  Conditioned out{rest + gain_t.transpose() * residual, RealMatrix(dim, dim), 0.0};
  const RealMatrix update = gain_t.transpose() * cross_t;
  for (Index j = 0; j < dim; ++j) {
    for (Index i = 0; i < dim; ++i) {
      out.covariance(i, j) =
          0.5 * ((cov(kept(i, m), kept(j, m)) - update(i, j)) +
                 (cov(kept(j, m), kept(i, m)) - update(j, i)));
    }
  }
  const Eigen::LLT<Eigen::Matrix2d> llt(moments.covariance);
  const Eigen::Vector2d white = llt.matrixL().solve(residual);
  const Eigen::Matrix2d lower = llt.matrixL();
  const Real log_det = 2.0 * (std::log(lower(0, 0)) + std::log(lower(1, 1)));
  out.density =
      std::exp(-0.5 * (2.0 * std::log(2.0 * std::numbers::pi) + log_det + white.squaredNorm()));
  return out;
}

}  // namespace photographiq::gaussian
