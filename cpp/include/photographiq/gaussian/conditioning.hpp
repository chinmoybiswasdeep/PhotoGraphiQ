// Exact destructive Gaussian measurements by Schur complement, for a GIVEN outcome.
//
// Sampling is deliberately not done here: callers draw outcomes from the returned
// moments (Python uses the same numpy Generator calls as GaussianBackend), then condition.
//
// Homodyne(theta, eta, nu) records x = q cos(theta) + p sin(theta) calibrated to the
// incident quadrature: x ~ N(u.mu, u^T B u + (1 - eta)/eta + nu), u = (cos, sin).
// Generaldyne with statistical seed covariance D records (q, p) ~ N(mu, B + D);
// heterodyne is D = I. B, mu are the measured mode's covariance block and mean.
#pragma once

#include "photographiq/gaussian/operations.hpp"

namespace photographiq::gaussian {

struct HomodyneMoments {
  Real expected;
  Real variance;  // outcome variance including detector noise
};

struct PhaseSpaceMoments {
  Eigen::Vector2d expected;
  Eigen::Matrix2d covariance;  // B + D
};

struct Conditioned {
  RealVector mean;
  RealMatrix covariance;
  Real density;  // Gaussian density of the given outcome in calibrated outcome units
};

// Throws PhysicsError for an invalid detector or a nonpositive/nonfinite variance
// ("Singular homodyne variance", as GaussianBackend).
HomodyneMoments homodyne_moments(ConstMeanRef mean, ConstCovRef cov, Index mode, Real angle,
                                 Real efficiency, Real noise);

// Conditional moments of the remaining modes written into out_mean (2n-2) and out_cov
// ((2n-2)^2); returns the outcome density. Core of the batch API (O(n) scratch only).
Real condition_homodyne_into(ConstMeanRef mean, ConstCovRef cov, Index mode, Real angle,
                             Real efficiency, Real noise, Real outcome, MeanRef out_mean,
                             CovRef out_cov);

Conditioned condition_homodyne(ConstMeanRef mean, ConstCovRef cov, Index mode, Real angle,
                               Real efficiency, Real noise, Real outcome);

// Throws PhysicsError unless the detector covariance is symmetric and B + D is positive
// definite (Cholesky succeeds).
PhaseSpaceMoments generaldyne_moments(ConstMeanRef mean, ConstCovRef cov, Index mode,
                                      const Eigen::Matrix2d& detector);

Conditioned condition_generaldyne(ConstMeanRef mean, ConstCovRef cov, Index mode,
                                  const Eigen::Matrix2d& detector, const Eigen::Vector2d& outcome);

}  // namespace photographiq::gaussian
