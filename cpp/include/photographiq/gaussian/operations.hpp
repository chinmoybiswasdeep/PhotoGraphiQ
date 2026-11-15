// Local Gaussian state updates on (mean, covariance) without forming global matrices.
//
// Conventions: see photographiq/core/types.hpp (hbar = 2, interleaved quadratures,
// statistical covariance with vacuum V = I). A mode index m owns rows/columns 2m, 2m+1.
//
// Every in-place update below costs O(n k) for a gate on k of n modes: the touched
// rows are transformed, then the touched columns, then only those rows/columns are
// re-symmetrised (untouched entries of an exactly symmetric matrix stay symmetric).
// This reproduces X V X^T + Y of the NumPy backend up to floating-point summation order.
#pragma once

#include <vector>

#include "photographiq/core/types.hpp"

namespace photographiq::gaussian {

using MeanRef = Eigen::Ref<RealVector>;
using CovRef = Eigen::Ref<RealMatrix>;
using ConstMeanRef = Eigen::Ref<const RealVector>;
using ConstCovRef = Eigen::Ref<const RealMatrix>;
using ModeList = std::vector<Index>;

// Mean and covariance returned by operations that change the number of modes.
struct Moments {
  RealVector mean;
  RealMatrix covariance;
};

// Checks the moment shapes (2n mean, 2n x 2n covariance) and returns n.
Index mode_count(ConstMeanRef mean, ConstCovRef cov);

// Interleaved quadrature indices (2m, 2m + 1) of distinct, in-range modes.
std::vector<Index> quadrature_indices(const ModeList& modes, Index num_modes);

// True when |S Omega S^T - Omega| <= atol + rtol |Omega| entrywise (numpy.allclose rule).
bool is_symplectic(const RealMatrix& matrix, Real rtol = 1e-5, Real atol = 1e-8);

// V -> S V S^T, mean -> S mean on the listed modes; throws PhysicsError if S is not
// symplectic (numpy.allclose tolerances) or has the wrong shape.
void apply_symplectic(MeanRef mean, CovRef cov, const ModeList& modes, const RealMatrix& matrix);

// General Gaussian channel on the listed modes: V -> X V X^T + Y, mean -> X mean.
// Throws PhysicsError unless Y is symmetric and Y + i Omega - i X Omega X^T >= 0
// (complete positivity) up to `tolerance` times (1 + max|Y| + max|X|^2).
void apply_channel(MeanRef mean, CovRef cov, const ModeList& modes, const RealMatrix& x,
                   const RealMatrix& y, Real tolerance = 1e-10);

// Thermal loss: V -> eta V + (1 - eta)(2 n_th + 1) I, mean -> sqrt(eta) mean on one mode.
void apply_loss(MeanRef mean, CovRef cov, Index mode, Real transmissivity, Real thermal_photons);

// Quadrature translation mean[2m] += q, mean[2m + 1] += p.
void displace(MeanRef mean, Index mode, Real q, Real p);

// Block-diagonal extension by one uncorrelated mode, appended last.
Moments append_mode(ConstMeanRef mean, ConstCovRef cov, const Eigen::Vector2d& mode_mean,
                    const Eigen::Matrix2d& mode_cov);

// Partial trace over one mode; the remaining modes keep their order.
Moments remove_mode(ConstMeanRef mean, ConstCovRef cov, Index mode);

}  // namespace photographiq::gaussian
