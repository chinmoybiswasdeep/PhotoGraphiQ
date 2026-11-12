// Batched kernels over B independent states, parallelised with OpenMP over the batch.
//
// Layout (row-major, matching C-contiguous NumPy arrays reshaped to (B, -1)):
//   means       B x 2n
//   covariances B x (2n)^2, each row one flattened covariance. Each state is replaced by
//               its symmetric part (V + V^T)/2 inside the parallel region (in place for
//               gates, a local copy for conditioning), so row- and column-major flattening
//               coincide. This is exact: sym(S V S^T) = S sym(V) S^T, and the NumPy backend
//               stores (V + V^T)/2 after every update.
//   matrices    B x (2k)^2, each row one row-major 2k x 2k matrix.
// All inputs are validated serially before the parallel region; a failure inside it is
// rethrown after the loop.
#pragma once

#include "photographiq/gaussian/conditioning.hpp"

namespace photographiq::gaussian {

using RowMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using BatchRef = Eigen::Ref<RowMatrix>;
using ConstBatchRef = Eigen::Ref<const RowMatrix>;

// In place: per-state symplectic S_b on the same listed modes (covariances symmetrised).
void batch_apply_symplectic(BatchRef means, BatchRef covariances, const ModeList& modes,
                            ConstBatchRef matrices);

// Per-state homodyne moments (angle_b, shared detector); outputs have length B.
void batch_homodyne_moments(ConstBatchRef means, ConstBatchRef covariances, Index mode,
                            const RealVector& angles, Real efficiency, Real noise,
                            Eigen::Ref<RealVector> expected, Eigen::Ref<RealVector> variance);

// Per-state conditioning on given outcomes; out_* are B x (2n-2) and B x (2n-2)^2.
void batch_condition_homodyne(ConstBatchRef means, ConstBatchRef covariances, Index mode,
                              const RealVector& angles, Real efficiency, Real noise,
                              const RealVector& outcomes, BatchRef out_means, BatchRef out_covs,
                              Eigen::Ref<RealVector> densities);

// Number of OpenMP threads a parallel region would use (1 without OpenMP).
int max_threads();

}  // namespace photographiq::gaussian
