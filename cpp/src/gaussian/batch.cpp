#include "photographiq/gaussian/batch.hpp"

#include <exception>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace photographiq::gaussian {

namespace {

using RowMatrixMap = Eigen::Map<const RowMatrix>;

// Runs body(b) for b in [0, count) in parallel; the first exception is rethrown afterwards
// (exceptions must not escape an OpenMP region).
template <class Body>
void parallel_for(Index count, const Body& body) {
  std::exception_ptr error = nullptr;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (Index b = 0; b < count; ++b) {
    try {
      body(b);
    } catch (...) {
#ifdef _OPENMP
#pragma omp critical(pgq_gaussian_batch_error)
#endif
      {
        if (!error) error = std::current_exception();
      }
    }
  }
  if (error) std::rethrow_exception(error);
}

// Validates batch shapes and returns the per-state quadrature dimension 2n.
Index batch_dim(const auto& means, const auto& covariances) {
  const Index dim = means.cols();
  if (dim % 2 != 0 || covariances.rows() != means.rows() || covariances.cols() != dim * dim) {
    throw PhysicsError("Batch shapes must be (B, 2n) means and (B, 2n, 2n) covariances");
  }
  return dim;
}

// In-place (V + V^T)/2; a no-op (bitwise) on exactly symmetric input.
void symmetrise(Eigen::Map<RealMatrix>& cov) {
  for (Index j = 0; j < cov.cols(); ++j) {
    for (Index i = j + 1; i < cov.rows(); ++i) {
      const Real value = 0.5 * (cov(i, j) + cov(j, i));
      cov(i, j) = value;
      cov(j, i) = value;
    }
  }
}

void check_vector(const RealVector& values, Index batch, const char* name) {
  if (values.size() != batch) throw PhysicsError(std::string(name) + " must have length B");
}

}  // namespace

int max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

void batch_apply_symplectic(BatchRef means, BatchRef covariances, const ModeList& modes,
                            ConstBatchRef matrices) {
  const Index dim = batch_dim(means, covariances);
  const Index k = 2 * static_cast<Index>(modes.size());
  const std::vector<Index> idx = quadrature_indices(modes, dim / 2);
  if (matrices.rows() != means.rows() || matrices.cols() != k * k) {
    throw PhysicsError("Batch matrices must have shape (B, 2k, 2k)");
  }
  std::vector<RealMatrix> local(static_cast<std::size_t>(means.rows()));
  for (Index b = 0; b < means.rows(); ++b) {
    local[static_cast<std::size_t>(b)] = RowMatrixMap(matrices.row(b).data(), k, k);
    if (!is_symplectic(local[static_cast<std::size_t>(b)])) {
      throw PhysicsError("Transformation must be symplectic");
    }
  }
  parallel_for(means.rows(), [&](Index b) {
    Eigen::Map<RealVector> mean(means.row(b).data(), dim);
    Eigen::Map<RealMatrix> cov(covariances.row(b).data(), dim, dim);
    symmetrise(cov);
    apply_symplectic(mean, cov, modes, local[static_cast<std::size_t>(b)]);
  });
}

void batch_homodyne_moments(ConstBatchRef means, ConstBatchRef covariances, Index mode,
                            const RealVector& angles, Real efficiency, Real noise,
                            Eigen::Ref<RealVector> expected, Eigen::Ref<RealVector> variance) {
  const Index dim = batch_dim(means, covariances);
  const Index batch = means.rows();
  check_vector(angles, batch, "angles");
  if (expected.size() != batch || variance.size() != batch) {
    throw PhysicsError("Outputs must have length B");
  }
  parallel_for(batch, [&](Index b) {
    const Eigen::Map<const RealVector> mean(means.row(b).data(), dim);
    const Eigen::Map<const RealMatrix> cov(covariances.row(b).data(), dim, dim);
    const HomodyneMoments m = homodyne_moments(mean, cov, mode, angles(b), efficiency, noise);
    expected(b) = m.expected;
    variance(b) = m.variance;
  });
}

void batch_condition_homodyne(ConstBatchRef means, ConstBatchRef covariances, Index mode,
                              const RealVector& angles, Real efficiency, Real noise,
                              const RealVector& outcomes, BatchRef out_means, BatchRef out_covs,
                              Eigen::Ref<RealVector> densities) {
  const Index dim = batch_dim(means, covariances);
  const Index batch = means.rows();
  const Index out_dim = dim - 2;
  check_vector(angles, batch, "angles");
  check_vector(outcomes, batch, "outcomes");
  if (out_means.rows() != batch || out_means.cols() != out_dim || out_covs.rows() != batch ||
      out_covs.cols() != out_dim * out_dim || densities.size() != batch) {
    throw PhysicsError("Output batch shapes must match the post-measurement dimension");
  }
  parallel_for(batch, [&](Index b) {
    const Eigen::Map<const RealVector> mean(means.row(b).data(), dim);
    const Eigen::Map<const RealMatrix> stored(covariances.row(b).data(), dim, dim);
    const RealMatrix cov = 0.5 * (stored + stored.transpose());
    Eigen::Map<RealVector> out_mean(out_means.row(b).data(), out_dim);
    Eigen::Map<RealMatrix> out_cov(out_covs.row(b).data(), out_dim, out_dim);
    densities(b) = condition_homodyne_into(mean, cov, mode, angles(b), efficiency, noise,
                                           outcomes(b), out_mean, out_cov);
  });
}

}  // namespace photographiq::gaussian
