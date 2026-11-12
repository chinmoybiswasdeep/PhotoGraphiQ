// Torontonian and loop torontonian as compensated subset sums of LU determinants.
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include "photographiq/sampling/matrix_functions.hpp"
#include "photographiq/sampling/summation.hpp"

namespace photographiq::sampling {
namespace {

constexpr int kMaxModes = 40;
constexpr std::int64_t kChunkBits = 6;

Complex subset_sum(const ComplexMatrix& matrix, const ComplexVector* gamma) {
  if (matrix.rows() != matrix.cols() || matrix.rows() % 2 != 0) {
    throw std::invalid_argument("Torontonian requires a 2n x 2n matrix");
  }
  if (gamma != nullptr && gamma->size() != matrix.rows()) {
    throw std::invalid_argument("Loop torontonian vector must match the matrix size");
  }
  const int n = static_cast<int>(matrix.rows() / 2);
  if (n > kMaxModes) {
    throw std::invalid_argument("Torontonian dimension too large");
  }
  const std::int64_t subsets = std::int64_t{1} << n;
  const std::int64_t chunk_bits = std::min<std::int64_t>(kChunkBits, n);
  const std::int64_t chunk_size = std::int64_t{1} << (n - chunk_bits);
  const std::int64_t chunks = subsets / chunk_size;

  auto term = [&](std::uint64_t mask) -> Complex {
    std::vector<Index> index;
    int size = 0;
    for (int i = 0; i < n; ++i) {
      if (((mask >> i) & 1U) != 0U) {
        index.push_back(i);
        ++size;
      }
    }
    for (int k = 0; k < size; ++k) {
      index.push_back(index[static_cast<std::size_t>(k)] + n);
    }
    Complex value{1.0, 0.0};
    if (size > 0) {
      const ComplexMatrix m =
          ComplexMatrix::Identity(2 * size, 2 * size) - matrix(index, index);
      const Eigen::PartialPivLU<ComplexMatrix> lu(m);
      value = 1.0 / std::sqrt(lu.determinant());
      if (gamma != nullptr) {
        const ComplexVector g = (*gamma)(index);
        const ComplexVector solved = lu.solve(g.conjugate());
        value *= std::exp(0.5 * (g.transpose() * solved)(0));
      }
    }
    return ((n - size) % 2 == 0) ? value : -value;
  };
  auto chunk_sum = [&](std::int64_t chunk) {
    detail::CompensatedComplex sum;
    const std::int64_t begin = chunk * chunk_size;
    for (std::int64_t mask = begin; mask < begin + chunk_size; ++mask) {
      sum.add(term(static_cast<std::uint64_t>(mask)));
    }
    return sum.value();
  };
  return detail::ordered_parallel_sum(chunks, chunk_sum);
}

}  // namespace

std::pair<Real, Real> threshold_vacuum_sum(const RealMatrix& excess, const RealVector& shift) {
  if (excess.rows() != excess.cols() || excess.rows() % 2 != 0 || shift.size() != excess.rows()) {
    throw std::invalid_argument("threshold_vacuum_sum needs a 2k x 2k matrix and a 2k vector");
  }
  const int k = static_cast<int>(excess.rows() / 2);
  if (k > kMaxModes) {
    throw std::invalid_argument("Too many clicked modes");
  }
  const bool displaced = k > 0 && shift.cwiseAbs().maxCoeff() != 0.0;
  const std::int64_t subsets = std::int64_t{1} << k;
  const std::int64_t chunk_bits = std::min<std::int64_t>(kChunkBits, k);
  const std::int64_t chunk_size = std::int64_t{1} << (k - chunk_bits);
  const std::int64_t chunks = subsets / chunk_size;
  bool failed = false;
  auto term = [&](std::uint64_t mask) -> Real {
    std::vector<Index> index;
    for (int i = 0; i < k; ++i) {
      if (((mask >> i) & 1U) != 0U) {
        index.push_back(2 * i);
        index.push_back(2 * i + 1);
      }
    }
    if (index.empty()) {
      return 0.0;  // w_T = expm1(0)
    }
    const RealMatrix sub = excess(index, index);
    const Eigen::SelfAdjointEigenSolver<RealMatrix> eig(sub, Eigen::EigenvaluesOnly);
    if (eig.info() != Eigen::Success || eig.eigenvalues().minCoeff() <= -1.0) {
      failed = true;  // benign race: only ever set to true
      return 0.0;
    }
    Real exponent = -0.5 * eig.eigenvalues().array().log1p().sum();
    if (displaced) {
      const RealMatrix c = 2.0 * (RealMatrix::Identity(sub.rows(), sub.cols()) + sub);
      const RealVector e = shift(index);
      exponent -= 0.5 * e.dot(Eigen::LLT<RealMatrix>(c).solve(e));
    }
    const Real w = std::expm1(exponent);
    return (index.size() / 2) % 2 == 0 ? w : -w;
  };
  // Complex packing: the real part accumulates the terms, the imaginary part |terms|.
  auto chunk_sum = [&](std::int64_t chunk) {
    detail::CompensatedComplex sum;
    const std::int64_t begin = chunk * chunk_size;
    for (std::int64_t mask = begin; mask < begin + chunk_size; ++mask) {
      const Real t = term(static_cast<std::uint64_t>(mask));
      sum.add(Complex{t, std::abs(t)});
    }
    return sum.value();
  };
  const Complex total = detail::ordered_parallel_sum(chunks, chunk_sum);
  if (failed) {
    throw PhysicsError("Conditioned covariance block is not positive definite");
  }
  const Real empty = k == 0 ? 1.0 : 0.0;
  return {empty + total.real(), empty + total.imag()};
}

Complex torontonian(const ComplexMatrix& matrix) { return subset_sum(matrix, nullptr); }

Complex loop_torontonian(const ComplexMatrix& matrix, const ComplexVector& gamma) {
  return subset_sum(matrix, &gamma);
}

}  // namespace photographiq::sampling
