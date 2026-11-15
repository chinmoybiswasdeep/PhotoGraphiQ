// Hafnian and loop hafnian by the Glynn-type power-trace formula, OpenMP over sign vectors.
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "photographiq/sampling/matrix_functions.hpp"
#include "photographiq/sampling/summation.hpp"

namespace photographiq::sampling {
namespace {

constexpr int kMaxPairs = 40;
constexpr std::int64_t kChunkBits = 6;

void require_symmetric(const ComplexMatrix& matrix) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("Hafnian requires a square matrix");
  }
  if (matrix.size() == 0) {
    return;  // maxCoeff is undefined on an empty matrix
  }
  if (!matrix.allFinite()) {
    throw std::invalid_argument("Hafnian requires finite entries");
  }
  const Real scale = std::max<Real>(1.0, matrix.cwiseAbs().maxCoeff());
  if ((matrix - matrix.transpose()).cwiseAbs().maxCoeff() > 1e-12 * scale) {
    throw std::invalid_argument("Hafnian requires a symmetric matrix");
  }
}

// [eta^n] exp(sum_{k=1}^n g[k-1] eta^k) via e_j = (1/j) sum_{k=1}^j k g_k e_{j-k}.
Complex exp_series_coefficient(const std::vector<Complex>& g, int n) {
  std::vector<Complex> e(static_cast<std::size_t>(n) + 1, Complex{0.0, 0.0});
  e[0] = 1.0;
  for (int j = 1; j <= n; ++j) {
    Complex acc{0.0, 0.0};
    for (int k = 1; k <= j; ++k) {
      acc += static_cast<Real>(k) * g[static_cast<std::size_t>(k) - 1] *
             e[static_cast<std::size_t>(j - k)];
    }
    e[static_cast<std::size_t>(j)] = acc / static_cast<Real>(j);
  }
  return e[static_cast<std::size_t>(n)];
}

// Glynn-type term for the sign vector x (x_0 = +1, x_p = -1 iff bit p-1 of `code`), times
// prod_p x_p. C = A X diag(x (x) (1, 1)) flips the sign of both columns of every pair
// with x_p = -1. Power traces tr(C^k) come from explicit powers C^1..C^h, h = ceil(n/2),
// and tr(C^{h+b}) = sum_ij (C^h)_ij (C^b)_ji, i.e. ceil(n/2) - 1 products per term.
Complex glynn_term(const ComplexMatrix& ax, const ComplexVector* loops, int n,
                   std::uint64_t code) {
  ComplexMatrix c = ax;
  ComplexVector xs = ComplexVector::Ones(c.rows());
  int negatives = 0;
  for (int p = 1; p < n; ++p) {
    if (((code >> (p - 1)) & 1U) != 0U) {
      c.middleCols(2 * p, 2) *= -1.0;
      xs.segment(2 * p, 2).setConstant(-1.0);
      ++negatives;
    }
  }
  const int h = (n + 1) / 2;
  std::vector<ComplexMatrix> powers;
  powers.reserve(static_cast<std::size_t>(h));
  powers.push_back(c);
  for (int k = 2; k <= h; ++k) {
    powers.push_back(powers.back() * c);
  }
  std::vector<Complex> g(static_cast<std::size_t>(n));
  for (int k = 1; k <= n; ++k) {
    const Complex trace =
        k <= h ? powers[static_cast<std::size_t>(k) - 1].trace()
               : powers[static_cast<std::size_t>(h) - 1]
                     .cwiseProduct(powers[static_cast<std::size_t>(k - h) - 1].transpose())
                     .sum();
    g[static_cast<std::size_t>(k) - 1] = trace / static_cast<Real>(2 * k);
  }
  if (loops != nullptr) {
    // v^T X_x C^{k-1} v / 2 with (X_x v)_i = x_pair(i) v_partner(i).
    ComplexVector xv(loops->size());
    for (Index i = 0; i < loops->size(); ++i) {
      xv(i) = xs(i) * (*loops)(i ^ 1);
    }
    ComplexVector w = *loops;
    for (int k = 0; k < n; ++k) {
      g[static_cast<std::size_t>(k)] += 0.5 * (xv.transpose() * w)(0);  // bilinear, no conj
      w = c * w;
    }
  }
  const Complex coefficient = exp_series_coefficient(g, n);
  return (negatives % 2 == 0) ? coefficient : -coefficient;
}

Complex power_trace(const ComplexMatrix& matrix, const ComplexVector* loops) {
  const int n = static_cast<int>(matrix.rows() / 2);
  if (n == 0) {
    return {1.0, 0.0};
  }
  if (n > kMaxPairs) {
    throw std::invalid_argument("Hafnian dimension too large");
  }
  // A X with the diagonal of A removed (loops enter only through `loops`).
  ComplexMatrix ax(matrix.rows(), matrix.cols());
  for (Index j = 0; j < matrix.cols(); ++j) {
    ax.col(j) = matrix.col(j ^ 1);
    ax(j ^ 1, j) = 0.0;
  }
  const std::int64_t terms = std::int64_t{1} << (n - 1);
  const std::int64_t chunk_bits = std::min<std::int64_t>(kChunkBits, n - 1);
  const std::int64_t chunk_size = std::int64_t{1} << (n - 1 - chunk_bits);
  const std::int64_t chunks = terms / chunk_size;
  auto chunk_sum = [&](std::int64_t chunk) {
    detail::CompensatedComplex sum;
    const std::int64_t begin = chunk * chunk_size;
    for (std::int64_t code = begin; code < begin + chunk_size; ++code) {
      sum.add(glynn_term(ax, loops, n, static_cast<std::uint64_t>(code)));
    }
    return sum.value();
  };
  return detail::ordered_parallel_sum(chunks, chunk_sum) / static_cast<Real>(terms);
}

}  // namespace

Complex hafnian(const ComplexMatrix& matrix) {
  require_symmetric(matrix);
  if (matrix.rows() % 2 != 0) {
    return {0.0, 0.0};
  }
  return power_trace(matrix, nullptr);
}

Complex loop_hafnian(const ComplexMatrix& matrix) {
  require_symmetric(matrix);
  if (matrix.rows() % 2 == 0) {
    const ComplexVector loops = matrix.diagonal();
    return power_trace(matrix, &loops);
  }
  // Odd dimension: add an isolated vertex with loop weight 1 (it must be matched by its loop).
  const Index d = matrix.rows();
  ComplexMatrix padded = ComplexMatrix::Zero(d + 1, d + 1);
  padded.topLeftCorner(d, d) = matrix;
  padded(d, d) = 1.0;
  const ComplexVector loops = padded.diagonal();
  return power_trace(padded, &loops);
}

}  // namespace photographiq::sampling
