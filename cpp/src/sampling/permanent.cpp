// Ryser permanent with row/column multiplicities in reflected mixed-radix Gray order.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "photographiq/sampling/matrix_functions.hpp"
#include "photographiq/sampling/summation.hpp"

namespace photographiq::sampling {
namespace {

constexpr std::int64_t kTargetChunks = 256;

Complex integer_power(Complex base, int exponent) {
  Complex result{1.0, 0.0};
  while (exponent > 0) {
    if ((exponent & 1) != 0) {
      result *= base;
    }
    base *= base;
    exponent >>= 1;
  }
  return result;
}

Real enumeration_size(const std::vector<int>& multiplicity) {
  Real size = 1.0;
  for (const int m : multiplicity) {
    size *= static_cast<Real>(m + 1);
  }
  return size;
}

// Ryser sum over x_j in [0, c_j] for a compressed matrix (all multiplicities >= 1).
Complex ryser(const ComplexMatrix& m, const std::vector<int>& rows, const std::vector<int>& cols) {
  const Index n_rows = m.rows();
  const int n_cols = static_cast<int>(cols.size());
  const int total = std::accumulate(cols.begin(), cols.end(), 0);
  const bool unit_weights = std::all_of(cols.begin(), cols.end(), [](int c) { return c == 1; });

  std::vector<std::vector<Real>> binomial(cols.size());
  for (std::size_t j = 0; j < cols.size(); ++j) {
    binomial[j].assign(static_cast<std::size_t>(cols[j]) + 1, 1.0);
    for (int a = 1; a <= cols[j]; ++a) {
      binomial[j][static_cast<std::size_t>(a)] =
          binomial[j][static_cast<std::size_t>(a) - 1] * static_cast<Real>(cols[j] - a + 1) / a;
    }
  }

  // Digits [0, inner) run in Gray order inside a chunk; digits [inner, n_cols) select the
  // chunk. Chunk count depends only on the multiplicities, never on the thread count.
  int inner = n_cols;
  std::int64_t chunks = 1;
  while (inner > 0 && chunks < kTargetChunks) {
    --inner;
    chunks *= cols[static_cast<std::size_t>(inner)] + 1;
  }

  auto chunk_sum = [&](std::int64_t chunk) {
    std::vector<int> digit(cols.size(), 0);
    std::int64_t code = chunk;
    for (int j = inner; j < n_cols; ++j) {
      const int radix = cols[static_cast<std::size_t>(j)] + 1;
      digit[static_cast<std::size_t>(j)] = static_cast<int>(code % radix);
      code /= radix;
    }
    ComplexVector row_sum = ComplexVector::Zero(n_rows);
    int digit_sum = 0;
    for (int j = inner; j < n_cols; ++j) {
      const int d = digit[static_cast<std::size_t>(j)];
      if (d != 0) {
        row_sum += static_cast<Real>(d) * m.col(j);
        digit_sum += d;
      }
    }
    detail::CompensatedComplex sum;
    auto visit = [&]() {
      // Real arithmetic for the product: std::complex multiplication is not inlined as
      // tightly by MSVC and dominated the Gray-code loop (measured ~2x slower at n = 20).
      Real re = 1.0;
      Real im = 0.0;
      const Complex* data = row_sum.data();
      for (Index i = 0; i < n_rows; ++i) {
        const int power = rows[static_cast<std::size_t>(i)];
        const Complex factor = power == 1 ? data[i] : integer_power(data[i], power);
        const Real next_re = re * factor.real() - im * factor.imag();
        im = re * factor.imag() + im * factor.real();
        re = next_re;
      }
      Complex term{re, im};
      if (!unit_weights) {
        for (std::size_t j = 0; j < cols.size(); ++j) {
          term *= binomial[j][static_cast<std::size_t>(digit[j])];
        }
      }
      sum.add(((total - digit_sum) % 2 == 0) ? term : -term);
    };
    // Knuth Algorithm H (loopless reflected mixed-radix Gray code) on digits [0, inner).
    std::vector<int> direction(static_cast<std::size_t>(inner), 1);
    std::vector<int> focus(static_cast<std::size_t>(inner) + 1);
    std::iota(focus.begin(), focus.end(), 0);
    visit();
    while (true) {
      const int j = focus[0];
      focus[0] = 0;
      if (j == inner) {
        break;
      }
      const auto uj = static_cast<std::size_t>(j);
      digit[uj] += direction[uj];
      digit_sum += direction[uj];
      if (direction[uj] > 0) {
        row_sum += m.col(j);
      } else {
        row_sum -= m.col(j);
      }
      if (digit[uj] == 0 || digit[uj] == cols[uj]) {
        direction[uj] = -direction[uj];
        focus[uj] = focus[uj + 1];
        focus[uj + 1] = j + 1;
      }
      visit();
    }
    return sum.value();
  };
  return detail::ordered_parallel_sum(chunks, chunk_sum);
}

}  // namespace

Complex permanent_repeated(const ComplexMatrix& matrix, const std::vector<int>& row_multiplicity,
                           const std::vector<int>& column_multiplicity) {
  if (static_cast<Index>(row_multiplicity.size()) != matrix.rows() ||
      static_cast<Index>(column_multiplicity.size()) != matrix.cols()) {
    throw std::invalid_argument("Multiplicity vectors must match the matrix shape");
  }
  const auto negative = [](int x) { return x < 0; };
  if (std::any_of(row_multiplicity.begin(), row_multiplicity.end(), negative) ||
      std::any_of(column_multiplicity.begin(), column_multiplicity.end(), negative)) {
    throw std::invalid_argument("Multiplicities must be nonnegative");
  }
  const std::int64_t n_rows_total =
      std::accumulate(row_multiplicity.begin(), row_multiplicity.end(), std::int64_t{0});
  const std::int64_t n_cols_total =
      std::accumulate(column_multiplicity.begin(), column_multiplicity.end(), std::int64_t{0});
  if (n_rows_total != n_cols_total) {
    throw std::invalid_argument("Row and column multiplicities must have equal sums");
  }
  if (n_rows_total == 0) {
    return {1.0, 0.0};
  }
  if (n_rows_total > std::numeric_limits<int>::max() / 2) {
    throw std::invalid_argument("Too many photons for a permanent");
  }
  // Compress away zero multiplicities.
  std::vector<Index> row_index;
  std::vector<Index> col_index;
  std::vector<int> rows;
  std::vector<int> cols;
  for (Index i = 0; i < matrix.rows(); ++i) {
    if (row_multiplicity[static_cast<std::size_t>(i)] > 0) {
      row_index.push_back(i);
      rows.push_back(row_multiplicity[static_cast<std::size_t>(i)]);
    }
  }
  for (Index j = 0; j < matrix.cols(); ++j) {
    if (column_multiplicity[static_cast<std::size_t>(j)] > 0) {
      col_index.push_back(j);
      cols.push_back(column_multiplicity[static_cast<std::size_t>(j)]);
    }
  }
  ComplexMatrix compressed = matrix(row_index, col_index);
  if (enumeration_size(rows) < enumeration_size(cols)) {
    if (enumeration_size(rows) > 9.0e18) {
      throw std::invalid_argument("Permanent enumeration too large");
    }
    return ryser(compressed.transpose(), cols, rows);
  }
  if (enumeration_size(cols) > 9.0e18) {
    throw std::invalid_argument("Permanent enumeration too large");
  }
  return ryser(compressed, rows, cols);
}

Complex permanent(const ComplexMatrix& matrix) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("Permanent requires a square matrix");
  }
  const std::vector<int> ones(static_cast<std::size_t>(matrix.rows()), 1);
  return permanent_repeated(matrix, ones, ones);
}

int max_threads() {
#if PGQ_HAVE_OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

void set_num_threads(int threads) {
  if (threads < 1) {
    throw std::invalid_argument("Thread count must be positive");
  }
#if PGQ_HAVE_OPENMP
  omp_set_num_threads(threads);
#endif
}

}  // namespace photographiq::sampling
