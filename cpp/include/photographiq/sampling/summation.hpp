// Internal helpers shared by the sampling kernels: compensated summation and chunked
// OpenMP reduction with a thread-count independent summation order.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "photographiq/core/types.hpp"

#if PGQ_HAVE_OPENMP
#include <omp.h>
#endif

namespace photographiq::sampling::detail {

// Neumaier (improved Kahan) summation of one real component.
class CompensatedReal {
 public:
  void add(Real value) {
    const Real t = sum_ + value;
    compensation_ += std::abs(sum_) >= std::abs(value) ? (sum_ - t) + value : (value - t) + sum_;
    sum_ = t;
  }
  [[nodiscard]] Real value() const { return sum_ + compensation_; }

 private:
  Real sum_ = 0.0;
  Real compensation_ = 0.0;
};

class CompensatedComplex {
 public:
  void add(const Complex& value) {
    real_.add(value.real());
    imag_.add(value.imag());
  }
  [[nodiscard]] Complex value() const { return {real_.value(), imag_.value()}; }

 private:
  CompensatedReal real_;
  CompensatedReal imag_;
};

// Evaluate chunk_sum(c) for c in [0, chunks) in parallel and add the results in index order.
template <typename ChunkSum>
Complex ordered_parallel_sum(std::int64_t chunks, const ChunkSum& chunk_sum) {
  std::vector<Complex> partial(static_cast<std::size_t>(chunks));
#if PGQ_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (std::int64_t c = 0; c < chunks; ++c) {
    partial[static_cast<std::size_t>(c)] = chunk_sum(c);
  }
  CompensatedComplex total;
  for (const Complex& value : partial) {
    total.add(value);
  }
  return total.value();
}

}  // namespace photographiq::sampling::detail
