// C++ unit tests of photographiq::sampling against definitions and exact identities.
#if PGQ_WITH_SAMPLING

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include "photographiq/sampling/matrix_functions.hpp"

using photographiq::Complex;
using photographiq::ComplexMatrix;
using photographiq::ComplexVector;
using photographiq::Index;
using photographiq::Real;
namespace s = photographiq::sampling;

namespace {

int failures = 0;

void check(bool condition, const char* what, Real error) {
  if (!condition) {
    ++failures;
    std::printf("FAIL %s (error %.3e)\n", what, error);
  }
}

void check_close(Complex got, Complex expected, Real rtol, const char* what) {
  const Real error = std::abs(got - expected) / std::max<Real>(1.0, std::abs(expected));
  check(error <= rtol, what, error);
}

ComplexMatrix random_matrix(std::mt19937_64& rng, Index rows, Index cols) {
  std::normal_distribution<Real> normal;
  ComplexMatrix m(rows, cols);
  for (Index i = 0; i < rows; ++i) {
    for (Index j = 0; j < cols; ++j) {
      m(i, j) = Complex{normal(rng), normal(rng)};
    }
  }
  return m;
}

Complex permanent_by_definition(const ComplexMatrix& m) {
  std::vector<int> p(static_cast<std::size_t>(m.rows()));
  std::iota(p.begin(), p.end(), 0);
  Complex total{0.0, 0.0};
  do {
    Complex term{1.0, 0.0};
    for (Index i = 0; i < m.rows(); ++i) {
      term *= m(i, p[static_cast<std::size_t>(i)]);
    }
    total += term;
  } while (std::next_permutation(p.begin(), p.end()));
  return total;
}

// Sum over (loop) perfect matchings of the vertex list.
Complex matchings(const ComplexMatrix& a, std::vector<Index> vertices, bool loops) {
  if (vertices.empty()) {
    return {1.0, 0.0};
  }
  const Index first = vertices.front();
  vertices.erase(vertices.begin());
  Complex total = loops ? a(first, first) * matchings(a, vertices, loops) : Complex{0.0, 0.0};
  for (std::size_t k = 0; k < vertices.size(); ++k) {
    std::vector<Index> rest = vertices;
    rest.erase(rest.begin() + static_cast<std::ptrdiff_t>(k));
    total += a(first, vertices[k]) * matchings(a, rest, loops);
  }
  return total;
}

std::vector<Index> all_vertices(Index n) {
  std::vector<Index> v(static_cast<std::size_t>(n));
  std::iota(v.begin(), v.end(), Index{0});
  return v;
}

ComplexMatrix repeat(const ComplexMatrix& m, const std::vector<int>& r, const std::vector<int>& c) {
  std::vector<Index> rows;
  std::vector<Index> cols;
  for (std::size_t i = 0; i < r.size(); ++i) {
    rows.insert(rows.end(), static_cast<std::size_t>(r[i]), static_cast<Index>(i));
  }
  for (std::size_t j = 0; j < c.size(); ++j) {
    cols.insert(cols.end(), static_cast<std::size_t>(c[j]), static_cast<Index>(j));
  }
  return m(rows, cols);
}

}  // namespace

int main() {
  std::mt19937_64 rng(20261110);

  for (Index n = 1; n <= 8; ++n) {
    const ComplexMatrix m = random_matrix(rng, n, n);
    check_close(s::permanent(m), permanent_by_definition(m), 1e-12, "permanent vs definition");
  }
  {
    const ComplexMatrix m = random_matrix(rng, 4, 5);
    const std::vector<int> r{2, 0, 3, 1};
    const std::vector<int> c{1, 3, 0, 1, 1};
    check_close(s::permanent_repeated(m, r, c), permanent_by_definition(repeat(m, r, c)), 1e-12,
                "repeated permanent vs duplicated matrix");
    check_close(s::permanent_repeated(m, std::vector<int>(4, 0), std::vector<int>(5, 0)), 1.0,
                1e-15, "empty permanent");
  }
  // Perm of the all-ones n x n matrix is n!.
  check_close(s::permanent(ComplexMatrix::Ones(10, 10)), 3628800.0, 1e-13, "Perm(J_10) = 10!");

  for (Index d = 0; d <= 10; ++d) {
    ComplexMatrix a = random_matrix(rng, d, d);
    a = (a + a.transpose()).eval();
    check_close(s::hafnian(a), matchings(a, all_vertices(d), false), 1e-11, "hafnian vs definition");
    check_close(s::loop_hafnian(a), matchings(a, all_vertices(d), true), 1e-11,
                "loop hafnian vs definition");
  }
  {
    // Haf([[0, B], [B^T, 0]]) = Perm(B); loop hafnian with zero diagonal = hafnian.
    const ComplexMatrix b = random_matrix(rng, 6, 6);
    ComplexMatrix block = ComplexMatrix::Zero(12, 12);
    block.topRightCorner(6, 6) = b;
    block.bottomLeftCorner(6, 6) = b.transpose();
    check_close(s::hafnian(block), s::permanent(b), 1e-11, "Haf(bipartite) = Perm");
    check_close(s::loop_hafnian(block), s::permanent(b), 1e-11, "lhaf zero diagonal = haf");
    // Haf of the all-ones 2n x 2n matrix is (2n-1)!!: 11!! = 10395.
    check_close(s::hafnian(ComplexMatrix::Ones(12, 12)), 10395.0, 1e-12, "Haf(J_12) = 11!!");
  }
  {
    // Torontonian factorizes over block-diagonal (mode-independent) O; for one mode,
    // Tor([[x, y], [y*, x]]) = 1 / sqrt((1 - x)^2 - |y|^2) - 1.
    const Index n = 3;
    ComplexMatrix o = ComplexMatrix::Zero(2 * n, 2 * n);
    Complex expected{1.0, 0.0};
    ComplexVector gamma(2 * n);
    Complex expected_loop{1.0, 0.0};
    for (Index i = 0; i < n; ++i) {
      const Real x = 0.1 + 0.2 * static_cast<Real>(i);
      const Complex y{0.1, 0.05 * static_cast<Real>(i)};
      o(i, i) = x;
      o(i + n, i + n) = x;
      o(i, i + n) = y;
      o(i + n, i) = std::conj(y);
      const Complex det = (1.0 - x) * (1.0 - x) - std::norm(y);
      expected *= 1.0 / std::sqrt(det) - 1.0;
      const Complex g{0.3, -0.2 * static_cast<Real>(i)};
      gamma(i) = g;
      gamma(i + n) = std::conj(g);
      ComplexMatrix m(2, 2);
      m << 1.0 - x, -y, -std::conj(y), 1.0 - x;
      ComplexVector gi(2);
      gi << g, std::conj(g);
      const Complex quad = (gi.transpose() * m.inverse() * gi.conjugate())(0);
      expected_loop *= std::exp(0.5 * quad) / std::sqrt(det) - 1.0;
    }
    check_close(s::torontonian(o), expected, 1e-13, "torontonian product identity");
    check_close(s::loop_torontonian(o, gamma), expected_loop, 1e-13,
                "loop torontonian product identity");
    check_close(s::loop_torontonian(o, ComplexVector::Zero(2 * n)), s::torontonian(o), 1e-14,
                "ltor(O, 0) = Tor(O)");
    check_close(s::torontonian(ComplexMatrix::Zero(8, 8)), 0.0, 1e-15, "Tor(0) = 0");
  }
  {
    // Results do not depend on the thread count (fixed chunking and summation order).
    const ComplexMatrix m = random_matrix(rng, 14, 14);
    s::set_num_threads(1);
    const Complex one = s::permanent(m);
    s::set_num_threads(4);
    const Complex four = s::permanent(m);
    check(one == four, "permanent bitwise reproducible across thread counts",
          std::abs(one - four));
  }
  std::printf("%s (%d failures)\n", failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}

#else
int main() { return 0; }
#endif
