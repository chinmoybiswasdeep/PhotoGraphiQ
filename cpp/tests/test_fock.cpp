// C++ unit tests of the truncated Fock engine: basis ranking, exact Gaussian matrix
// elements against closed forms, kernels against dense references, loss and cubic phase.
#if PGQ_WITH_FOCK

#include <cmath>
#include <complex>
#include <cstdio>
#include <numbers>
#include <string>
#include <vector>

#include "photographiq/fock/basis.hpp"
#include "photographiq/fock/gates.hpp"
#include "photographiq/fock/kernels.hpp"

using namespace photographiq;
using namespace photographiq::fock;

namespace {

int failures = 0;

void check(bool ok, const std::string& what, double value = 0.0) {
  if (!ok) {
    ++failures;
    std::printf("FAIL %s (value %.3e)\n", what.c_str(), value);
  }
}

double factorial(int n) { return std::tgamma(n + 1.0); }

RealMatrix rotation(double phi) {
  RealMatrix s(2, 2);
  s << std::cos(phi), -std::sin(phi), std::sin(phi), std::cos(phi);
  return s;
}

void test_basis() {
  check(dimension(3, 4) == 20 && dimension(0, 5) == 1 && dimension(1, 7) == 7, "dimension");
  const FockBasis basis(3, 4);
  const std::vector<std::vector<int>> expected = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {2, 0, 0},
                                                  {1, 1, 0}, {1, 0, 1}, {0, 2, 0}, {0, 1, 1}, {0, 0, 2},
                                                  {3, 0, 0}, {2, 1, 0}};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto occ = basis.occupation(static_cast<Index>(i));
    for (int m = 0; m < 3; ++m) check(occ[m] == expected[i][m], "piquasso ordering");
  }
  for (int modes = 1; modes <= 5; ++modes) {
    const FockBasis b(modes, 6);
    for (Index i = 0; i < b.size(); ++i) check(b.rank(b.occupation(i)) == i, "rank(unrank(i)) == i");
  }
  bool threw = false;
  try {
    const std::vector<Occupation> outside = {2, 2, 0};
    (void)basis.rank(outside);
  } catch (const PhysicsError&) {
    threw = true;
  }
  check(threw, "rank rejects total >= cutoff");
  // Fibres partition the basis and are ordered by local rank.
  const std::vector<int> local = {2, 0};
  const Fibers fibers = make_fibers(basis, local);
  std::vector<int> seen(static_cast<std::size_t>(basis.size()), 0);
  for (const Index g : fibers.indices) ++seen[static_cast<std::size_t>(g)];
  for (const int s : seen) check(s == 1, "fibres partition the basis");
}

void test_displacement() {
  const Complex alpha(0.7, -0.4);
  const int cutoff = 25;
  RealVector d(2);
  d << 2 * alpha.real(), 2 * alpha.imag();
  const ComplexMatrix u = gaussian_unitary(RealMatrix::Identity(2, 2), d, cutoff);
  const double x = std::norm(alpha);
  double worst = 0.0;
  for (int m = 0; m < cutoff; ++m) {
    for (int n = 0; n < cutoff; ++n) {
      Complex exact;
      if (m >= n) {
        exact = std::sqrt(factorial(n) / factorial(m)) * std::pow(alpha, m - n) * std::exp(-x / 2) *
                std::assoc_laguerre(n, m - n, x);
      } else {
        exact = std::sqrt(factorial(m) / factorial(n)) * std::pow(-std::conj(alpha), n - m) *
                std::exp(-x / 2) * std::assoc_laguerre(m, n - m, x);
      }
      worst = std::max(worst, std::abs(u(m, n) - exact));
    }
  }
  check(worst < 1e-12, "displacement matrix elements == Laguerre closed form", worst);
  // S = rotation(2 pi) is not exactly the identity, so it takes the Bargmann path.
  const ComplexMatrix bargmann = gaussian_unitary(rotation(2 * std::numbers::pi), d, cutoff);
  const double paths = (bargmann - u).cwiseAbs().maxCoeff();
  check(paths < 1e-12, "displacement: Laguerre path == Bargmann path", paths);
}

// Elements of P U P are bounded by one. The old bra-first pivot violated this by orders of
// magnitude (5e4 for S(r = 1) at cutoff 150, 2e10 for D(4) at cutoff 100).
void test_recurrence_stability() {
  RealVector d(2);
  d << 8.0 * std::cos(0.7), 8.0 * std::sin(0.7);  // |alpha| = 4
  const ComplexMatrix disp = gaussian_unitary(RealMatrix::Identity(2, 2), d, 120);
  const ComplexMatrix disp_b = gaussian_unitary(rotation(2 * std::numbers::pi), d, 40);
  RealMatrix s = RealMatrix::Zero(2, 2);
  s(0, 0) = std::exp(-1.0), s(1, 1) = std::exp(1.0);
  const ComplexMatrix sq = gaussian_unitary(s, RealVector::Zero(2), 150);
  check(disp.cwiseAbs().maxCoeff() <= 1 + 1e-12, "|<m|D|n>| <= 1", disp.cwiseAbs().maxCoeff());
  check(sq.cwiseAbs().maxCoeff() <= 1 + 1e-12, "|<m|S|n>| <= 1", sq.cwiseAbs().maxCoeff());
  // Bargmann recurrence with displacement at cutoff 40 vs the Laguerre path (the old
  // pivot was off by 5.8e-3 here).
  const double diff = (disp_b - disp.topLeftCorner(40, 40)).cwiseAbs().maxCoeff();
  check(diff < 1e-8, "Bargmann vs Laguerre displacement at |alpha| = 4, cutoff 40", diff);
  // S(-r) S(r) = 1 on low photon numbers needs accurate high rows (tail tanh(1)^75 ~ 1e-9).
  s(0, 0) = std::exp(1.0), s(1, 1) = std::exp(-1.0);
  const ComplexMatrix inverse = gaussian_unitary(s, RealVector::Zero(2), 150);
  const double residual = ((inverse * sq).topLeftCorner(8, 8) - ComplexMatrix::Identity(8, 8)).cwiseAbs().maxCoeff();
  check(residual < 1e-8, "S(-1) S(1) = 1 at cutoff 150", residual);
}

void test_squeezing() {
  const double r = 0.6;
  const int cutoff = 30;
  RealMatrix s = RealMatrix::Zero(2, 2);
  s(0, 0) = std::exp(-r);
  s(1, 1) = std::exp(r);
  const ComplexMatrix u = gaussian_unitary(s, RealVector::Zero(2), cutoff);
  double worst = 0.0;
  for (int n = 0; 2 * n < cutoff; ++n) {
    const double exact = std::sqrt(1 / std::cosh(r)) * std::pow(-std::tanh(r), n) *
                         std::sqrt(factorial(2 * n)) / (std::pow(2.0, n) * factorial(n));
    worst = std::max(worst, std::abs(u(2 * n, 0) - exact));
    if (2 * n + 1 < cutoff) worst = std::max(worst, std::abs(u(2 * n + 1, 0)));
  }
  check(worst < 1e-14, "squeezed vacuum amplitudes", worst);
  // Composition S(-r) S(r) = 1 on low photon numbers. The truncated product misses
  // sum_{k >= c} <m|S(-r)|k><k|S(r)|n> ~ tanh(r)^(c/2): ~1e-4 at c = 30, ~1e-11 at c = 80.
  const int large = 80;
  const ComplexMatrix forward = gaussian_unitary(s, RealVector::Zero(2), large);
  s(0, 0) = std::exp(r);
  s(1, 1) = std::exp(-r);
  const ComplexMatrix inverse = gaussian_unitary(s, RealVector::Zero(2), large);
  const ComplexMatrix product = (inverse * forward).topLeftCorner(5, 5);
  const double residual = (product - ComplexMatrix::Identity(5, 5)).cwiseAbs().maxCoeff();
  check(residual < 1e-9, "S(-r) S(r) = 1 on low photon numbers", residual);
}

// One mode, squeezing and displacement combined (the Bargmann recurrence was off by 1.4e2
// at |alpha| = 6, r = 0.3, cutoff 100, and squeezing alone by 7e-5 at size 400). Checked
// through identities exact for the matrix elements: a U = U (T a + R a^dag + delta) and
// S(-r) S(r) = 1. 40-digit references: tests/native/test_native_fock_gaussian_unitary_stability.py.
void test_single_mode_stability() {
  const int cutoff = 100;
  RealMatrix squeeze = RealMatrix::Zero(2, 2);
  squeeze(0, 0) = std::exp(-0.3), squeeze(1, 1) = std::exp(0.3);
  const RealMatrix s = rotation(1.9) * squeeze * rotation(0.3);
  const Complex alpha = std::polar(6.0, 2.1);
  RealVector d(2);
  d << 2 * alpha.real(), 2 * alpha.imag();
  const ComplexMatrix u = gaussian_unitary(s, d, cutoff);
  const Complex i1(0.0, 1.0);
  const Complex t = (s(0, 0) + s(1, 1) + i1 * (s(1, 0) - s(0, 1))) / 2.0;
  const Complex r = (s(0, 0) - s(1, 1) + i1 * (s(1, 0) + s(0, 1))) / 2.0;
  double heisenberg = 0.0;
  for (int m = 0; m + 1 < cutoff; ++m) {
    for (int n = 0; n + 1 < cutoff; ++n) {
      Complex rhs = r * std::sqrt(n + 1.0) * u(m, n + 1) + alpha * u(m, n);
      if (n > 0) rhs += t * std::sqrt(Real(n)) * u(m, n - 1);
      heisenberg = std::max(heisenberg, std::abs(std::sqrt(m + 1.0) * u(m + 1, n) - rhs));
    }
  }
  check(heisenberg < 1e-12, "a U = U (T a + R a^dag + delta) at |alpha| = 6, cutoff 100", heisenberg);
  // Columns 0..5 lose < 1e-13 above the cutoff, so they are unit vectors.
  double norm = 0.0;
  for (int n = 0; n < 6; ++n) norm = std::max(norm, std::abs(u.col(n).squaredNorm() - 1.0));
  check(norm < 1e-13, "low columns of the combined unitary are normalised", norm);

  RealMatrix sq = RealMatrix::Zero(2, 2);
  sq(0, 0) = std::exp(-1.0), sq(1, 1) = std::exp(1.0);
  // S(1)|n> reaches ~e^2 n photons, so 70 x 70 at size 800 has no truncation error
  // (2e-14; the Bargmann recurrence gave 1.1e-8).
  const ComplexMatrix forward = gaussian_unitary(rotation(0.4) * sq, RealVector::Zero(2), 800);
  sq(0, 0) = std::exp(1.0), sq(1, 1) = std::exp(-1.0);
  const ComplexMatrix inverse = gaussian_unitary(sq * rotation(-0.4), RealVector::Zero(2), 800);
  const double residual =
      ((inverse * forward).topLeftCorner(70, 70) - ComplexMatrix::Identity(70, 70)).cwiseAbs().maxCoeff();
  check(residual < 1e-12, "S(-1) S(1) = 1 on 70 x 70 at size 800", residual);
}

void test_rotation_and_beamsplitter() {
  const int cutoff = 12;
  const double phi = 0.37;
  const ComplexMatrix r = gaussian_unitary(rotation(phi), RealVector::Zero(2), cutoff);
  double worst = 0.0;
  for (int n = 0; n < cutoff; ++n) {
    for (int m = 0; m < cutoff; ++m) {
      const Complex exact = m == n ? std::polar(1.0, phi * n) : Complex(0.0);
      worst = std::max(worst, std::abs(r(m, n) - exact));
    }
  }
  check(worst < 1e-13, "rotation = exp(i phi n)", worst);

  const double theta = std::numbers::pi / 4;
  RealMatrix bs = RealMatrix::Zero(4, 4);
  const double c = std::cos(theta), s = std::sin(theta);
  bs(0, 0) = c, bs(0, 2) = -s, bs(2, 0) = s, bs(2, 2) = c;
  bs(1, 1) = c, bs(1, 3) = -s, bs(3, 1) = s, bs(3, 3) = c;
  const ComplexMatrix u = gaussian_unitary(bs, RealVector::Zero(4), cutoff);
  const FockBasis basis(2, cutoff);
  const std::vector<Occupation> o11 = {1, 1}, o20 = {2, 0}, o02 = {0, 2};
  const Index i11 = basis.rank(o11), i20 = basis.rank(o20), i02 = basis.rank(o02);
  check(std::abs(u(i11, i11)) < 1e-14, "Hong-Ou-Mandel suppression", std::abs(u(i11, i11)));
  check(std::abs(std::norm(u(i20, i11)) - 0.5) < 1e-14 && std::abs(std::norm(u(i02, i11)) - 0.5) < 1e-14,
        "Hong-Ou-Mandel bunching");
  // Exactly unitary and photon-number conserving on the truncated space.
  const double unitarity = (u.adjoint() * u - ComplexMatrix::Identity(u.rows(), u.cols())).cwiseAbs().maxCoeff();
  check(unitarity < 1e-12, "beam splitter unitary within sectors", unitarity);
}

void test_kernels() {
  const int cutoff = 8;
  const FockBasis basis(3, cutoff);
  ComplexVector psi(basis.size());
  for (Index i = 0; i < psi.size(); ++i) psi(i) = Complex(std::sin(1.0 + i), std::cos(2.0 * i)) / (1.0 + i);
  psi.normalize();
  RealVector d(2);
  d << 0.4, -0.3;
  const ComplexMatrix u = gaussian_unitary(rotation(0.2), d, cutoff);
  // Dense reference: <g'|U_mode1|g> = u(n'_1, n_1) if others agree and n'_1 + rest < cutoff.
  ComplexMatrix dense = ComplexMatrix::Zero(basis.size(), basis.size());
  for (Index i = 0; i < basis.size(); ++i) {
    for (Index j = 0; j < basis.size(); ++j) {
      const auto a = basis.occupation(i), b = basis.occupation(j);
      if (a[0] == b[0] && a[2] == b[2]) dense(i, j) = u(a[1], b[1]);
    }
  }
  ComplexVector out = psi;
  const std::vector<int> local = {1};
  apply_local(StateRef(out), basis, local, u);
  const double pure_error = (out - dense * psi).cwiseAbs().maxCoeff();
  check(pure_error < 1e-14, "single-mode kernel == dense restricted operator", pure_error);

  RowComplexMatrix rho = psi * psi.adjoint();
  apply_local(DensityRef(rho), basis, local, u);
  const ComplexMatrix expected = dense * (psi * psi.adjoint()) * dense.adjoint();
  const double mixed_error = (ComplexMatrix(rho) - expected).cwiseAbs().maxCoeff();
  check(mixed_error < 1e-14, "density kernel == U rho U^dag", mixed_error);

  // Partial trace in requested order agrees between pure and mixed paths.
  const std::vector<int> keep = {2, 0};
  const ComplexMatrix a = reduced_density(ConstStateRef(psi), basis, keep);
  const RowComplexMatrix rho0 = psi * psi.adjoint();
  const ComplexMatrix b = reduced_density(ConstDensityRef(rho0), basis, keep);
  check((a - b).cwiseAbs().maxCoeff() < 1e-15 && std::abs(a.trace() - 1.0) < 1e-14, "reduced density");
  const Physicality p = physicality(RowComplexMatrix(a));
  check(p.hermiticity < 1e-15 && p.min_eigenvalue > -1e-14, "reduced state physical");
}

void test_loss() {
  // |n> through pure loss: binomial photon statistics, exactly (loss never leaves the space).
  const int cutoff = 10;
  const FockBasis basis(1, cutoff);
  RowComplexMatrix rho = RowComplexMatrix::Zero(cutoff, cutoff);
  rho(6, 6) = 1.0;
  const double eta = 0.7;
  const RowComplexMatrix out = thermal_loss(rho, basis, 0, eta, 0.0);
  double worst = 0.0;
  for (int k = 0; k <= 6; ++k) {
    const double exact = std::tgamma(7.0) / (std::tgamma(k + 1.0) * std::tgamma(7.0 - k)) *
                         std::pow(eta, k) * std::pow(1 - eta, 6 - k);
    worst = std::max(worst, std::abs(out(k, k) - exact));
  }
  check(worst < 1e-15, "pure loss of a Fock state is binomial", worst);
  // Thermal loss of vacuum: thermal state with mean (1 - T) n_th (geometric).
  const int big = 60;
  const FockBasis large(1, big);
  RowComplexMatrix vac = RowComplexMatrix::Zero(big, big);
  vac(0, 0) = 1.0;
  const double t = 0.6, nth = 0.8, mean = (1 - t) * nth;
  const RowComplexMatrix th = thermal_loss(vac, large, 0, t, nth);
  worst = 0.0;
  for (int n = 0; n < 20; ++n) worst = std::max(worst, std::abs(th(n, n) - std::pow(mean, n) / std::pow(1 + mean, n + 1)));
  check(worst < 1e-14, "thermal loss of vacuum is thermal", worst);
}

void test_cubic_phase() {
  const int cutoff = 12;
  const double gamma = 0.15;
  const CubicPhaseMatrix projected = cubic_phase_projected(gamma, cutoff);
  check(projected.change < 1e-13, "cubic quadrature converged", projected.change);
  // Independent method: truncated generator in a much larger space converges to P V P.
  const ComplexMatrix big = cubic_phase_truncated_generator(gamma, 400).topLeftCorner(cutoff, cutoff);
  const double difference = (big - projected.matrix).topLeftCorner(6, 6).cwiseAbs().maxCoeff();
  check(difference < 1e-6, "cubic phase: quadrature vs large truncated generator", difference);
  const ComplexMatrix zero = cubic_phase_projected(0.0, cutoff).matrix;
  check((zero - ComplexMatrix::Identity(cutoff, cutoff)).cwiseAbs().maxCoeff() < 1e-13, "gamma = 0 is identity");
}

}  // namespace

int main() {
  test_basis();
  test_displacement();
  test_recurrence_stability();
  test_squeezing();
  test_single_mode_stability();
  test_rotation_and_beamsplitter();
  test_kernels();
  test_loss();
  test_cubic_phase();
  std::printf("test_fock: %d failure(s)\n", failures);
  return failures == 0 ? 0 : 1;
}

#else
int main() { return 0; }
#endif
