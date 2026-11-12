#include "photographiq/fock/gates.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include "photographiq/core/symplectic.hpp"
#include "photographiq/fock/basis.hpp"

namespace photographiq::fock {
namespace {

// Rank of occupation(index) - e_mode, or -1 when that occupation is zero.
std::vector<Index> lowered_table(const FockBasis& basis) {
  const int k = basis.modes();
  std::vector<Index> table(static_cast<std::size_t>(basis.size() * k), -1);
  std::vector<Occupation> occ(static_cast<std::size_t>(k));
  for (Index r = 0; r < basis.size(); ++r) {
    const auto source = basis.occupation(r);
    for (int j = 0; j < k; ++j) {
      if (source[j] == 0) continue;
      occ.assign(source.begin(), source.end());
      --occ[j];
      table[static_cast<std::size_t>(r * k + j)] = basis.rank(occ);
    }
  }
  return table;
}

// <m|D(alpha)|n> along each diagonal m - n = j from the normalised Laguerre recurrence
// in the degree, e_n = <n + j|D|n> = sqrt(n!/(n + j)!) alpha^j e^{-x/2} L_n^(j)(x),
// x = |alpha|^2:  e_{n+1} = ((2n + 1 + j - x) e_n - sqrt(n (n + j)) e_{n-1}) /
// sqrt((n + 1)(n + j + 1)), and <n|D|n + j> = (-1)^j conj(e_n). Unlike the Bargmann
// recurrence (1e-5 error at |alpha| = 4, cutoff 80 even with the stable pivot), this
// stays at 1e-15 for |alpha| <= 7, cutoff <= 120, measured against 60-digit mpmath
// arithmetic (regression tests: cpp/tests/test_fock.cpp, tests/native expm references).
ComplexMatrix displacement_matrix(Complex alpha, int cutoff) {
  const Real x = std::norm(alpha);
  const Real log_abs = x > 0.0 ? 0.5 * std::log(x) : 0.0;
  const Real phase = std::arg(alpha);
  ComplexMatrix d = ComplexMatrix::Zero(cutoff, cutoff);
  for (int j = 0; j < cutoff; ++j) {
    if (x == 0.0 && j > 0) break;
    const Real log_e0 = j * log_abs - 0.5 * x - 0.5 * std::lgamma(j + 1.0);
    Complex previous = std::polar(std::exp(log_e0), j * phase);
    Complex current = (1.0 + j - x) * previous / std::sqrt(j + 1.0);
    const Real sign = j % 2 == 0 ? 1.0 : -1.0;
    for (int n = 0; n + j < cutoff; ++n) {
      const Complex value = n == 0 ? previous : current;
      d(n + j, n) = value;
      if (j > 0) d(n, n + j) = sign * std::conj(value);
      if (n >= 1) {
        const Complex next = ((2.0 * n + 1 + j - x) * current - std::sqrt(Real(n) * (n + j)) * previous) /
                             std::sqrt((n + 1.0) * (n + j + 1.0));
        previous = current;
        current = next;
      }
    }
  }
  return d;
}

}  // namespace

ComplexMatrix gaussian_unitary(const RealMatrix& symplectic, const RealVector& displacement,
                               int cutoff) {
  if (symplectic.rows() != symplectic.cols() || symplectic.rows() % 2 != 0 ||
      symplectic.rows() == 0 || displacement.size() != symplectic.rows()) {
    throw PhysicsError("Gaussian unitary needs a 2k x 2k symplectic matrix and 2k displacement");
  }
  if (!symplectic.allFinite() || !displacement.allFinite()) {
    throw PhysicsError("Gaussian unitary parameters must be finite");
  }
  const Real scale = std::max<Real>(1.0, symplectic.cwiseAbs().maxCoeff());
  if (symplectic_residual(symplectic) > 1e-9 * scale * scale) {
    throw PhysicsError("Matrix is not symplectic in the interleaved convention");
  }
  const Index k = symplectic.rows() / 2;
  if (cutoff < 1) throw PhysicsError("Fock cutoff must be at least one");
  if (k == 1 && symplectic == RealMatrix::Identity(2, 2)) {
    return displacement_matrix(Complex(displacement(0), displacement(1)) / 2.0, cutoff);
  }
  const Complex i1(0.0, 1.0);
  // x = L (a; a^dag) with q = a + a^dag, p = -i (a - a^dag); L^-1 recovers (a; a^dag).
  ComplexMatrix to_quad = ComplexMatrix::Zero(2 * k, 2 * k);
  ComplexMatrix to_ladder = ComplexMatrix::Zero(2 * k, 2 * k);
  for (Index j = 0; j < k; ++j) {
    to_quad(2 * j, j) = 1.0;
    to_quad(2 * j, k + j) = 1.0;
    to_quad(2 * j + 1, j) = -i1;
    to_quad(2 * j + 1, k + j) = i1;
    to_ladder(j, 2 * j) = 0.5;
    to_ladder(j, 2 * j + 1) = 0.5 * i1;
    to_ladder(k + j, 2 * j) = 0.5;
    to_ladder(k + j, 2 * j + 1) = -0.5 * i1;
  }
  const ComplexMatrix bogoliubov = to_ladder * symplectic.cast<Complex>() * to_quad;
  const ComplexMatrix t = bogoliubov.topLeftCorner(k, k);
  const ComplexMatrix r = bogoliubov.topRightCorner(k, k);
  ComplexVector delta(k);
  for (Index j = 0; j < k; ++j) delta(j) = Complex(displacement(2 * j), displacement(2 * j + 1)) / 2.0;

  const Eigen::PartialPivLU<ComplexMatrix> t_conj_lu(t.conjugate());
  const ComplexMatrix t_conj_inv = t_conj_lu.inverse();
  const ComplexMatrix t_adj_inv = t_conj_inv.transpose();  // (T^dag)^-1 = (conj(T)^-1)^T
  ComplexMatrix a(2 * k, 2 * k);
  a.topLeftCorner(k, k) = r * t_conj_inv;                       // A_zz
  a.topRightCorner(k, k) = t_adj_inv;                           // A_zw
  a.bottomLeftCorner(k, k) = t_adj_inv.transpose();             // A_wz = A_zw^T
  a.bottomRightCorner(k, k) = -r.adjoint() * t_adj_inv;         // A_ww
  ComplexVector b(2 * k);
  b.tail(k) = -t_conj_inv * delta.conjugate();                  // b_w
  b.head(k) = delta + r * b.tail(k);                            // b_z
  const Complex quadratic = (delta.conjugate().transpose() * a.topLeftCorner(k, k) *
                             delta.conjugate())(0, 0);
  const Real det_abs = std::abs(t_conj_lu.determinant());
  const Complex vacuum =
      std::exp(-0.5 * delta.squaredNorm() + 0.5 * quadratic) / std::sqrt(det_abs);

  const FockBasis basis(static_cast<int>(k), cutoff);
  const Index dim = basis.size();
  const std::vector<Index> lower = lowered_table(basis);
  const Occupation* occ = basis.table().data();
  // Pivot of the recurrence: the side (bra or ket) with more photons, and on it the
  // most occupied mode. Always lowering the bra first (the previous rule) is unstable:
  // in double precision it amplified rounding to 5.8e-3 for D(alpha = 4) at cutoff 40
  // and to 0.27 for S(r = 1) at cutoff 120; this rule keeps the multipliers
  // sqrt(K_j / K_i) <= 1 (errors 1e-14 to 1e-12 at those sizes, tests/native).
  std::vector<Index> most(static_cast<std::size_t>(dim), -1);
  std::vector<int> total(static_cast<std::size_t>(dim), 0);
  for (Index rank = 1; rank < dim; ++rank) {
    Occupation best = 0;
    for (Index j = 0; j < k; ++j) {
      const Occupation nj = occ[rank * k + j];
      total[rank] += nj;
      if (nj > best) best = nj, most[rank] = j;
    }
  }
  std::vector<Real> root(static_cast<std::size_t>(cutoff) + 1);
  for (int n = 0; n <= cutoff; ++n) root[n] = std::sqrt(Real(n));
  // Row-major copy of A for contiguous access along the pivot row.
  const Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> arow = a;

  // A passive transformation (R = 0, delta = 0) conserves photon number, so blocks
  // between different total-photon sectors vanish identically and are skipped. R is zero
  // in exact arithmetic for an orthogonal symplectic S; 1e-14 only absorbs rounding.
  const bool passive = r.cwiseAbs().maxCoeff() <= 1e-14 && delta.squaredNorm() == 0.0;
  std::vector<Index> sector_start(static_cast<std::size_t>(cutoff) + 1);
  for (int n = 0; n <= cutoff; ++n) sector_start[n] = n == 0 ? 0 : dimension(static_cast<int>(k), n);

  ComplexMatrix g = ComplexMatrix::Zero(dim, dim);
  g(0, 0) = vacuum;
  // Columns n (ket) outer, rows m (bra) inner: every predecessor has a smaller rank in
  // the decremented index, so it is already available.
  int sector = 0;
  for (Index n = 0; n < dim; ++n) {
    while (n >= sector_start[sector + 1]) ++sector;
    const Index m_begin = passive ? sector_start[sector] : 0;
    const Index m_end = passive ? sector_start[sector + 1] : dim;
    for (Index m = m_begin; m < m_end; ++m) {
      if (m == 0 && n == 0) continue;
      Index pivot;  // joint index: 0..k-1 bra (z), k..2k-1 ket (w)
      Index pm = m;
      Index pn = n;
      Occupation count;
      if (total[m] >= total[n]) {
        pivot = most[m];
        count = occ[m * k + pivot];
        pm = lower[static_cast<std::size_t>(m * k + pivot)];
      } else {
        const Index j = most[n];
        pivot = k + j;
        count = occ[n * k + j];
        pn = lower[static_cast<std::size_t>(n * k + j)];
      }
      const Complex* apivot = arow.data() + pivot * 2 * k;
      Complex value = b(pivot) * g(pm, pn);
      for (Index j = 0; j < k; ++j) {
        const Occupation nm = occ[pm * k + j];
        const Occupation nn = occ[pn * k + j];
        if (nm > 0) value += apivot[j] * root[nm] * g(lower[static_cast<std::size_t>(pm * k + j)], pn);
        if (nn > 0) value += apivot[k + j] * root[nn] * g(pm, lower[static_cast<std::size_t>(pn * k + j)]);
      }
      g(m, n) = value / root[count];
    }
  }
  return g;
}

namespace {

constexpr Index kSampleBlock = 4096;

// sum_j h_m(y_j) h_n(y_j) exp(i gamma sqrt(2) y_j^3 / 3) over y_j = first + j * spacing,
// j < count, with orthonormal Hermite functions h_n (stable forward recurrence).
ComplexMatrix hermite_phase_sum(Real gamma, int cutoff, Real first, Real spacing, Index count) {
  RealMatrix re = RealMatrix::Zero(cutoff, cutoff);
  RealMatrix im = RealMatrix::Zero(cutoff, cutoff);
  const Real log_pim4 = -0.25 * std::log(std::numbers::pi);
  for (Index start = 0; start < count; start += kSampleBlock) {
    const Index size = std::min(kSampleBlock, count - start);
    RealMatrix h(size, cutoff);
    RealVector cosine(size);
    RealVector sine(size);
    for (Index j = 0; j < size; ++j) {
      const Real y = first + static_cast<Real>(start + j) * spacing;
      h(j, 0) = std::exp(log_pim4 - 0.5 * y * y);
      if (cutoff > 1) h(j, 1) = std::sqrt(2.0) * y * h(j, 0);
      for (int n = 1; n + 1 < cutoff; ++n) {
        h(j, n + 1) = std::sqrt(2.0 / (n + 1)) * y * h(j, n) - std::sqrt(Real(n) / (n + 1)) * h(j, n - 1);
      }
      const Real phase = gamma * std::sqrt(2.0) * y * y * y / 3.0;
      cosine(j) = std::cos(phase);
      sine(j) = std::sin(phase);
    }
    re.noalias() += h.transpose() * cosine.asDiagonal() * h;
    im.noalias() += h.transpose() * sine.asDiagonal() * h;
  }
  ComplexMatrix result(cutoff, cutoff);
  result.real() = re;
  result.imag() = im;
  return result;
}

}  // namespace

CubicPhaseMatrix cubic_phase_projected(Real gamma, int cutoff, Real tolerance, Index max_points) {
  if (!std::isfinite(gamma) || cutoff < 1 || !(tolerance > 0.0) || max_points < 16) {
    throw PhysicsError("Invalid cubic-phase parameters");
  }
  // h_n(y), n < cutoff, decays beyond the turning point sqrt(2n + 1); ten units further
  // it is below exp(-80), so the window [-y_max, y_max] loses nothing representable.
  const Real y_max = std::sqrt(2.0 * cutoff + 1.0) + 10.0;
  Index intervals = 16;
  Real spacing = 2.0 * y_max / static_cast<Real>(intervals);
  ComplexMatrix previous = spacing * hermite_phase_sum(gamma, cutoff, -y_max, spacing, intervals + 1);
  for (int level = 1; intervals * 2 + 1 <= max_points; ++level) {
    // Nested refinement: keep the old samples, add the midpoints.
    spacing /= 2.0;
    ComplexMatrix current =
        previous / 2.0 + spacing * hermite_phase_sum(gamma, cutoff, -y_max + spacing, 2.0 * spacing, intervals);
    intervals *= 2;
    const Real change = (current - previous).cwiseAbs().maxCoeff();
    if (level >= 3 && change <= tolerance) return {std::move(current), intervals + 1, change};
    previous = std::move(current);
  }
  throw std::runtime_error("Cubic-phase quadrature did not converge to " + std::to_string(tolerance) +
                           " within " + std::to_string(max_points) + " samples");
}

ComplexMatrix cubic_phase_truncated_generator(Real gamma, int cutoff) {
  if (!std::isfinite(gamma) || cutoff < 1) throw PhysicsError("Invalid cubic-phase parameters");
  RealMatrix q = RealMatrix::Zero(cutoff, cutoff);
  for (int n = 1; n < cutoff; ++n) {
    q(n - 1, n) = std::sqrt(Real(n));
    q(n, n - 1) = std::sqrt(Real(n));
  }
  const RealMatrix cube = q * q * q;
  const Eigen::SelfAdjointEigenSolver<RealMatrix> solver(cube);
  const RealMatrix& vectors = solver.eigenvectors();
  ComplexVector phases(cutoff);
  for (int n = 0; n < cutoff; ++n) phases(n) = std::polar(1.0, gamma * solver.eigenvalues()(n) / 6.0);
  const ComplexMatrix v = vectors.cast<Complex>();
  return v * phases.asDiagonal() * v.transpose();
}

}  // namespace photographiq::fock
