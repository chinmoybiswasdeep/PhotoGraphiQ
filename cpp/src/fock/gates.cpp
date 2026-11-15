#include "photographiq/fock/gates.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>

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

// <m|D(alpha)|n>, m < rows, n < cols, along each diagonal m - n = j from the normalised
// Laguerre recurrence in the degree, e_n = <n + j|D|n> = sqrt(n!/(n + j)!) alpha^j
// e^{-x/2} L_n^(j)(x), x = |alpha|^2:  e_{n+1} = ((2n + 1 + j - x) e_n -
// sqrt(n (n + j)) e_{n-1}) / sqrt((n + 1)(n + j + 1)), and <n|D|n + j> = (-1)^j conj(e_n).
// The polynomial is never the minimal solution of its recurrence, so rounding does not
// grow: 4e-14 for |beta| <= 9.4 at size 541 (docs/validation/native-fock-gaussian-unitary.md).
ComplexMatrix displacement_matrix(Complex alpha, int rows, int cols) {
  const Real x = std::norm(alpha);
  const Real log_abs = x > 0.0 ? 0.5 * std::log(x) : 0.0;
  const Real phase = std::arg(alpha);
  const int size = std::max(rows, cols);
  const int small = std::min(rows, cols);  // every stored element has min(m, n) < small
  ComplexMatrix d = ComplexMatrix::Zero(rows, cols);
  for (int j = 0; j < size; ++j) {
    if (x == 0.0 && j > 0) break;
    const Real log_e0 = j * log_abs - 0.5 * x - 0.5 * std::lgamma(j + 1.0);
    Complex previous = std::polar(std::exp(log_e0), j * phase);
    Complex current = (1.0 + j - x) * previous / std::sqrt(j + 1.0);
    const Real sign = j % 2 == 0 ? 1.0 : -1.0;
    for (int n = 0; n < small && n + j < size; ++n) {
      const Complex value = n == 0 ? previous : current;
      if (n + j < rows) d(n + j, n) = value;
      if (j > 0 && n + j < cols) d(n, n + j) = sign * std::conj(value);
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

// <m|S_r|n>, m < rows, n < cols, for the Gaussian unitary of diag(e^-r, e^r) (any real r),
// with the Bargmann phase convention (<0|S_r|0> = sech(r)^(1/2) > 0). With t = tanh r,
// parity p, beta = p - 1/2, m = 2(nu + j) + p >= n = 2 nu + p, the Bargmann sum is a
// Jacobi polynomial (SU(1,1) analogue of the Laguerre form of D):
//   e_nu = <m|S_r|n> = sech(r)^(p + 1/2) (-t)^j sqrt(nu! G(nu + j + beta + 1) /
//          ((nu + j)! G(nu + beta + 1))) P_nu^(j, beta)(1 - 2 t^2),   <n|S_r|m> = (-1)^j e_nu,
// computed by the three-term recurrence of P in nu, rescaled to avoid under/overflow.
// The argument lies in (-1, 1] and the polynomial is never the minimal solution, so
// rounding stays at 2e-14 for |r| = 1 at size 600 (rows < 100), where the Bargmann
// recurrence loses 7e-5 (docs/validation/native-fock-gaussian-unitary.md).
RealMatrix squeezing_matrix(Real r, int rows, int cols) {
  const int size = std::max(rows, cols);
  const int small = std::min(rows, cols);
  RealMatrix out = RealMatrix::Zero(rows, cols);
  const Real t = std::tanh(r);
  const Real y = 1.0 - 2.0 * t * t;
  const Real log_sech = -std::log(std::cosh(r));
  const Real log_t = t != 0.0 ? std::log(std::abs(t)) : 0.0;
  constexpr Real kHuge = 1e150;
  for (int p = 0; p < 2; ++p) {
    const Real b = p - 0.5;
    for (int j = 0; 2 * j + p < size; ++j) {
      if (t == 0.0 && j > 0) break;
      const Real a = j;
      // N_{nu+1} / N_nu
      auto ratio = [&](int nu) {
        return std::sqrt((nu + 1.0) * (nu + a + b + 1.0) / ((nu + a + 1.0) * (nu + b + 1.0)));
      };
      Real log_scale = (p + 0.5) * log_sech + j * log_t +
                       0.5 * (std::lgamma(a + b + 1.0) - std::lgamma(a + 1.0) - std::lgamma(b + 1.0));
      const Real sign0 = (t > 0.0 && j % 2 == 1) ? -1.0 : 1.0;  // sign of (-t)^j
      Real previous = 1.0;                                        // e_0 / exp(log_scale)
      Real current = ratio(0) * ((a + 1.0) + (a + b + 2.0) * (y - 1.0) / 2.0);
      for (int nu = 0; 2 * nu + p < small && 2 * (nu + j) + p < size; ++nu) {
        const Real value = sign0 * (nu == 0 ? previous : current) * std::exp(log_scale);
        const int m = 2 * (nu + j) + p;
        const int n = 2 * nu + p;
        if (m < rows) out(m, n) = value;
        if (j > 0 && m < cols) out(n, m) = j % 2 == 0 ? value : -value;
        if (nu >= 1) {
          const Real c = 2.0 * nu + a + b;
          const Real denominator = 2.0 * (nu + 1.0) * (nu + a + b + 1.0) * c;
          const Real forward = (c + 1.0) * ((c + 2.0) * c * y + a * a - b * b) / denominator;
          const Real back = 2.0 * (nu + a) * (nu + b) * (c + 2.0) / denominator;
          const Real next = ratio(nu) * (forward * current - ratio(nu - 1) * back * previous);
          previous = current;
          current = next;
          if (std::abs(current) > kHuge) {
            previous /= kHuge;
            current /= kHuge;
            log_scale += std::log(kHuge);
          }
        }
      }
    }
  }
  return out;
}

// Smallest L >= cutoff such that sum_{l >= L} |<l|D(beta)|n>|^2 <= tol^2 for every n < cutoff.
// Majorant: |<l|D|n>| <= M_ln = e^{-x/2} sqrt(l! n!) sum_k |beta|^(l+n-2k) / (k!(l-k)!(n-k)!)
// (absolute values in the Bargmann sum), which obeys the positive recurrence
// M_{l+1,n} = (|beta| M_ln + sqrt(n) M_{l,n-1}) / sqrt(l + 1). Termwise, M_{l+1,n} <= q_l M_ln
// with q_l = |beta| sqrt(l + 1) / (l + 1 - n), decreasing in l, so once q_L < 1 the tail is
// at most M_Ln / sqrt(1 - q_L^2). Rows are rescaled (log scale) against over/underflow.
int displacement_tail_cutoff(Real beta, int cutoff, Real tol) {
  if (beta == 0.0) return cutoff;
  std::vector<Real> row(static_cast<std::size_t>(cutoff));
  std::vector<Real> next(row.size());
  Real log_scale = -std::numeric_limits<Real>::infinity();
  for (int n = 0; n < cutoff; ++n) {
    row[n] = -0.5 * beta * beta + n * std::log(beta) - 0.5 * std::lgamma(n + 1.0);
    log_scale = std::max(log_scale, row[n]);
  }
  for (Real& v : row) v = std::exp(v - log_scale);
  const Real log_tol = std::log(tol);
  for (int l = 0;; ++l) {
    if (l >= cutoff) {
      const Real q = beta * std::sqrt(l + 1.0) / (l + 2.0 - cutoff);
      const Real largest = *std::max_element(row.begin(), row.end());
      if (q < 1.0 && std::log(largest) + log_scale - 0.5 * std::log1p(-q * q) <= log_tol) return l;
    }
    Real largest = 0.0;
    for (int n = 0; n < cutoff; ++n) {
      next[n] = (beta * row[n] + (n > 0 ? std::sqrt(Real(n)) * row[n - 1] : 0.0)) / std::sqrt(l + 1.0);
      largest = std::max(largest, next[n]);
    }
    for (int n = 0; n < cutoff; ++n) row[n] = next[n] / largest;
    log_scale += std::log(largest);
  }
}

// Single-mode U with U^dag x U = S x + d as U = R(phi1) S_r R(phi2) D(beta), from the
// decomposition S = rotation(phi1) diag(e^-r, e^r) rotation(phi2) (SVD) and d_beta = S^-1 d.
// Rotations are exact phases exp(i phi n); D(beta) is needed on L > cutoff rows, with L
// chosen by displacement_tail_cutoff so that the discarded sum
// sum_{l >= L} <m|R S_r R|l><l|D|n> is at most ||P_{>=L} D|n>|| <= 1e-16 (Cauchy-Schwarz,
// rows of the unitary R S_r R have norm <= 1). Same operator and global phase as the
// Bargmann form: D(delta) U_S = U_S D(beta) exactly, and both have <0|U_S|0> > 0.
ComplexMatrix single_mode_unitary(const RealMatrix& symplectic, const RealVector& displacement,
                                  int cutoff) {
  if (displacement.isZero(0.0) && symplectic == RealMatrix::Identity(2, 2)) {
    return ComplexMatrix::Identity(cutoff, cutoff);
  }
  if (symplectic == RealMatrix::Identity(2, 2)) {
    return displacement_matrix(Complex(displacement(0), displacement(1)) / 2.0, cutoff, cutoff);
  }
  const Eigen::JacobiSVD<RealMatrix> svd(symplectic, Eigen::ComputeFullU | Eigen::ComputeFullV);
  RealMatrix u = svd.matrixU();
  RealMatrix v = svd.matrixV();
  if (u.determinant() < 0.0) {  // S = U Sigma V^T is unchanged by flipping a column of both
    u.col(1) *= -1.0;
    v.col(1) *= -1.0;
  }
  const RealVector sigma = svd.singularValues();  // sigma_0 >= sigma_1, sigma_0 sigma_1 = 1
  const Real r = 0.5 * (std::log(sigma(1)) - std::log(sigma(0)));  // diag(e^-r, e^r), r <= 0
  const Real phi1 = std::atan2(u(1, 0), u(0, 0));
  const Real phi2 = std::atan2(v(0, 1), v(0, 0));  // V^T = rotation(phi2)
  Complex beta(0.0, 0.0);
  if (!displacement.isZero(0.0)) {
    const RealVector shift = symplectic.partialPivLu().solve(displacement);
    beta = Complex(shift(0), shift(1)) / 2.0;
  }
  const int inner = r == 0.0 ? cutoff : displacement_tail_cutoff(std::abs(beta), cutoff, 1e-16);
  const RealMatrix sq = squeezing_matrix(r, cutoff, inner);
  ComplexVector left(cutoff);
  ComplexVector right(inner);
  for (int m = 0; m < cutoff; ++m) left(m) = std::polar(1.0, phi1 * m);
  for (int l = 0; l < inner; ++l) right(l) = std::polar(1.0, phi2 * l);
  if (beta == Complex(0.0, 0.0)) return left.asDiagonal() * sq.cast<Complex>() * right.asDiagonal();
  const ComplexMatrix shifted = right.asDiagonal() * displacement_matrix(beta, inner, cutoff);
  ComplexMatrix product(cutoff, cutoff);  // real S_r times complex: two real products
  product.real() = sq * shifted.real();
  product.imag() = sq * shifted.imag();
  return left.asDiagonal() * product;
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
  if (k == 1) return single_mode_unitary(symplectic, displacement, cutoff);
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
