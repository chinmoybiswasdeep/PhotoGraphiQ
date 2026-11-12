// Fock-basis matrices of local gates on the truncated space of k modes.
//
// Rows/columns are indexed by FockBasis(k, cutoff) ranks (total photon number < cutoff).
// Conventions follow photographiq/core/types.hpp (hbar = 2, q = a + a^dag,
// p = -i (a - a^dag), interleaved (q0, p0, q1, p1, ...)).
#pragma once

#include "photographiq/core/types.hpp"

namespace photographiq::fock {

// EXACT matrix elements <m|U|n> (|m|, |n| < cutoff) of the Gaussian unitary U with
// Heisenberg action U^dag x U = S x + d (S real symplectic 2k x 2k, d real 2k).
//
// Method: Bargmann generating function. With U^dag a U = T a + R a^dag + delta,
//   <0|e^{z.a} U e^{w.a^dag}|0> = C exp(z.A_zz z/2 + z.A_zw w + w.A_ww w/2 + b_z.z + b_w.w),
//   A_zw = (T^dag)^-1, A_zz = R conj(T)^-1, A_ww = -R^dag (T^dag)^-1,
//   b_w = -conj(T)^-1 conj(delta), b_z = delta + R b_w,
//   C = |det T|^(-1/2) exp(-|delta|^2/2 + conj(delta).A_zz conj(delta)/2),
// and the matrix elements obey G_{k+e_i} = (b_i G_k + sum_j A_ij sqrt(k_j) G_{k-e_j}) /
// sqrt(k_i + 1) over the joint index (m, n), pivoting on the most occupied mode of the
// side (bra or ket) with more photons, which keeps rounding at the 1e-12 level at
// cutoffs ~100 (always lowering the bra amplified it to O(1)). A pure single-mode
// displacement (S = 1) instead uses the Laguerre recurrence along each diagonal, stable
// at 1e-15 for |alpha| <= 7, cutoff <= 120; the Bargmann recurrence with displacement and
// squeezing combined still loses accuracy for large |alpha| and cutoff (7.5e-5 at
// |alpha| = 4, r = 0.3, cutoff 80), column 0 (Gaussian inputs) excepted (1e-16).
// Only finitely many terms are involved, so
// every returned element is exact up to rounding: truncation enters solely through the
// projector P (the returned matrix is P U P, not unitary when U leaks above the cutoff).
// Global phase: <0|U|0> has the phase of exp(conj(delta).A_zz conj(delta)/2) times a
// positive real number, i.e. the displacement-free part fixes the vacuum amplitude to be
// real positive (passive gates act as exp(i a^dag H a)). Physical predictions never
// depend on this phase. Throws PhysicsError if S is not symplectic (residual > 1e-9).
ComplexMatrix gaussian_unitary(const RealMatrix& symplectic, const RealVector& displacement,
                               int cutoff);

// Matrix elements <m|exp(i gamma q^3 / 6)|n>, m, n < cutoff, computed EXACTLY up to a
// controlled quadrature error as integrals of orthonormal Hermite functions,
//   <m|V|n> = int h_m(y) h_n(y) exp(i gamma sqrt(2) y^3 / 3) dy     (q = sqrt(2) y).
// The integrand is analytic and below exp(-80) outside |y| <= sqrt(2 cutoff + 1) + 10, so
// the trapezoidal rule on that window converges spectrally; the step is halved (reusing
// all previous samples) until two successive estimates agree to `tolerance` in max-abs
// (at least three halvings). The result is the projected operator P V P. Throws
// std::runtime_error if `max_points` samples do not suffice (very large |gamma| cutoff^1.5).
struct CubicPhaseMatrix {
  ComplexMatrix matrix;
  Index points;  // samples of the accepted rule
  Real change;   // max |difference| to the previous rule
};
CubicPhaseMatrix cubic_phase_projected(Real gamma, int cutoff, Real tolerance = 1e-13,
                                       Index max_points = Index{1} << 22);

// Piquasso's definition: expm(i gamma Q^3 / 6) with Q the cutoff x cutoff truncation of
// q = a + a^dag (a truncated-generator approximation that is exactly unitary on the
// truncated space, so it never reports norm leakage). Provided for cross-checking.
ComplexMatrix cubic_phase_truncated_generator(Real gamma, int cutoff);

}  // namespace photographiq::fock
