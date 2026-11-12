// Matrix functions of (Gaussian) boson sampling: permanent, hafnian, loop hafnian,
// torontonian and loop torontonian, in complex double precision.
//
// Algorithms (the Python package photographiq.sampling.linalg documents the same formulas):
//   * permanent_repeated: Ryser's formula with multiplicities,
//       Perm = (-1)^N sum_{0<=x_j<=c_j} (-1)^{|x|} prod_j C(c_j, x_j) prod_i (sum_j x_j M_ij)^{r_i},
//     enumerated in reflected mixed-radix Gray order (Knuth 7.2.1.1, Algorithm H) so each
//     step updates the row sums by one column; prod_j (c_j + 1) terms. The cheaper of the
//     row/column enumerations is chosen (Perm(M) = Perm(M^T)). All-ones multiplicities give
//     the classic Gray-code Ryser algorithm, O(2^n n).
//   * hafnian / loop_hafnian: Glynn-type power-trace formula (Bjorklund, Gupt, Quesada
//     2019; finite-difference sieve as in Bulmer et al. 2022) with the loop correction of
//     Bjorklund, Cygan, Koivisto, Kaski: for a 2n x 2n symmetric A,
//       haf(A) = 2^{1-n} sum_{x in {+-1}^n, x_0 = 1} (prod_p x_p) [eta^n] exp(sum_k g_k eta^k),
//       g_k = tr(C^k)/(2k) + v^T X_x C^{k-1} v / 2,  C = A X_x (diagonal of A zeroed),
//     X_x swapping the index pairs (2p, 2p+1) with weight x_p and v the loop weights
//     (diagonal of A; v = 0 for the plain hafnian). Power traces come from explicit matrix
//     powers (ceil(n/2) products per term): O(n^4 2^n) operations. Compared with the
//     subset/eigenvalue variant this is ~1e3 times more accurate (measured relative error
//     ~1e-15 vs ~1e-11 at 2n = 20..24 against 40-digit references). An odd-dimensional
//     loop hafnian is padded with an isolated vertex of loop weight 1.
//   * torontonian / loop_torontonian: for a 2n x 2n matrix O in block (a, a^dag) ordering,
//       Tor(O)     = sum_{Z subset [n]} (-1)^{n-|Z|} / sqrt(det(I - O_Z)),
//       ltor(O, g) = sum_Z (-1)^{n-|Z|} exp(g_Z^T (I - O_Z)^{-1} g_Z^* / 2) / sqrt(det(I - O_Z)),
//     where O_Z keeps rows/columns {i, i + n : i in Z}; principal square root; LU per subset.
//
// Parallelism: subsets (or outer Gray-code digits) are split into a fixed number of
// chunks that does not depend on the thread count; chunk partial sums are added in a
// fixed order with Neumaier compensation, so results are bitwise reproducible for any
// number of OpenMP threads.
#pragma once

#include <utility>
#include <vector>

#include "photographiq/core/types.hpp"

namespace photographiq::sampling {

// Permanent of a square matrix (Gray-code Ryser).
Complex permanent(const ComplexMatrix& matrix);

// Permanent of the N x N matrix formed by repeating row i row_multiplicity[i] times and
// column j column_multiplicity[j] times (N = sum of either); `matrix` may be rectangular.
Complex permanent_repeated(const ComplexMatrix& matrix, const std::vector<int>& row_multiplicity,
                           const std::vector<int>& column_multiplicity);

// Hafnian of a symmetric matrix (diagonal ignored); zero for odd dimension.
Complex hafnian(const ComplexMatrix& matrix);

// Loop hafnian of a symmetric matrix whose diagonal holds the loop weights.
Complex loop_hafnian(const ComplexMatrix& matrix);

// Torontonian of a 2n x 2n matrix in block (a, a^dag) ordering.
Complex torontonian(const ComplexMatrix& matrix);

// Loop torontonian of a 2n x 2n matrix and a 2n vector in block (a, a^dag) ordering.
Complex loop_torontonian(const ComplexMatrix& matrix, const ComplexVector& gamma);

// Threshold-detector inclusion-exclusion over vacuum projections, real deviation form
// (hbar = 2; photographiq.sampling.gbs documents the derivation):
//   [k = 0] + sum_{T subset [k]} (-1)^{|T|} w_T,
//   w_T = expm1(-sum log1p(eig E_T) / 2 - e_T^T (2 (I + E_T))^{-1} e_T / 2),
// with E (2k x 2k symmetric, I + E positive definite) the dark-mode-conditioned covariance
// excess and e the conditioned mean; E_T keeps quadrature pairs (2i, 2i+1), i in T.
// Returns {sum, [k = 0] + sum_T |w_T|}; the second value times machine epsilon bounds the
// rounding error of the alternating sum in all tested cases.
std::pair<Real, Real> threshold_vacuum_sum(const RealMatrix& excess, const RealVector& shift);

// OpenMP thread control (1 and no-op without OpenMP).
int max_threads();
void set_num_threads(int threads);

}  // namespace photographiq::sampling
