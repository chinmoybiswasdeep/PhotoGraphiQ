// Pure-state and density-matrix kernels on a total-photon-truncated Fock space.
//
// States are indexed by FockBasis(modes, cutoff) ranks. A local operator on the ordered
// modes `local` is a FockBasis(local.size(), cutoff) matrix; on a fibre whose other modes
// hold s photons it acts through its leading block for cutoff - s (P U P, identical to
// Piquasso's truncation semantics). No global operator is ever formed: a k-mode gate
// costs sum_f dim_f^2 per state vector (and twice that per density-matrix column).
// Kernels are parallelised over fibres with OpenMP when compiled with PGQ_HAVE_OPENMP.
#pragma once

#include <span>

#include "photographiq/core/types.hpp"
#include "photographiq/fock/basis.hpp"

namespace photographiq::fock {

using RowComplexMatrix = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using StateRef = Eigen::Ref<ComplexVector>;
using ConstStateRef = Eigen::Ref<const ComplexVector>;
using DensityRef = Eigen::Ref<RowComplexMatrix>;
using ConstDensityRef = Eigen::Ref<const RowComplexMatrix>;
using OperatorRef = Eigen::Ref<const ComplexMatrix>;  // F-ordered arrays bind without a copy

// psi -> P U P psi (in place).
void apply_local(StateRef state, const FockBasis& basis, std::span<const int> local,
                 const OperatorRef& op);
// rho -> (P U P) rho (P U P)^dag (in place, square rho).
void apply_local(DensityRef rho, const FockBasis& basis, std::span<const int> local,
                 const OperatorRef& op);

// Diagonal gate exp(i theta(local occupation)); `phases` holds theta per local rank
// (length FockBasis(local.size(), cutoff).size()). Rotation, Kerr and cross-Kerr.
void apply_local_phase(StateRef state, const FockBasis& basis, std::span<const int> local,
                       const RealVector& phases);
void apply_local_phase(DensityRef rho, const FockBasis& basis, std::span<const int> local,
                       const RealVector& phases);

// Partial trace onto the ordered modes `keep`, indexed by FockBasis(keep.size(), cutoff).
ComplexMatrix reduced_density(ConstStateRef state, const FockBasis& basis, std::span<const int> keep);
ComplexMatrix reduced_density(ConstDensityRef rho, const FockBasis& basis, std::span<const int> keep);

// Joint photon-number distribution of the ordered modes `keep` (unnormalised: sums to the
// state norm / trace), indexed by FockBasis(keep.size(), cutoff).
RealVector photon_marginal(ConstStateRef state, const FockBasis& basis, std::span<const int> keep);
RealVector photon_marginal(ConstDensityRef rho, const FockBasis& basis, std::span<const int> keep);

// Thermal attenuator on `mode`, exactly the decomposition used by mixed_fock.py:
// pure loss with eta = transmissivity / gain, Kraus
//   K_l = sum_n sqrt(C(n, l) (1 - eta)^l eta^(n - l)) |n - l><n|   (l = 0..cutoff-1),
// then, if gain = 1 + (1 - transmissivity) n_th > 1, the quantum-limited amplifier
//   A_k = sum_n sqrt(C(n + k, k) (gain - 1)^k / gain^(n + k + 1)) |n + k><n|
// restricted to total photon number < cutoff (the discarded weight shows up as trace loss).
RowComplexMatrix thermal_loss(ConstDensityRef rho, const FockBasis& basis, int mode,
                              Real transmissivity, Real thermal_photons);

struct Physicality {
  Real hermiticity;     // max |rho - rho^dag|
  Complex trace;
  Real min_eigenvalue;  // of (rho + rho^dag) / 2
  Real max_eigenvalue;
};
Physicality physicality(ConstDensityRef rho);

// Number of OpenMP threads kernels may use (1 without OpenMP).
int max_threads();

}  // namespace photographiq::fock
