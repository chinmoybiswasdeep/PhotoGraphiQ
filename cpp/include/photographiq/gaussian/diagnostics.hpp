// Information-theoretic diagnostics of Gaussian states (semantics of states.GaussianState).
#pragma once

#include "photographiq/gaussian/operations.hpp"

namespace photographiq::gaussian {

// Sorted symplectic eigenvalues (vacuum 1). With V = L L^T, i L^T Omega L is Hermitian
// and isospectral to i Omega V (eigenvalues +-nu_k); one of each pair is kept. A covariance
// that is not positive definite falls back to the general eigenvalues of Omega V.
RealVector symplectic_eigenvalues(ConstCovRef cov);

// Tr(rho^2) = |det V|^(-1/2).
Real purity(ConstCovRef cov);

// Von Neumann entropy in nats, sum_k g(nu_k); eigenvalues within `tolerance` below 1 count
// as 1, smaller ones throw PhysicsError.
Real entropy(ConstCovRef cov, Real tolerance = 1e-7);

// Base-2 logarithmic negativity of (modes | rest): p flipped on `modes`.
Real log_negativity(ConstCovRef cov, const ModeList& modes);

// Tr(rho sigma) = 2^n exp(-d^T (V1+V2)^-1 d / 2) / sqrt(det(V1+V2)).
Real overlap(ConstMeanRef mean1, ConstCovRef cov1, ConstMeanRef mean2, ConstCovRef cov2);

// Squared Uhlmann fidelity. If either state is pure (all |nu_k - 1| < pure_tolerance) the
// exact identity F = Tr(rho sigma) is used. Otherwise the Banchi-Braunstein-Pirandola formula
// (PRL 115, 260501) in units with vacuum V/2, evaluated spectrally:
//   det[2 (sqrt(1 + (Vaux Omega)^-2 / 4) + 1) Vaux] = det(2 Vaux) prod_j (1 + sqrt(1 + 1/(4 mu_j^2)))
// with mu_j the eigenvalues of Vaux Omega, so no matrix square root is needed. Near purity
// 1 + 1/(4 mu^2) cancels, so the relative error grows to about sqrt(eps) * exp(2 r) for
// squeezing r (observed ~2e-7 at r = 1.5 and ~4e-6 at r = 3 for nu - 1 <= 1e-5; ~1e-15 for
// unsqueezed, clearly mixed states).
Real fidelity(ConstMeanRef mean1, ConstCovRef cov1, ConstMeanRef mean2, ConstCovRef cov2,
              Real pure_tolerance = 1e-9);

}  // namespace photographiq::gaussian
