// Unit tests of photographiq::gaussian against closed forms and dense reference algebra.
#if PGQ_WITH_GAUSSIAN

#include <cmath>
#include <cstdio>
#include <numbers>
#include <random>
#include <string>

#include "photographiq/core/symplectic.hpp"
#include "photographiq/gaussian/batch.hpp"
#include "photographiq/gaussian/conditioning.hpp"
#include "photographiq/gaussian/diagnostics.hpp"
#include "photographiq/gaussian/operations.hpp"

namespace g = photographiq::gaussian;
using photographiq::Index;
using photographiq::PhysicsError;
using photographiq::Real;
using photographiq::RealMatrix;
using photographiq::RealVector;

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
  if (!condition) {
    ++failures;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

void close(Real actual, Real expected, Real tol, const std::string& what) {
  if (!(std::abs(actual - expected) <= tol)) {
    ++failures;
    std::printf("FAIL: %s: %.17g vs %.17g (tol %.1e)\n", what.c_str(), actual, expected, tol);
  }
}

template <class F>
void throws(F&& f, const std::string& what) {
  try {
    f();
  } catch (const PhysicsError&) {
    return;
  }
  ++failures;
  std::printf("FAIL: expected PhysicsError: %s\n", what.c_str());
}

RealMatrix rotation(Real phi) {
  RealMatrix r(2, 2);
  r << std::cos(phi), -std::sin(phi), std::sin(phi), std::cos(phi);
  return r;
}

RealMatrix squeezing(Real r) {
  RealMatrix s = RealMatrix::Zero(2, 2);
  s(0, 0) = std::exp(-r);
  s(1, 1) = std::exp(r);
  return s;
}

RealMatrix beamsplitter(Real theta) {
  const Real c = std::cos(theta);
  const Real s = std::sin(theta);
  RealMatrix b(4, 4);
  b << c, 0, -s, 0, 0, c, 0, -s, s, 0, c, 0, 0, s, 0, c;
  return b;
}

RealMatrix cz(Real weight) {
  RealMatrix m = RealMatrix::Identity(4, 4);
  m(1, 2) = weight;
  m(3, 0) = weight;
  return m;
}

RealMatrix embed(const RealMatrix& local, const g::ModeList& modes, Index n) {
  RealMatrix full = RealMatrix::Identity(2 * n, 2 * n);
  const std::vector<Index> idx = g::quadrature_indices(modes, n);
  for (std::size_t a = 0; a < idx.size(); ++a) {
    for (std::size_t b = 0; b < idx.size(); ++b) {
      full(idx[a], idx[b]) = local(static_cast<Index>(a), static_cast<Index>(b));
    }
  }
  return full;
}

Real thermal_entropy(Real nbar) {
  return (nbar + 1) * std::log(nbar + 1) - (nbar > 0 ? nbar * std::log(nbar) : 0.0);
}

// Two-mode squeezed vacuum from q- and p-squeezed inputs on a 50:50 beam splitter.
void tmsv(Real r, RealVector& mean, RealMatrix& cov) {
  mean = RealVector::Zero(4);
  cov = RealMatrix::Identity(4, 4);
  g::apply_symplectic(mean, cov, {0}, squeezing(r));
  g::apply_symplectic(mean, cov, {1}, squeezing(-r));
  g::apply_symplectic(mean, cov, {0, 1}, beamsplitter(std::numbers::pi / 4));
}

void test_local_updates_match_dense_algebra() {
  const Index n = 5;
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<Real> u(-1.0, 1.0);
  RealVector mean(2 * n);
  for (Index i = 0; i < 2 * n; ++i) mean(i) = u(rng);
  RealMatrix cov = RealMatrix::Identity(2 * n, 2 * n);
  RealVector ref_mean = mean;
  RealMatrix ref_cov = cov;
  for (int step = 0; step < 40; ++step) {
    const Index a = static_cast<Index>(rng() % n);
    const Index b = (a + 1 + static_cast<Index>(rng() % (n - 1))) % n;
    RealMatrix local;
    g::ModeList modes;
    switch (step % 4) {
      case 0: local = rotation(3 * u(rng)), modes = {a}; break;
      case 1: local = squeezing(u(rng)), modes = {a}; break;
      case 2: local = beamsplitter(3 * u(rng)), modes = {a, b}; break;
      default: local = cz(u(rng)), modes = {b, a}; break;
    }
    g::apply_symplectic(mean, cov, modes, local);
    const RealMatrix full = embed(local, modes, n);
    ref_mean = full * ref_mean;
    ref_cov = full * ref_cov * full.transpose();
    ref_cov = 0.5 * (ref_cov + ref_cov.transpose()).eval();
  }
  const Real scale = ref_cov.cwiseAbs().maxCoeff();
  close((cov - ref_cov).cwiseAbs().maxCoeff() / scale, 0.0, 1e-13, "local vs dense covariance");
  close((mean - ref_mean).cwiseAbs().maxCoeff(), 0.0, 1e-12, "local vs dense mean");
  check((cov - cov.transpose()).cwiseAbs().maxCoeff() == 0.0, "exact symmetry after updates");
  // Pure state stays pure: all symplectic eigenvalues 1.
  const RealVector nu = g::symplectic_eigenvalues(cov);
  close((nu.array() - 1.0).abs().maxCoeff(), 0.0, 1e-9, "symplectic eigenvalues preserved");
  close(g::purity(cov), 1.0, 1e-10, "purity preserved");
}

void test_symplecticity_checks() {
  check(g::is_symplectic(beamsplitter(0.3) * cz(0.7)), "BS * CZ symplectic");
  check(!g::is_symplectic(2.0 * RealMatrix::Identity(2, 2)), "2I not symplectic");
  RealVector mean = RealVector::Zero(4);
  RealMatrix cov = RealMatrix::Identity(4, 4);
  throws([&] { g::apply_symplectic(mean, cov, {0}, 2.0 * RealMatrix::Identity(2, 2)); },
         "non-symplectic gate");
  throws([&] { g::apply_symplectic(mean, cov, {0, 0}, cz(1.0)); }, "repeated modes");
  throws([&] { g::apply_symplectic(mean, cov, {2}, rotation(0.1)); }, "mode out of range");
  // Phase-insensitive amplifier without added noise violates complete positivity.
  throws([&] {
    g::apply_channel(mean, cov, {0}, 2.0 * RealMatrix::Identity(2, 2), RealMatrix::Zero(2, 2));
  }, "noiseless amplifier");
  // Quantum-limited amplifier (gain G, Y = (G - 1) I) is allowed.
  const Real gain = 2.5;
  g::apply_channel(mean, cov, {0}, std::sqrt(gain) * RealMatrix::Identity(2, 2),
                   (gain - 1.0) * RealMatrix::Identity(2, 2));
  close(cov(0, 0), 2 * gain - 1, 1e-14, "amplified vacuum variance");
}

void test_vacuum_and_thermal() {
  const RealMatrix vac = RealMatrix::Identity(6, 6);
  close(g::symplectic_eigenvalues(vac).maxCoeff(), 1.0, 1e-14, "vacuum nu");
  close(g::entropy(vac), 0.0, 1e-14, "vacuum entropy");
  close(g::purity(vac), 1.0, 1e-14, "vacuum purity");
  const Real nbar = 0.8;
  const RealMatrix thermal = (2 * nbar + 1) * RealMatrix::Identity(2, 2);
  close(g::entropy(thermal), thermal_entropy(nbar), 1e-13, "thermal entropy");
  close(g::purity(thermal), 1.0 / (2 * nbar + 1), 1e-14, "thermal purity");
  // Loss of a coherent state with thermal environment.
  RealVector mean(2);
  mean << 1.2, -0.4;
  RealMatrix cov = RealMatrix::Identity(2, 2);
  g::apply_loss(mean, cov, 0, 0.36, 0.5);
  close(mean(0), 0.6 * 1.2, 1e-15, "lossy mean");
  close(cov(0, 0), 0.36 + 0.64 * 2.0, 1e-15, "lossy variance");
  throws([&] { g::apply_loss(mean, cov, 0, 1.5, 0.0); }, "eta > 1");
  throws([&] { (void)g::entropy(0.5 * RealMatrix::Identity(2, 2)); }, "unphysical entropy");
}

void test_tmsv_closed_forms() {
  const Real r = 0.9;
  RealVector mean;
  RealMatrix cov;
  tmsv(r, mean, cov);
  close(g::log_negativity(cov, {0}), 2 * r / std::log(2.0), 1e-12, "TMSV log negativity");
  const g::Moments reduced = g::remove_mode(mean, cov, 1);
  close(g::entropy(reduced.covariance), thermal_entropy(std::sinh(r) * std::sinh(r)), 1e-12,
        "TMSV reduced entropy");
  // Ideal q homodyne of mode 0: partner (q, p) variances 1/cosh(2r), cosh(2r).
  const g::Conditioned out = g::condition_homodyne(mean, cov, 0, 0.0, 1.0, 0.0, 0.3);
  close(out.covariance(0, 0), 1.0 / std::cosh(2 * r), 1e-13, "conditional q variance");
  close(out.covariance(1, 1), std::cosh(2 * r), 1e-12, "conditional p variance");
  const Real var = std::cosh(2 * r);
  close(out.density, std::exp(-0.09 / (2 * var)) / std::sqrt(2 * std::numbers::pi * var), 1e-15,
        "homodyne density");
  close(out.mean(0), -std::tanh(2 * r) * 0.3, 1e-14, "conditional mean (q correlation)");
  // Inefficient detection adds calibrated variance (1-eta)/eta + noise.
  const g::HomodyneMoments noisy = g::homodyne_moments(mean, cov, 0, 0.0, 0.8, 0.1);
  close(noisy.variance, var + 0.25 + 0.1, 1e-13, "inefficient homodyne variance");
}

void test_heterodyne_and_generaldyne() {
  RealVector mean(2);
  mean << 0.5, -1.0;
  const RealMatrix cov = RealMatrix::Identity(2, 2);
  const Eigen::Vector2d y(0.5, -1.0);
  const g::Conditioned het = g::condition_generaldyne(mean, cov, 0, Eigen::Matrix2d::Identity(), y);
  close(het.density, 1.0 / (4 * std::numbers::pi), 1e-15, "heterodyne peak density");
  // Two-mode: generaldyne with a squeezed seed approaches homodyne as the seed squeezes.
  RealVector m2;
  RealMatrix c2;
  tmsv(0.5, m2, c2);
  Eigen::Matrix2d seed = Eigen::Matrix2d::Zero();
  seed(0, 0) = 1e-9;
  seed(1, 1) = 1e9;
  const g::Conditioned gen = g::condition_generaldyne(m2, c2, 0, seed, Eigen::Vector2d(0.3, 0.0));
  const g::Conditioned hom = g::condition_homodyne(m2, c2, 0, 0.0, 1.0, 0.0, 0.3);
  close((gen.covariance - hom.covariance).cwiseAbs().maxCoeff(), 0.0, 1e-8, "generaldyne limit");
  throws([&] {
    Eigen::Matrix2d bad = -2.0 * Eigen::Matrix2d::Identity();
    (void)g::generaldyne_moments(m2, c2, 0, bad);
  }, "indefinite outcome covariance");
}

void test_fidelity_closed_forms() {
  RealVector vac = RealVector::Zero(2);
  RealVector coh(2);
  const Real re = 0.6, im = 0.3;
  coh << 2 * re, 2 * im;
  const RealMatrix id = RealMatrix::Identity(2, 2);
  close(g::fidelity(vac, id, coh, id), std::exp(-(re * re + im * im)), 1e-14, "coherent fidelity");
  RealVector zero = RealVector::Zero(2);
  RealMatrix sq = RealMatrix::Identity(2, 2);
  g::apply_symplectic(zero, sq, {0}, squeezing(0.7));
  close(g::fidelity(vac, id, vac, sq), 1.0 / std::cosh(0.7), 1e-14, "squeezed fidelity");
  const Real n1 = 0.4, n2 = 1.7;
  const RealMatrix t1 = (2 * n1 + 1) * id, t2 = (2 * n2 + 1) * id;
  const Real expected = std::pow(std::sqrt((n1 + 1) * (n2 + 1)) - std::sqrt(n1 * n2), -2.0);
  close(g::fidelity(vac, t1, vac, t2), expected, 1e-12, "thermal fidelity (mixed BBP)");
  close(g::fidelity(vac, t1, vac, t1), 1.0, 1e-12, "self fidelity of a mixed state");
  close(g::overlap(vac, id, vac, id), 1.0, 1e-15, "vacuum overlap");
}

void test_batch_matches_single() {
  const Index batch = 6, n = 3, dim = 2 * n;
  g::RowMatrix means = g::RowMatrix::Zero(batch, dim);
  g::RowMatrix covs(batch, dim * dim);
  g::RowMatrix mats(batch, 16);
  for (Index b = 0; b < batch; ++b) {
    RealMatrix cov = RealMatrix::Identity(dim, dim);
    RealVector mean = RealVector::Constant(dim, 0.1 * static_cast<Real>(b));
    g::apply_symplectic(mean, cov, {0}, squeezing(0.2 * static_cast<Real>(b)));
    means.row(b) = mean.transpose();
    covs.row(b) = Eigen::Map<const RealVector>(cov.data(), dim * dim).transpose();
    const RealMatrix s = beamsplitter(0.3 * static_cast<Real>(b)) * cz(0.5);
    const g::RowMatrix row_major = s;
    mats.row(b) = Eigen::Map<const RealVector>(row_major.data(), 16).transpose();
  }
  const g::RowMatrix means0 = means, covs0 = covs;
  g::batch_apply_symplectic(means, covs, {2, 0}, mats);
  RealVector angles(batch), outcomes(batch);
  for (Index b = 0; b < batch; ++b) angles(b) = 0.4 * static_cast<Real>(b), outcomes(b) = 0.1;
  g::RowMatrix out_means(batch, dim - 2), out_covs(batch, (dim - 2) * (dim - 2));
  RealVector densities(batch);
  g::batch_condition_homodyne(means, covs, 1, angles, 0.9, 0.0, outcomes, out_means, out_covs,
                              densities);
  for (Index b = 0; b < batch; ++b) {
    RealVector mean = means0.row(b).transpose();
    RealMatrix cov = Eigen::Map<const RealMatrix>(covs0.row(b).data(), dim, dim);
    const RealMatrix s = beamsplitter(0.3 * static_cast<Real>(b)) * cz(0.5);
    g::apply_symplectic(mean, cov, {2, 0}, s);
    const g::Conditioned single =
        g::condition_homodyne(mean, cov, 1, angles(b), 0.9, 0.0, outcomes(b));
    const RealMatrix batched = Eigen::Map<const RealMatrix>(out_covs.row(b).data(), dim - 2, dim - 2);
    check((batched - single.covariance).cwiseAbs().maxCoeff() == 0.0, "batch cov bitwise");
    check((out_means.row(b).transpose() - single.mean).cwiseAbs().maxCoeff() == 0.0,
          "batch mean bitwise");
    check(densities(b) == single.density, "batch density bitwise");
  }
  g::RowMatrix bad = mats;
  bad(2, 0) = 5.0;
  throws([&] { g::batch_apply_symplectic(means, covs, {2, 0}, bad); }, "batch non-symplectic");
}

}  // namespace

int main() {
  test_local_updates_match_dense_algebra();
  test_symplecticity_checks();
  test_vacuum_and_thermal();
  test_tmsv_closed_forms();
  test_heterodyne_and_generaldyne();
  test_fidelity_closed_forms();
  test_batch_matches_single();
  {  // zero-mode inputs (state after measuring every mode) must not crash
    const RealVector m0(0);
    const RealMatrix c0(0, 0);
    check(g::symplectic_eigenvalues(c0).size() == 0, "zero-mode symplectic eigenvalues");
    close(g::entropy(c0), 0.0, 0.0, "zero-mode entropy");
    close(g::log_negativity(c0, {}), 0.0, 0.0, "zero-mode log negativity");
    close(g::fidelity(m0, c0, m0, c0), 1.0, 0.0, "zero-mode fidelity");
    RealVector m2 = RealVector::Zero(2);
    RealMatrix c2 = RealMatrix::Identity(2, 2);
    g::apply_channel(m2, c2, {}, RealMatrix(0, 0), RealMatrix(0, 0));
    check(c2.isIdentity(0.0), "empty channel is the identity");
  }
  std::printf("test_gaussian: %d failure(s), OpenMP threads %d\n", failures, g::max_threads());
  return failures == 0 ? 0 : 1;
}

#else
int main() { return 0; }
#endif
