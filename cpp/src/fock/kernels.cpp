#include "photographiq/fock/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <Eigen/Eigenvalues>

#if PGQ_HAVE_OPENMP
#include <omp.h>
#endif

namespace photographiq::fock {
namespace {

// Below this many complex multiply-adds (~2-5 ms serial) a kernel runs serially. Measured
// on a shared, loaded 12-thread laptop: every parallel region can wait one scheduler
// quantum (10-30 ms) for a preempted thread, so the earlier 2^16 threshold made 0.3 ms
// gates take 25 ms; above 2^22 the region is long enough to amortise that.
constexpr std::int64_t kParallelWork = std::int64_t{1} << 22;
constexpr Index kChunk = 64;

void check_state(Index size, const FockBasis& basis) {
  if (size != basis.size()) throw PhysicsError("State size does not match the Fock basis");
}

void check_operator(const OperatorRef& op, const FockBasis& basis, std::span<const int> local) {
  const Index dim = dimension(static_cast<int>(local.size()), basis.cutoff());
  if (op.rows() != dim || op.cols() != dim) {
    throw PhysicsError("Local operator must be square with the local truncated dimension");
  }
}

Index max_fiber(const Fibers& fibers) {
  Index largest = 0;
  for (Index f = 0; f < fibers.count(); ++f) largest = std::max(largest, fibers.size(f));
  return largest;
}

std::int64_t gate_work(const Fibers& fibers) {
  std::int64_t work = 0;
  for (Index f = 0; f < fibers.count(); ++f) work += fibers.size(f) * fibers.size(f);
  return work;
}

// rows idx_f of rho <- U_L rows idx_f   (left = true)
// cols idx_f of rho <- cols idx_f U_L^dag (left = false)
void multiply_fibres(DensityRef rho, const Fibers& fibers, const OperatorRef& op, bool left) {
  const Index dim = rho.rows();
  const Index chunks = (dim + kChunk - 1) / kChunk;
  const std::int64_t tasks = static_cast<std::int64_t>(fibers.count() * chunks);
  const Index largest = max_fiber(fibers);
  const bool parallel = gate_work(fibers) * dim > kParallelWork;
#if PGQ_HAVE_OPENMP
#pragma omp parallel if (parallel)
#endif
  {
    ComplexMatrix block(largest, kChunk);
    ComplexMatrix result(largest, kChunk);
#if PGQ_HAVE_OPENMP
#pragma omp for schedule(dynamic, 4)
#endif
    for (std::int64_t task = 0; task < tasks; ++task) {
      const Index f = static_cast<Index>(task) / chunks;
      const Index start = (static_cast<Index>(task) % chunks) * kChunk;
      const Index width = std::min(kChunk, dim - start);
      const Index size = fibers.size(f);
      const Index* idx = fibers.begin(f);
      const auto u = op.topLeftCorner(size, size);
      if (left) {
        for (Index c = 0; c < width; ++c)
          for (Index r = 0; r < size; ++r) block(r, c) = rho(idx[r], start + c);
        result.topLeftCorner(size, width).noalias() = u * block.topLeftCorner(size, width);
        for (Index c = 0; c < width; ++c)
          for (Index r = 0; r < size; ++r) rho(idx[r], start + c) = result(r, c);
      } else {
        // Work on the transpose: (rows x size) U^dag  ==  (conj(U) (size x rows))^T.
        for (Index c = 0; c < width; ++c)
          for (Index r = 0; r < size; ++r) block(r, c) = rho(start + c, idx[r]);
        result.topLeftCorner(size, width).noalias() = u.conjugate() * block.topLeftCorner(size, width);
        for (Index c = 0; c < width; ++c)
          for (Index r = 0; r < size; ++r) rho(start + c, idx[r]) = result(r, c);
      }
    }
  }
  (void)parallel;
}

ComplexVector global_phases(const FockBasis& basis, std::span<const int> local,
                            const RealVector& phases) {
  if (phases.size() != dimension(static_cast<int>(local.size()), basis.cutoff())) {
    throw PhysicsError("Phase table must have the local truncated dimension");
  }
  const Fibers fibers = make_fibers(basis, local);
  ComplexVector result(basis.size());
  for (Index f = 0; f < fibers.count(); ++f) {
    const Index* idx = fibers.begin(f);
    for (Index r = 0; r < fibers.size(f); ++r) result(idx[r]) = std::polar(1.0, phases(r));
  }
  return result;
}

std::vector<Real> binomial_row_table(int rows) {
  std::vector<Real> table(static_cast<std::size_t>(rows) * rows, 0.0);
  for (int n = 0; n < rows; ++n) {
    table[static_cast<std::size_t>(n) * rows] = 1.0;
    for (int k = 1; k <= n; ++k) {
      table[static_cast<std::size_t>(n) * rows + k] =
          table[static_cast<std::size_t>(n - 1) * rows + k - 1] +
          table[static_cast<std::size_t>(n - 1) * rows + k];
    }
  }
  return table;
}

struct KrausMap {
  std::vector<Index> sources;
  std::vector<Index> targets;
  std::vector<Real> factors;
};

// out[t_i, t_j] += f_i f_j in[s_i, s_j]; targets are distinct, so rows are race-free.
void accumulate(RowComplexMatrix& out, ConstDensityRef in, const KrausMap& map) {
  const std::int64_t count = static_cast<std::int64_t>(map.sources.size());
  const bool parallel = count * count > kParallelWork;
#if PGQ_HAVE_OPENMP
#pragma omp parallel for schedule(static) if (parallel)
#endif
  for (std::int64_t i = 0; i < count; ++i) {
    const Index si = map.sources[i];
    const Index ti = map.targets[i];
    const Real fi = map.factors[i];
    for (std::int64_t j = 0; j < count; ++j) {
      out(ti, map.targets[j]) += fi * map.factors[j] * in(si, map.sources[j]);
    }
  }
  (void)parallel;
}

}  // namespace

int max_threads() {
#if PGQ_HAVE_OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

void apply_local(StateRef state, const FockBasis& basis, std::span<const int> local,
                 const OperatorRef& op) {
  check_state(state.size(), basis);
  check_operator(op, basis, local);
  const Fibers fibers = make_fibers(basis, local);
  const std::int64_t count = static_cast<std::int64_t>(fibers.count());
  const Index largest = max_fiber(fibers);
  const bool parallel = gate_work(fibers) > kParallelWork;
#if PGQ_HAVE_OPENMP
#pragma omp parallel if (parallel)
#endif
  {
    ComplexVector in(largest);
    ComplexVector out(largest);
#if PGQ_HAVE_OPENMP
#pragma omp for schedule(dynamic, 16)
#endif
    for (std::int64_t f = 0; f < count; ++f) {
      const Index size = fibers.size(static_cast<Index>(f));
      const Index* idx = fibers.begin(static_cast<Index>(f));
      for (Index r = 0; r < size; ++r) in(r) = state(idx[r]);
      out.head(size).noalias() = op.topLeftCorner(size, size) * in.head(size);
      for (Index r = 0; r < size; ++r) state(idx[r]) = out(r);
    }
  }
  (void)parallel;
}

void apply_local(DensityRef rho, const FockBasis& basis, std::span<const int> local,
                 const OperatorRef& op) {
  check_state(rho.rows(), basis);
  check_state(rho.cols(), basis);
  check_operator(op, basis, local);
  const Fibers fibers = make_fibers(basis, local);
  multiply_fibres(rho, fibers, op, true);
  multiply_fibres(rho, fibers, op, false);
}

void apply_local_phase(StateRef state, const FockBasis& basis, std::span<const int> local,
                       const RealVector& phases) {
  check_state(state.size(), basis);
  state.array() *= global_phases(basis, local, phases).array();
}

void apply_local_phase(DensityRef rho, const FockBasis& basis, std::span<const int> local,
                       const RealVector& phases) {
  check_state(rho.rows(), basis);
  check_state(rho.cols(), basis);
  const ComplexVector phase = global_phases(basis, local, phases);
  const ComplexVector conj = phase.conjugate();
  const std::int64_t dim = static_cast<std::int64_t>(rho.rows());
#if PGQ_HAVE_OPENMP
#pragma omp parallel for schedule(static) if (dim * dim > kParallelWork)
#endif
  for (std::int64_t i = 0; i < dim; ++i) {
    rho.row(i).array() *= phase(i) * conj.transpose().array();
  }
}

ComplexMatrix reduced_density(ConstStateRef state, const FockBasis& basis, std::span<const int> keep) {
  check_state(state.size(), basis);
  const Fibers fibers = make_fibers(basis, keep);
  const Index dim = dimension(static_cast<int>(keep.size()), basis.cutoff());
  ComplexMatrix out = ComplexMatrix::Zero(dim, dim);
  ComplexVector v(dim);
  for (Index f = 0; f < fibers.count(); ++f) {
    const Index size = fibers.size(f);
    const Index* idx = fibers.begin(f);
    for (Index r = 0; r < size; ++r) v(r) = state(idx[r]);
    out.topLeftCorner(size, size).noalias() += v.head(size) * v.head(size).adjoint();
  }
  return out;
}

ComplexMatrix reduced_density(ConstDensityRef rho, const FockBasis& basis, std::span<const int> keep) {
  check_state(rho.rows(), basis);
  check_state(rho.cols(), basis);
  const Fibers fibers = make_fibers(basis, keep);
  const Index dim = dimension(static_cast<int>(keep.size()), basis.cutoff());
  ComplexMatrix out = ComplexMatrix::Zero(dim, dim);
  for (Index f = 0; f < fibers.count(); ++f) {
    const Index size = fibers.size(f);
    const Index* idx = fibers.begin(f);
    for (Index s = 0; s < size; ++s)
      for (Index r = 0; r < size; ++r) out(r, s) += rho(idx[r], idx[s]);
  }
  return out;
}

RealVector photon_marginal(ConstStateRef state, const FockBasis& basis, std::span<const int> keep) {
  check_state(state.size(), basis);
  const Fibers fibers = make_fibers(basis, keep);
  RealVector out = RealVector::Zero(dimension(static_cast<int>(keep.size()), basis.cutoff()));
  for (Index f = 0; f < fibers.count(); ++f) {
    const Index* idx = fibers.begin(f);
    for (Index r = 0; r < fibers.size(f); ++r) out(r) += std::norm(state(idx[r]));
  }
  return out;
}

RealVector photon_marginal(ConstDensityRef rho, const FockBasis& basis, std::span<const int> keep) {
  check_state(rho.rows(), basis);
  const Fibers fibers = make_fibers(basis, keep);
  RealVector out = RealVector::Zero(dimension(static_cast<int>(keep.size()), basis.cutoff()));
  for (Index f = 0; f < fibers.count(); ++f) {
    const Index* idx = fibers.begin(f);
    for (Index r = 0; r < fibers.size(f); ++r) out(r) += rho(idx[r], idx[r]).real();
  }
  return out;
}

RowComplexMatrix thermal_loss(ConstDensityRef rho, const FockBasis& basis, int mode,
                              Real transmissivity, Real thermal_photons) {
  check_state(rho.rows(), basis);
  check_state(rho.cols(), basis);
  if (mode < 0 || mode >= basis.modes()) throw PhysicsError("Loss mode out of range");
  if (!std::isfinite(transmissivity) || !std::isfinite(thermal_photons) || transmissivity < 0.0 ||
      transmissivity > 1.0 || thermal_photons < 0.0) {
    throw PhysicsError("Loss requires 0 <= transmissivity <= 1 and thermal_photons >= 0");
  }
  const int cutoff = basis.cutoff();
  const Index dim = basis.size();
  const Real gain = 1.0 + (1.0 - transmissivity) * thermal_photons;
  const Real eta = transmissivity / gain;
  const std::vector<Real> binom = binomial_row_table(cutoff + 1);
  auto choose = [&](int n, int k) { return binom[static_cast<std::size_t>(n) * (cutoff + 1) + k]; };
  std::vector<Occupation> occ(static_cast<std::size_t>(basis.modes()));

  RowComplexMatrix lost = RowComplexMatrix::Zero(dim, dim);
  for (int l = 0; l < cutoff; ++l) {
    KrausMap map;
    for (Index g = 0; g < dim; ++g) {
      const auto source = basis.occupation(g);
      const int n = source[mode];
      if (n < l) continue;
      occ.assign(source.begin(), source.end());
      occ[mode] -= l;
      map.sources.push_back(g);
      map.targets.push_back(basis.rank(occ));
      map.factors.push_back(
          std::sqrt(choose(n, l) * std::pow(1.0 - eta, l) * std::pow(eta, n - l)));
    }
    accumulate(lost, rho, map);
  }
  if (gain == 1.0) return lost;
  RowComplexMatrix out = RowComplexMatrix::Zero(dim, dim);
  for (int added = 0; added < cutoff; ++added) {
    KrausMap map;
    for (Index g = 0; g < dim; ++g) {
      const auto source = basis.occupation(g);
      int total = 0;
      for (const Occupation n : source) total += n;
      if (total + added >= cutoff) continue;
      const int n = source[mode];
      occ.assign(source.begin(), source.end());
      occ[mode] += added;
      map.sources.push_back(g);
      map.targets.push_back(basis.rank(occ));
      map.factors.push_back(std::sqrt(choose(n + added, added) * std::pow(gain - 1.0, added) /
                                      std::pow(gain, n + added + 1)));
    }
    accumulate(out, lost, map);
  }
  return out;
}

Physicality physicality(ConstDensityRef rho) {
  if (rho.rows() != rho.cols()) throw PhysicsError("Density matrix must be square");
  const ComplexMatrix dense = rho;
  const ComplexMatrix hermitian = (dense + dense.adjoint()) / 2.0;
  const Eigen::SelfAdjointEigenSolver<ComplexMatrix> solver(hermitian, Eigen::EigenvaluesOnly);
  return {(dense - dense.adjoint()).cwiseAbs().maxCoeff(), dense.trace(),
          solver.eigenvalues().minCoeff(), solver.eigenvalues().maxCoeff()};
}

}  // namespace photographiq::fock
