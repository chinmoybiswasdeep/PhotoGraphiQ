#include "photographiq/fock/basis.hpp"

#include <limits>
#include <string>

namespace photographiq::fock {
namespace {

constexpr Index kMaxBasisEntries = Index{1} << 33;

void validate_shape(int modes, int cutoff) {
  if (modes < 0) throw PhysicsError("Mode count must be nonnegative");
  if (cutoff < 1) throw PhysicsError("Fock cutoff must be at least one");
}

// Pascal triangle with rows 0..n-1, saturating at a sentinel instead of overflowing.
std::vector<Index> pascal(int rows) {
  constexpr Index kSaturated = std::numeric_limits<Index>::max() / 2;
  std::vector<Index> table(static_cast<std::size_t>(rows) * rows, 0);
  for (int n = 0; n < rows; ++n) {
    table[static_cast<std::size_t>(n) * rows] = 1;
    for (int k = 1; k <= n; ++k) {
      const Index left = table[static_cast<std::size_t>(n - 1) * rows + k - 1];
      const Index up = table[static_cast<std::size_t>(n - 1) * rows + k];
      table[static_cast<std::size_t>(n) * rows + k] =
          (left >= kSaturated || up >= kSaturated || left + up >= kSaturated) ? kSaturated
                                                                              : left + up;
    }
  }
  return table;
}

void fill_sector(int position, int remaining, std::vector<Occupation>& current,
                 std::vector<Occupation>& table) {
  const int modes = static_cast<int>(current.size());
  if (position == modes - 1) {
    current[position] = remaining;
    table.insert(table.end(), current.begin(), current.end());
    return;
  }
  for (int k = remaining; k >= 0; --k) {
    current[position] = k;
    fill_sector(position + 1, remaining - k, current, table);
  }
}

}  // namespace

Index dimension(int modes, int cutoff) {
  validate_shape(modes, cutoff);
  if (modes == 0) return 1;
  // C(modes + cutoff - 1, modes) with overflow detection.
  long double value = 1.0L;
  Index exact = 1;
  const int k = modes < cutoff - 1 ? modes : cutoff - 1;
  for (int i = 1; i <= k; ++i) {
    value = value * (modes + cutoff - 1 - k + i) / i;
    if (value > static_cast<long double>(kMaxBasisEntries)) {
      throw PhysicsError("Truncated Fock dimension exceeds 2^33 states");
    }
    exact = exact * (modes + cutoff - 1 - k + i) / i;  // exact: product of i consecutive / i!
  }
  return exact;
}

FockBasis::FockBasis(int modes, int cutoff)
    : modes_(modes), cutoff_(cutoff), size_(dimension(modes, cutoff)), rows_(modes + cutoff + 1) {
  if (size_ * static_cast<Index>(modes > 0 ? modes : 1) > kMaxBasisEntries) {
    throw PhysicsError("Fock basis table too large");
  }
  pascal_ = pascal(rows_);
  table_.reserve(static_cast<std::size_t>(size_ * modes));
  if (modes > 0) {
    std::vector<Occupation> current(static_cast<std::size_t>(modes), 0);
    for (int n = 0; n < cutoff; ++n) fill_sector(0, n, current, table_);
  }
}

Index FockBasis::binomial(int n, int k) const {
  if (k < 0 || n < 0 || k > n) return 0;
  return pascal_[static_cast<std::size_t>(n) * rows_ + k];
}

std::span<const Occupation> FockBasis::occupation(Index index) const {
  if (index < 0 || index >= size_) throw PhysicsError("Fock basis index out of range");
  return {table_.data() + index * modes_, static_cast<std::size_t>(modes_)};
}

Index FockBasis::rank(std::span<const Occupation> occupation) const {
  if (static_cast<int>(occupation.size()) != modes_) {
    throw PhysicsError("Occupation length " + std::to_string(occupation.size()) +
                       " does not match " + std::to_string(modes_) + " modes");
  }
  int total = 0;
  for (const Occupation n : occupation) {
    if (n < 0) throw PhysicsError("Occupations must be nonnegative");
    total += n;
    if (total >= cutoff_) throw PhysicsError("Occupation exceeds the total-photon cutoff");
  }
  if (modes_ == 0) return 0;
  // States of lower sectors, then states of this sector whose leading modes are larger.
  Index index = binomial(total + modes_ - 1, modes_);
  int remaining = total;
  for (int i = 0; i + 1 < modes_; ++i) {
    const int after = modes_ - i - 1;
    const int larger = remaining - occupation[i] - 1;
    if (larger >= 0) index += binomial(larger + after, after);
    remaining -= occupation[i];
  }
  return index;
}

Fibers make_fibers(const FockBasis& basis, std::span<const int> local) {
  const int modes = basis.modes();
  const int k = static_cast<int>(local.size());
  std::vector<char> is_local(static_cast<std::size_t>(modes), 0);
  for (const int m : local) {
    if (m < 0 || m >= modes || is_local[m]) {
      throw PhysicsError("Local modes must be distinct and in range");
    }
    is_local[m] = 1;
  }
  std::vector<int> rest;
  for (int m = 0; m < modes; ++m) {
    if (!is_local[m]) rest.push_back(m);
  }
  const FockBasis local_basis(k, basis.cutoff());
  const FockBasis rest_basis(static_cast<int>(rest.size()), basis.cutoff());

  Fibers fibers;
  fibers.offsets.resize(static_cast<std::size_t>(rest_basis.size()) + 1, 0);
  for (Index f = 0; f < rest_basis.size(); ++f) {
    int photons = 0;
    for (const Occupation n : rest_basis.occupation(f)) photons += n;
    fibers.offsets[f + 1] = fibers.offsets[f] + dimension(k, basis.cutoff() - photons);
  }
  fibers.indices.assign(static_cast<std::size_t>(basis.size()), -1);
  std::vector<Occupation> local_occ(static_cast<std::size_t>(k));
  std::vector<Occupation> rest_occ(rest.size());
  for (Index g = 0; g < basis.size(); ++g) {
    const auto occ = basis.occupation(g);
    for (int i = 0; i < k; ++i) local_occ[i] = occ[local[i]];
    for (std::size_t i = 0; i < rest.size(); ++i) rest_occ[i] = occ[rest[i]];
    const Index f = rest_basis.rank(rest_occ);
    fibers.indices[fibers.offsets[f] + local_basis.rank(local_occ)] = g;
  }
  return fibers;
}

}  // namespace photographiq::fock
