// Occupation basis of a truncated multimode Fock space.
//
// Truncation is by TOTAL photon number: an occupation (n_0, ..., n_{d-1}) is kept iff
// n_0 + ... + n_{d-1} < cutoff, so the dimension is C(d + cutoff - 1, d).
//
// Ordering (identical to Piquasso's get_fock_space_basis): states are grouped by total
// photon number N = 0, 1, ..., cutoff - 1; inside a sector the first mode is descending,
// then the second, ... e.g. d = 3: (000) (100) (010) (001) (200) (110) (101) (020) ...
// Because sectors come first, the states with total < L form the leading
// C(d + L - 1, d) block for every L <= cutoff. The kernels rely on this: an operator on
// k local modes truncated at `cutoff` acts on a fibre whose other modes hold s photons
// through its leading block for cutoff L = cutoff - s, i.e. as P U P (Piquasso semantics).
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "photographiq/core/types.hpp"

namespace photographiq::fock {

using Occupation = std::int32_t;

// Number of occupations of `modes` modes with total photon number < cutoff.
Index dimension(int modes, int cutoff);

class FockBasis {
 public:
  FockBasis(int modes, int cutoff);

  int modes() const noexcept { return modes_; }
  int cutoff() const noexcept { return cutoff_; }
  Index size() const noexcept { return size_; }

  // Occupation of basis state `index` (length modes()).
  std::span<const Occupation> occupation(Index index) const;
  // Combinatorial rank of an occupation; throws PhysicsError outside the truncated space.
  Index rank(std::span<const Occupation> occupation) const;
  // Row-major (size() x modes()) occupation table.
  const std::vector<Occupation>& table() const noexcept { return table_; }

 private:
  Index binomial(int n, int k) const;

  int modes_;
  int cutoff_;
  Index size_;
  std::vector<Occupation> table_;
  std::vector<Index> pascal_;  // (rows_ x rows_) Pascal triangle
  int rows_;
};

// Partition of a basis by the occupations of the modes outside `local`.
// Fibre f collects the states sharing one "rest" occupation with s photons; its
// indices[offsets[f] + r] is the global index whose local occupation (in the order
// given by `local`) has rank r in FockBasis(local.size(), cutoff). Fibre f has
// dimension(local.size(), cutoff - s) states.
struct Fibers {
  std::vector<Index> offsets;  // fibre count + 1 entries
  std::vector<Index> indices;  // basis.size() global indices, fibre-major
  Index count() const noexcept { return static_cast<Index>(offsets.size()) - 1; }
  Index size(Index f) const noexcept { return offsets[f + 1] - offsets[f]; }
  const Index* begin(Index f) const noexcept { return indices.data() + offsets[f]; }
};

// `local` must hold distinct, in-range modes; its order defines the local occupation.
Fibers make_fibers(const FockBasis& basis, std::span<const int> local);

}  // namespace photographiq::fock
