**R46 - implemented Fock-space optimisations.** Timings: final (idle machine declared via PGQ_IDLE_MACHINE=1).
(a) Basis size of the total-photon truncation n_1 + ... + n_m < c used by every Fock backend,
C(m + c - 1, m), vs a per-mode product basis c^m (a different truncation, shown for size only).
(b) Runtime of one beam splitter (theta = 0.7) on B = 1 and 32 random states for implementations of
the SAME truncated operator P U P: the native fibre kernel (no global operator), the global sparse
gate G applied with scipy.sparse and with cuSPARSE (CUDA, incl. host-device transfers),
and the dense D x D matrix (D <= 4000); all outputs agree to 1.7e-16 (acceptance
1e-12). At the largest grid point (m = 8, c = 9, D = 12870) the native kernel takes
5.37e-04 s vs 7.36e-04 s for scipy.sparse, B = 1. (c) Bytes of each gate representation
(local matrix vs CSR vs dense). (d) Photon-number conservation: the passive beam splitter's local
matrix is block diagonal in total-photon sectors (off-sector entries exactly
0, 97.9% zeros at cutoff 64), so `gaussian_unitary` builds only
the sector blocks; compared with the active CZ of the same local size (an observation: the skip cannot
be disabled). Diagonal gates by phase multiplication are a median 1.0x faster than applying the
diagonal matrix, with identical output. Cached basis/cubic/sparse-structure speedups are in
R46_fock_caches.csv. complex128, seed 20261146. Source: R46 (15_native_fock/46_fock_space_optimization.py).
