# Performance

!!! warning "CPU only"
    Every number on this page was measured with the NumPy reference engine on the
    development laptop's CPU. **No CUDA-Q or GPU measurement exists.** GPU performance
    (HYB-11) is blocked until real CUDA-Q execution passes its tests
    ([runtime validation](../validation/CUDAQ_REAL_RUNTIME_VALIDATION.md)). No speed-up is
    claimed.

## HYB-12: runtime and memory per trajectory

Timings are medians of 3 repeats after 1 warm-up. Peak memory is measured with
`tracemalloc`.

**Three-mode cubic-phase CV-MBQC (pure; the HYB-05 pattern)**

| Engine | Size | Hilbert dim | Median s | Peak MiB |
|---|---|---|---|---|
| hybrid numpy-reference | d = 20 | 8 000 | 0.25 | 7.5 |
| hybrid numpy-reference | d = 28 | 21 952 | 0.76 | 28.5 |
| hybrid numpy-reference | d = 32 | 32 768 | 1.30 | 48.5 |
| native-fock | cutoff 32 | 5 984 | 0.019 | 1.7 |
| native-fock | cutoff 48 | 19 600 | 0.11 | 5.0 |

At matched accuracy (HYB-05: hybrid d = 28 against native cutoff 48 for γ = 0.15), the
hybrid CPU path is about 7× slower and uses about 6× more memory.

**Two-mode lossy Kerr step (mixed)**

| Engine | Size | Median s | Peak MiB |
|---|---|---|---|
| hybrid numpy-reference | d = 24 | 0.28 | 20.3 |
| native-mixed-fock | cutoff 30 | 0.16 | 10.0 |

## Memory estimate

`plan(pattern, dims).peak_bytes` gives the state memory at the peak live dimension: 16·D
bytes for a pure state and 16·D² for a mixed one. In every HYB-12 run it was below the
measured peak allocation. It is a lower bound because it excludes gate matrices,
instrument grids and the cached loss superoperators.

## Where the time goes

| Operation | Cost |
|---|---|
| Two-mode gates (CZ, beam splitter) | dense expm of a d² × d² matrix |
| Mixed-state loss | a d² × d² superoperator expm per (η, n, d), cached and reused across trajectories |
| Homodyne | reduced density on a 4001-point grid (the measured mode only), plus one contraction for the conditional state |

These are the operations CUDA-Q Dynamics would take over on a GPU target. Whether that
is faster has **not** been measured.
