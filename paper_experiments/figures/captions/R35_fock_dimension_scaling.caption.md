**R35: Fock-space dimension, memory and measured cost.** Left: log10 of the total-photon Fock dimension
D = binom(modes + cutoff - 1, modes) for 1-20 modes and cutoffs 2-40 (the formula of both
`piquasso-fock` and `native-fock` equals the independent binomial at every grid point); centre: bytes of one complex128
state vector, with the default allocation guard max_dimension = 1000000 (red). Right: measured median wall
time of a two-mode workload (two coherent inputs, BeamSplitter, Kerr, CubicPhase) versus dimension, total (solid) and
state preparation only (faint), on `native-fock`, `piquasso-fock` and `native-mixed-fock` (density matrix, cutoff <=
32); at cutoff 64: native-fock 0.0308 s, piquasso-fock 0.491 s.
The engines' final states agree to fidelity >= 1.0000000000. Timings are
final (idle machine declared via PGQ_IDLE_MACHINE=1) (full configuration, 5 repeats,
12 logical CPUs, native build MSVC 19.50.35729.0,
OpenMP True); they are local observations, not cross-machine claims.
Source: `paper_experiments/11_performance/35_fock_dimension_scaling.py`.
