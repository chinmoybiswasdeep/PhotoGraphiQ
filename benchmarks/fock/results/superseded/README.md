# Superseded Fock benchmark results

`fock-Ryuk-20261110-213735.json` is kept for the record only. Do not cite its timings.

- It was recorded on a shared, loaded machine with a scratch build (`build_fock4`) that
  predates the fix of the Bargmann-recurrence pivot in `cpp/src/fock/gates.cpp`.
- A spot check on 2026-11-10 (`bench_fock.py --quick`, same machine, still shared) did
  not reproduce it. Ratios quick/original for the same configurations ranged from 0.01 to
  3.75. For example, Piquasso `rotate`, 2 modes, cutoff 10: 0.133 s originally, 1.3 ms in
  the check. Native mixed `wire` 4/16: 0.040 s originally, 0.150 s in the check.
- The thread-scaling rows are not monotone (pure 5 modes/cutoff 20: 3.6 ms with 1 thread,
  18.5 ms with 4 threads), which points to contention.

New timings must come from `python benchmarks/fock/bench_fock.py` on an idle machine.
