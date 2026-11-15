# Graphix qubit-MBQC benchmark

**Scope.** This benchmark compares PhotoGraphiQ's *qubit* MBQC tooling
(`photographiq.flows`, `photographiq.calculus.qubit`) with Graphix 0.4. It covers qubit
measurement-based computation only. Graphix does not simulate CV or GBS, so these results say
nothing about PhotoGraphiQ's CV, Gaussian, Fock or sampling engines. Both packages solve the
same task on the same input in every timed row, and their outputs are checked for agreement
before any timing.

| Task | Same input | Same output | PhotoGraphiQ call | Graphix call |
|---|---|---|---|---|
| causal flow | open graph, XY labels | existence + flow + layers | `find_causal_flow` (search + built-in verify) | `to_causalflow_or_none` (+ `check_well_formed`) |
| gflow | open graph, XY/XZ/YZ | existence + max-delayed gflow | `find_gflow` | `to_gflow_or_none` (+ `check_well_formed`) |
| Pauli flow | open graph, planes + X/Y/Z | existence + max-delayed Pauli flow | `find_pauli_flow` | `to_pauliflow_or_none` (+ `check_well_formed`) |
| standardize | Graphix `XZCorrections.to_pattern` (non-standard) | equivalent NEMC pattern | `qubit.standardize` | `optimization.standardize` |
| signal shift | the Graphix standard form | shifted pattern + signal map | `qubit.signal_shift` (appends `S`) | `Pattern.shift_signals("direct")` (returns dict) |
| statevector | the same standard pattern, one fixed branch, `\|+>` inputs | normalised output state | `qubit.run_branch` (inputs prepended as `N`) | `Pattern.simulate("statevector", FixedBranchSelector)` |

PhotoGraphiQ's flow finders always verify their certificate before returning. The timing
ratio therefore compares them with Graphix *search + `check_well_formed`*. Graphix search-only
time is reported in its own column. Each certificate is checked by both verifiers: PhotoGraphiQ
`verify_*` and Graphix `check_well_formed`, including on the other package's certificate. On
small sizes the map equivalence of standardization and signal shifting is checked by exact
branch-by-branch simulation (`qubit.equivalent`).

These comparisons are **not applicable**:

- any CV, Gaussian or GBS comparison
- PhotoGraphiQ's full Kraus map (`run_branch` with inputs) against Graphix's single output state
- Graphix's tensor-network and density-matrix backends

Families: linear clusters, 3xL grids and 4-row brickwork-like graphs (BFK-style rung pattern, not an exact BFK
reproduction), all with XY labels; for
the Pauli task, 30% of the nodes become X/Y. `lc_random` is a 3xL grid randomised by local
complementations with the Backens et al. (2021) plane update, which gives mixed planes and
keeps gflow. `er` is Erdos-Renyi graphs, 4 to 12 nodes, used only for the existence-agreement
sweep.

## Run

```
.venv/Scripts/python.exe benchmarks/graphix/bench_graphix_mbqc.py           # fast, a few minutes
.venv/Scripts/python.exe benchmarks/graphix/bench_graphix_mbqc.py --full    # larger sizes, long
  options: --seed N --repeats N --warmup N --cap SECONDS --no-plots
```

Results go to `results/graphix_mbqc_<fast|full>_<timestamp>.{json,svg,pdf}`. The JSON records
the environment (CPU, versions, git revision and dirty flag, thread variables), seeds,
warm-up, repeats, the median/IQR/min of each call, the first (cold) call, and the tracemalloc
peak. tracemalloc does not see numba or other C-level allocations. Calls faster than 2 ms
are timed in batches, so their IQR is the spread of batch means. Graphix compiles its numba
kernels at import (`setup_import_s`), not in the first call. Timings are only meaningful on an
otherwise idle machine; the script does not check load. A family stops growing once
a median passes `--cap`. A tiny smoke run is in
`tests/flow/test_flows_graphix_benchmark_smoke.py`.
