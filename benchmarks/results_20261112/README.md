# Reproduction run 2026-11-12 (`full-20261112-041139`)

Produced by `python scripts/reproduce_manuscript.py --mode full` on 2026-11-12, 04:11–04:56 local
time (UTC−5). The run used the local working tree on branch `upgraded`: HEAD `f30321f` plus the
uncommitted Phase 1–3 changes, so `summary.json` records `git_dirty: true`. Apart from a few
short architecture-figure renders, no other job was running; CPU load was 11 % at the start.

Machine:
- Windows 11 (10.0.26200);
- Intel Core i5-12450HX, 12 logical CPUs;
- NVIDIA GeForce RTX 4050 Laptop GPU (CuPy 14.2.0, CUDA runtime 13.2);
- Python 3.12.14, MSVC 19.50 native build with OpenMP;
- NumPy 2.5.3, SciPy 1.18.1, Piquasso 8.0.1 (numba 0.67.0), Graphix 0.4.

| File | Produced by |
| --- | --- |
| `summary.json` | step list, commands, exit codes, run times and environment |
| `bench-boson-sampling.json`, `bench-gbs.json`, `figures_default/` | `benchmarks/boson_sampling/bench_boson_sampling.py`, `benchmarks/gaussian_boson_sampling/bench_gbs.py` (default configs) |
| `bench-native-gaussian.json`, `figures_bench-native-gaussian/` | `benchmarks/gaussian/bench_native_gaussian.py`, `benchmarks/plot_results.py` |
| `bench-fock.json`, `figures_bench-fock/` | `benchmarks/fock/bench_fock.py` (full config), `benchmarks/plot_results.py` |
| `bench-gpu-gaussian.json`, `figures_bench-gpu-gaussian/` | `benchmarks/gpu/bench_gpu_gaussian.py --dtypes float64 float32` |
| `bench-gpu-fock.json` | `benchmarks/gpu/bench_gpu_fock.py` |
| `graphix/` | `benchmarks/graphix/bench_graphix_mbqc.py` (fast config) |
| `threshold_mp_reference.json` | `benchmarks/gaussian_boson_sampling/threshold_mp_reference.py 24,12,0.9 28,14,0.9 32,16,0.9` |
| `paper_experiments*-comparison*.json` | `scripts/compare_recorded_results.py`. It compares the re-run safe subsets, run from copies, with the recorded `paper_experiments*/results/json`. The `-atol` file uses `--atol 1e-12`. |

The earlier result files in each `benchmarks/*/results/` directory (2026-11-10/11) are kept
unchanged as the previous runs. Timings vary from run to run by roughly ±30 % for small cases
on this laptop. Accuracy fields are deterministic.
