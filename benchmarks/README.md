# PhotoGraphiQ benchmarks

| Directory | Script | What it measures |
|---|---|---|
| `boson_sampling/` | `bench_boson_sampling.py` | permanent kernels; single BS output probability (API and kernel level); full output distribution; exact sampler throughput. PhotoGraphiQ native / Python vs Piquasso. |
| `gaussian_boson_sampling/` | `bench_gbs.py` | hafnian, loop hafnian, torontonian, loop torontonian kernels; GBS PNR and threshold probabilities over sweeps of modes, photon number, squeezing, loss and displacement. PhotoGraphiQ native / Python vs Piquasso. |
| `gaussian/` | `bench_native_gaussian.py` | native Gaussian engine vs NumPy (dense backend, local update, vectorized batch, Python loop) for gates, homodyne and batches; see [Native Gaussian benchmark](#native-gaussian-benchmark-gaussian). |
| `fock/` | `bench_fock.py` | native-fock / native-mixed-fock vs piquasso-fock / piquasso-mixed-fock: per-gate and per-pattern runtime over modes and cutoff (each output checked against Piquasso before timing), process memory, OpenMP thread scaling, and accuracy vs cutoff against the exact Gaussian simulation. |
| `graphix/` | `bench_graphix_mbqc.py` | qubit MBQC only: flow finding, standardization, signal shifting and statevector simulation, PhotoGraphiQ vs Graphix 0.4; see [`graphix/README.md`](graphix/README.md). |
| `gpu/` | `bench_gpu_gaussian.py` | batched Gaussian kernels on GPU (CuPy) vs NumPy and native C++, transfers timed separately; see [`gpu/README.md`](gpu/README.md). |
| `gaussian_boson_sampling/` | `threshold_mp_reference.py` | 50-digit mpmath reference for the threshold probabilities of `bench_gbs.py` cases, with each implementation's actual and estimated relative error. |

`plot_results.py` draws the figures of the native Gaussian, Fock and GPU benchmarks from their
JSON (newest non-smoke file of each by default, or `--gaussian/--fock/--gpu PATH`) into
`<benchmark>/results/figures_<json stem>/`; the sampling and Graphix benchmarks draw their own.

`bench_common.py` holds the shared environment-metadata, timing, memory, JSON and figure helpers
used by the sampling, native Gaussian, GPU and Graphix benchmarks.

## How to run

From the repository root, with the package installed (native extension optional, Piquasso
optional):

```bash
python benchmarks/boson_sampling/bench_boson_sampling.py            # default, ~1 min
python benchmarks/gaussian_boson_sampling/bench_gbs.py              # default, ~1 min
python benchmarks/gaussian_boson_sampling/bench_gbs.py --smoke      # seconds (used by the test-suite)
python benchmarks/gaussian_boson_sampling/bench_gbs.py --full       # larger sizes, much longer
```

Options: `--seed`, `--output path.json`, `--no-figures`. Each run writes
`results/<name>_<config>_<timestamp>.json` and SVG + PDF figures into
`results/figures_<config>/`. Set `OMP_NUM_THREADS` / `NUMBA_NUM_THREADS` before starting
Python to change thread counts; they are recorded. The notebook
`examples/notebooks/boson_sampling_and_gbs_benchmark.ipynb` runs the same code live in a
smaller `notebook` configuration and plots it. The smoke test is
`tests/sampling/test_sampling_benchmark_smoke.py`.

## Methodology

* **Equivalent tasks only.** Every comparison row states what is compared (`comparisons` in
  the JSON). Kernels get the identical matrix (the torontonian only differs by an index
  relabelling, because Piquasso pairs `(2i, 2i+1)` and PhotoGraphiQ `(i, i+n)`).
  Probability APIs get the identical state (GBS covariances are checked equal to ~1e-15)
  and the identical output pattern. A single-probability evaluator is never compared to a
  full-state simulator; such rows are marked *not applicable*. PhotoGraphiQ rows with
  precomputed `GBSMatrices` have no Piquasso counterpart and are labelled as such.
* **Setup separated from computation.** Piquasso program execution and PhotoGraphiQ
  state / `GBSMatrices` construction are timed in `*_setup` rows, not inside the
  probability timings. The correctness value of each row is computed by an untimed call
  before timing starts, so one-off JIT compilation (Piquasso's numba hafnians) is excluded
  from every reported number and not itself recorded. The timer's own first call is
  reported as `first_call_s` and is never a sample (it is a cold call only for `*_setup`
  rows).
* **Timing.** `time.perf_counter`; after the first call, one call to estimate the cost,
  `warmup` untimed calls, then `repeats` samples, each the mean of `inner` back-to-back calls
  (a sample lasts >= 2 ms). Reported: median, IQR, min, mean. Cases whose estimated cost
  exceeds the per-case budget are measured without warm-up and with fewer repeats (`warmup`
  and `repeats` are recorded per row).
* **Correctness.** Each kernel row carries `rel_error` against a reference: the defining
  permutation / matching sum for small sizes, else the PhotoGraphiQ Python implementation
  where it was run, else Piquasso (`reference` names it). GBS rows carry relative and
  absolute differences against Piquasso and against PhotoGraphiQ Python. BS distributions
  report normalisation and the maximum difference to Piquasso; samplers report photon-number
  conservation and the total-variation distance to the exact distribution (only meaningful
  when shots >> number of patterns).
* **Memory.** `tracemalloc` peak of one call. It includes NumPy buffers but **not** memory
  allocated directly by C/C++ (PhotoGraphiQ native kernels, Piquasso's C++ permanent and
  torontonian) or by numba-compiled code, so it is a lower bound for those implementations.
* **Metadata.** CPU model, logical CPUs, OS, Python, package versions (photographiq,
  piquasso, graphix, numpy, scipy, numba), native build info, OpenMP / numba thread counts,
  thread environment variables, precision, seed, and the git revision with a dirty flag
  (read-only `git rev-parse HEAD` and `git status --porcelain` with `--no-optional-locks`).

## Caveats

* Wall-clock timings on one machine; small sizes are dominated by Python call overhead and
  input validation, not by the exponential kernels.
* Libraries use their own threading (PhotoGraphiQ OpenMP, Piquasso C++ / numba); runtime
  ratios are not single-core comparisons.
* Threshold (torontonian) probabilities lose digits as the number of clicks grows and the
  click probability shrinks. `gaussian_boson_sampling/threshold_mp_reference.py` computes a
  50-digit mpmath reference for exact `bench_gbs` cases
  (`gaussian_boson_sampling/results/threshold_mp_reference_20261111-094449.json`; r = 0.5,
  eta = 0.9). Measured relative errors native / Python / Piquasso: 1.7e-7 / 1.7e-7 / 2.2e-8
  (24 modes, 12 clicks, p = 2.7e-9), 2.8e-5 / 2.5e-5 / 2.2e-6 (28 modes, 14 clicks,
  p = 6.1e-11), 7.5e-3 / 2.9e-3 / 3.0e-4 (32 modes, 16 clicks, p = 9.0e-13). **On these cases
  Piquasso is about 10x more accurate than PhotoGraphiQ in double precision.**
  `threshold_probability(..., return_error=True)` returns a rounding bound that held in every
  case (1.4e-4, 2.0e-2 and 3.6 relative, i.e. 480x to 1300x the actual error), so the larger
  cases raise `ThresholdPrecisionWarning`; use `precision=` (mpmath, extra `precision`) when the
  digits matter. The benchmark's `rel_error_vs_piquasso` for threshold rows mixes the errors of
  both libraries (largest in the default run: 3.5e-6).
  PNR (hafnian) probabilities agree with Piquasso to <= 5.7e-12 relative in every default case
  (`gbs_default_20261111-092803.json`).
* Native Fock kernels use OpenMP only when a gate's work exceeds 2^22 multiply-adds
  (`cpp/src/fock/kernels.cpp`, `kParallelWork`), so the 5-mode pure thread-scaling case
  (dimension C(24, 5) = 42504) runs serially and shows no speedup by design; the 3-mode
  density-matrix case is above the threshold and scales.
* GPU rows: the float64 `n = 64, B = 4096` case exceeds the 512 MB per-array cap
  (`--max-bytes`) and is recorded under `skipped`, so the float64 curves end at B = 256.
* The sampling benchmarks run on the CPU only (GPU kernels cover batched Gaussian
  operations, `gpu/`). Graphix is a qubit-MBQC framework and is not a GBS
  simulator; its benchmark (if present, `benchmarks/graphix/results/`) is only displayed by
  the notebook, never compared with these results.

## Native Gaussian benchmark (`gaussian/`)

```bash
python benchmarks/gaussian/bench_native_gaussian.py --smoke   # seconds (used by the test-suite)
python benchmarks/gaussian/bench_native_gaussian.py           # default: n = 8..256, B <= 64
python benchmarks/gaussian/bench_native_gaussian.py --full    # n up to 512, B <= 256
```

Needs the `gaussian` native component. Writes
`gaussian/results/native_gaussian_<config>_<timestamp>.json`; smoke test:
`tests/native/test_native_gaussian_benchmark_smoke.py`.

**Implementations** (`kind` in each row):

| Task | `kind` | What runs |
|---|---|---|
| `gate` (1 state, 1 beam splitter, in place) | `numpy-dense-backend` | `GaussianBackend.transform`: dense embedded `S V S^T`, O(n^3), plus re-validation of the new state (complex `eigvalsh`, O(n^3)) |
| | `numpy-local-single` | NumPy O(n k) update of the touched rows/columns |
| | `native-single` | `NativeGaussianBackend.transform`, C++ O(n k) |
| `homodyne` (1 state, destructive, fixed outcome) | `numpy-dense-backend` | `GaussianBackend.measure` (also re-validates the new state, O(n^3)) |
| | `native-single` | `NativeGaussianBackend.measure` |
| `batch_gate`, `batch_homodyne` (B states) | `native-batch` | `batch_apply_symplectic` / `batch_condition_homodyne`, OpenMP over B |
| | `numpy-vectorized` | the C++ kernel's algorithm in batched NumPy (matmul / einsum / fancy indexing over a leading batch axis), no Python loop |
| | `python-loop` | Python loop over B calling the per-state NumPy path (local update; `GaussianBackend.measure`, which also re-validates every output state, O(n^3)) |

**Methodology.**

* Identical inputs: B distinct physical states (one squeezed, lossy CZ chain, built with the
  native updates and validated once, with per-state
  random displacements and added classical noise `V + l_b I`), distinct beam-splitter angles,
  homodyne angles and outcomes, efficiency 0.9. Built before any timing.
* Identical outputs, checked before timing: every implementation's output arrays (means,
  covariances, densities) are compared with native; the run aborts if
  `max|x - x_native| / max(1, max|x_native|) > 1e-12`. The value is stored per row as
  `max_scaled_diff_vs_native` (observed <= ~2e-16 for gates and 0 for homodyne in the smoke
  config and at n = 3..64, B = 8). Every implementation also symmetrises the input
  covariances, as the native kernels do; `numpy-vectorized` omits only the per-matrix
  symplecticity check of `batch_apply_symplectic` (O(B k^3), k = 4).
* Timing: `bench_common.time_call` (median, IQR, min, mean; cold first call excluded). The
  destructive single-state homodyne builds a fresh backend per sample outside the timed
  region. B is capped so one covariance batch stays below `batch_mb` MB (but B >= 4).
* `speedups` gives `ratio_of_medians` per task and n with the comparison named, e.g.
  `numpy-vectorized / native-batch` (the honest batched C++ gain) and
  `python-loop / native-batch` (includes Python loop overhead and, for `batch_homodyne`, an
  O(n^3) re-validation of every output state that neither batch implementation performs; an
  upper bound, not like-for-like).
* Environment adds the NumPy BLAS build, threadpoolctl info (if installed), and the native
  OpenMP max thread count to the common metadata.

**Caveats.**

* The native batch kernels run on all OpenMP threads (default: logical CPU count); NumPy's
  batched matmul loops over the batch axis serially (each small 2n x 4 product may use BLAS
  threads, the batch loop is not parallelised).
  `numpy-vectorized / native-batch` is therefore not a single-core comparison; rerun with
  `OMP_NUM_THREADS=1` for one.
* At small n both sides are dominated by call overhead; the earlier review found NumPy
  faster than native in some batched cases (n = 16). Read the ratios, do not assume them.
* The previous results file (Python-loop baseline only, overstated batched speedups) is in
  `gaussian/results/superseded/` and should not be cited.
