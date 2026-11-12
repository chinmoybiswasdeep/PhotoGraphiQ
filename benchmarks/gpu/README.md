# GPU benchmark: batched Gaussian kernels

`bench_gpu_gaussian.py` times identical batched Gaussian tasks on three engines:

- NumPy on the CPU: `photographiq.gpu` with `device="cpu"`, which runs the same algorithm as
  the GPU path;
- the native C++/OpenMP batch kernels (`photographiq.backends.native_gaussian.batch_*`);
- CuPy on the GPU (`device="gpu"`).

Run it on an idle machine:

```bash
python benchmarks/gpu/bench_gpu_gaussian.py                       # float64, default grid
python benchmarks/gpu/bench_gpu_gaussian.py --dtypes float64 float32
python benchmarks/gpu/bench_gpu_gaussian.py --smoke               # tiny grid: checks the script, timings meaningless
```

To time a scratch build of the native extension, set
`PHOTOGRAPHIQ_NATIVE_PATH=<build>/_native*.pyd`. To change the native thread count, set
`OMP_NUM_THREADS` before Python starts.

## Tasks

The input is B copies of a physical n-mode state: a squeezed CZ chain with loss.

| Task | Work per call | Notes |
|---|---|---|
| `local_symplectic` | O(B n) | Beam splitter on modes (n/3, 2n/3). |
| `dense_symplectic` | O(B n^3) | A different random 2n x 2n symplectic matrix per state, on all modes (batched GEMM). |
| `homodyne` | O(B n^2) | Destructive homodyne on mode n/3 with angle 0.4, efficiency 0.9 and outcome 0.1. Returns conditioned moments and densities. |

## Implementations

| Row | What is timed |
|---|---|
| `numpy` | `device="cpu"` (NumPy / OpenBLAS). |
| `native` | C++ batch kernel. float64 only, so there are no float32 rows. |
| `gpu_end_to_end` | NumPy in, NumPy out: H2D, kernels, D2H. |
| `gpu_kernel` | Device-resident input (including the per-state matrices of `dense_symplectic`), `on_device=True`, then `cupy.cuda.Device().synchronize()` inside the timed region. |
| `gpu_h2d` | Upload of means and covariances (and the per-state matrices for `dense_symplectic`), synchronised. |
| `gpu_d2h` | Download of the result covariances. |

`gpu_kernel` includes the device-to-host synchronisations of the validity checks
(symplecticity, positive variance). Users pay these on every call.

## Methodology

- **Timing.** `benchmarks/bench_common.time_call`. The first call is reported as
  `first_call_s` and is never a sample, which excludes CuPy's one-off kernel compilation and
  cuBLAS/cuSOLVER handle creation. Next come warm-up calls, then samples of back-to-back
  calls. The JSON reports median, IQR, min and mean.
- **Correctness first.** Before timing, each implementation's output is compared with the
  float64 `numpy` result: `max_rel_deviation_vs_numpy_float64`, the maximum absolute deviation
  divided by the largest entry. For float32 rows this measures the precision loss.
- **Memory cap.** Combinations whose covariance batch exceeds `--max-bytes` (default 512 MB)
  are skipped and listed under `skipped`.
- **Output.** `results/gpu_gaussian_<stamp>.json`, or `gpu_gaussian_smoke_<stamp>.json` with
  `--smoke`. The file records:
  - the environment, from `bench_common.environment`: CPU, threads, packages, native build
    and git state;
  - `gpu`: `photographiq.gpu.device_info()`, i.e. CuPy, CUDA runtime and driver, device name,
    compute capability and memory;
  - `native_gaussian_max_threads`.
- **No silent fallback.** Without CuPy or a device, GPU rows are not run and the script says
  so. No CPU number is ever labelled as GPU.

## Interpreting results

- GeForce GPUs run FP64 at 1/64 of their FP32 rate on the Ada generation. Compare
  float64 and float32 rows before drawing conclusions.
- Small `B * n^2` is dominated by fixed per-call latency: kernel launches and checks.
  `gpu_end_to_end` minus `gpu_kernel` is roughly the transfer cost, which `gpu_h2d` and
  `gpu_d2h` measure directly.

# GPU benchmark: batched Fock local gates

`bench_gpu_fock.py` times one local gate on B states in the native Fock engine's
total-photon truncated basis (`sum n < cutoff`). Pure batches are `(B, D)` and density
matrices `(B, D, D)`.

```bash
python benchmarks/gpu/bench_gpu_fock.py              # complex128 + complex64, default grid (~8 min)
python benchmarks/gpu/bench_gpu_fock.py --smoke      # tiny grid: checks the script, timings meaningless
```

**Grid.** Pure states: `(modes, cutoff)` in `(2, 30)`, `(4, 20)`, `(6, 10)`, `(5, 20)`, with
`B` in 1, 32, 256. Mixed states: `(2, 20)`, `(3, 10)`, `(3, 16)`, `(4, 12)`, with `B` in 1
and 8.

**Gates.** Displacement on mode 1 and CZ on modes (0, 1); the matrices come from the native
`gaussian_unitary`. Kerr on mode 1 as a phase gate, pure states only.

| Row | What is timed |
|---|---|
| `native_loop` | `apply_pure` / `apply_mixed` / `phase_*` state by state on a fresh copy of the batch (complex128 only). |
| `numpy` | `photographiq.gpu.fock` with `device="cpu"` (scipy CSR, the same algorithm as the GPU). |
| `gpu_end_to_end` | NumPy in, NumPy out: H2D, cuSPARSE, D2H. |
| `gpu_kernel` | Device-resident batch, `on_device=True`, then `Device().synchronize()` inside the timed region. Also reports `peak_device_pool_bytes`. |
| `gpu_h2d` / `gpu_d2h` | Upload of the batch / download of the result, synchronised. |

**Methodology.**

- **Timing.** `bench_common.time_call`, as for the Gaussian benchmark. `first_call_s`
  includes building and uploading the cached index maps.
- **Correctness first.** Before timing, every row is compared with the complex128 native
  result (`max_rel_deviation_vs_native_complex128`). The run aborts above 1e-12
  (complex128) or 1e-5 (complex64).
- **Peak memory.** `peak_device_pool_bytes` is the CuPy pool total after one call started
  from an emptied pool. It includes the batch, the cached maps, the cuSPARSE workspace and
  the output.
- **Load.** System CPU load is recorded at start and end.

**Results.** Run of 2026-11-11 (`results/gpu_fock_20261111-232613.json`): RTX 4050 Laptop
GPU, i5-12450HX shared with other jobs, load 82 % at the start and 25 % at the end.

| Case | GPU kernel / end-to-end factor vs native loop (complex128) |
|---|---|
| B >= 32 pure, all shapes and gates | 1.7-150x / 1-13x |
| B = 1 pure, D = 465 | **0.05-0.25x** / **0.04-0.2x** (fixed ~0.4-1 ms per call) |
| B = 1 pure, D = 5005-8855 | 0.5-1.9x / 0.3-1.2x (break-even) |
| B = 1 pure, D = 42504 | 3.6-8.9x / 2.6-3.2x |
| mixed, B = 1 | 1.7-8.7x / 0.9-2.9x |
| mixed, B = 8 | 1.3-14x / 1.1-6.9x |

complex64 kernels reach 610x (pure, Kerr), 210x (pure, displacement) and 46x (mixed). The full
analysis, including where the native engine spends its time, is in `docs/gpu.md`, section
"Fock kernels".
