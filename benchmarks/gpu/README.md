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
