# GPU acceleration (optional, CuPy)

`photographiq.gpu` runs **batched Gaussian kernels** on an NVIDIA GPU through
[CuPy](https://cupy.dev). It targets workloads with many independent states at once: Monte
Carlo trajectories, parameter sweeps, or postselected branches. It is optional: the core
package never imports CuPy, and `import photographiq.gpu` does not import it either.

```bash
pip install "photographiq[gpu-cuda12]"   # NVIDIA driver with CUDA 12 (cupy-cuda12x)
pip install "photographiq[gpu-cuda13]"   # NVIDIA driver with CUDA 13 (cupy-cuda13x[ctk])
```

Or install CuPy directly: `pip install cupy-cuda12x`, or `pip install "cupy-cuda13x[ctk]"`.
The `[ctk]` extra pulls the CUDA 13 runtime libraries (`cuda-toolkit`) from PyPI; plain
`cupy-cuda13x` expects a local CUDA Toolkit.

```python
import numpy as np
from photographiq import gpu

gpu.available()      # True if CuPy imports and a CUDA device is visible
gpu.device_info()    # backend, CuPy / CUDA runtime / driver versions, name, compute capability, memory

# means (B, 2n), covariances (B, 2n, 2n): B states in PhotoGraphiQ conventions
res = gpu.condition_homodyne(means, covs, mode=2, angles=thetas, outcomes=x, efficiency=0.9)
res.means, res.covariances, res.log_densities, res.densities   # NumPy arrays
res.execution
# Execution(kernel='condition_homodyne', device='cuda:0', backend='cupy',
#           device_name='NVIDIA GeForce RTX 4050 Laptop GPU', dtype='float64',
#           returned_on_device=False)
```

## Device-reporting guarantee

- Every kernel takes `device="gpu"` (the default) or `device="cpu"`. There is no `"auto"`.
- With `device="gpu"` the call **raises** if CuPy is missing (`ImportError` naming the CuPy
  wheel for each CUDA version) or no CUDA device is usable (`RuntimeError`). It **never falls back to the CPU
  silently**.
- `device="cpu"` runs the same array code with NumPy and reports `device="cpu"`.
- Every result carries an `Execution` record. On the GPU path, `device` (`"cuda:<id>"`) is
  read from the output buffer's own device (`array.device.id`), not assumed.
  `returned_on_device` says whether the returned arrays are CuPy arrays (`on_device=True`)
  or NumPy copies.
- NumPy results are synchronised by the device-to-host copy. With `on_device=True` the work
  may still be queued on the current stream; synchronise before timing it.

The tests check this on real hardware. With CuPy working, every kernel reports
`cuda:<current device>`. With `on_device=True`, every returned buffer is device memory on that
device (`cupy.cuda.runtime.pointerGetAttributes(ptr).type == memoryTypeDevice`). With CuPy
missing or no device (both simulated), `device="gpu"` raises and only `device="cpu"` runs.

## What is accelerated

All kernels are batched over the first axis and use PhotoGraphiQ conventions: hbar = 2,
interleaved `(q0, p0, q1, p1, ...)`, statistical covariance with vacuum `V = I`, and
homodyne records `q cos t + p sin t`, calibrated to the incident quadrature with variance
`Var + (1 - eta)/eta + noise`. Modes are integer indices.

| Function | Operation | Reference it matches |
|---|---|---|
| `apply_symplectic(means, covs, modes, S)` | `d -> S d`, `V -> S V S^T` on the listed modes. `S` is one matrix or one per state. This is a local `O(B n k)` update, and the output is exactly symmetric. | `GaussianBackend.transform` |
| `condition_homodyne(means, covs, mode, angles, outcomes, efficiency, noise)` | Schur-complement conditioning on given outcomes. Returns conditioned moments and the outcome (log-)densities. | `GaussianBackend.measure(..., outcome=x)` |
| `homodyne_moments(...)` | Mean and variance of the calibrated record. | `GaussianState.quadrature` plus calibration |
| `homodyne_log_density(...)` | Log-likelihood of outcomes, without conditioning. | `scipy.stats.norm.logpdf` |
| `purity(covs)` | `det(V)^(-1/2)` via batched `slogdet`. | `GaussianState.purity` |
| `symplectic_eigenvalues(covs)` | Cholesky, then a Hermitian eigensolver. Raises for non-positive-definite input. | `GaussianState.symplectic_eigenvalues` |
| `fidelity(m1, V1, m2, V2)` | Squared Uhlmann fidelity: overlap for pairs with a pure state, BBP formula otherwise. Not fully batched on the GPU: CuPy 14's `linalg.eigvals` (cuSOLVER `Xgeev`) loops in Python over the B matrices, one solver call each. | `GaussianState.fidelity` |

**Precision is explicit.** `dtype="float64"` is the default. `dtype="float32"` is used only
when asked, and the result's `execution.dtype` records it. GeForce-class GPUs run FP64 at a
small fraction of their FP32 rate (1/64 on the Ada generation, which includes the RTX 4050).
That caps float64 speedups. float32 is faster, but its relative error is about 5e-8 at the
moderate squeezing of the benchmark states, and it grows with squeezing.

**Cheap validity checks raise `ValueError`:**

- symplecticity of the gate matrices (`numpy.allclose`, evaluated in float64 before any
  cast);
- a positive, finite homodyne variance;
- finite homodyne outcomes (NaN/inf raise, as in `GaussianBackend.measure`);
- integer mode indices (`operator.index`, so `1.7` raises `TypeError` instead of selecting mode 1);
- `0 < efficiency <= 1`;
- finite diagnostics.

Each check costs one device-to-host synchronisation (the homodyne variance and outcome checks share one).
Unlike `GaussianBackend.measure`, `condition_homodyne` does not raise when an outcome's density
underflows to 0: `densities` is 0, `log_densities` stays finite, and the conditioned moments are
returned. Covariances are assumed symmetric and
physical: no `O(n^3)` physicality check runs on input. Outputs of `condition_homodyne` were
checked to be symmetric with `V + i Omega >= 0`.

## What is not accelerated, and why

- **Single-trajectory simulation** (`simulate`, `run_shots`). One small state per step is
  dominated by kernel-launch and synchronisation latency. Use `NativeGaussianBackend` (C++)
  instead.
- **Sampling of outcomes.** The kernels condition on **given** outcomes, so a batch is a set
  of postselected branches. CuPy's random streams differ from NumPy's, so sampled
  trajectories could not be reproduced across devices. Not implemented.
- **Fock kernels.** Not implemented. PhotoGraphiQ truncates the **total** photon number
  (`sum n < cutoff`), so a single-mode gate on a multimode state is not a plain tensor
  contraction. The single-mode case would be a trivial matrix product of little value.
  Neither kernel is validated, so neither ships.
- **CUDA-Q Dynamics.** CUDA-Q ships Linux wheels only and cannot be installed on native
  Windows. See `photographiq.backends.cudaq_dynamics`, a probe that executes nothing.
- **Custom compiled CUDA kernels.** There are none, because `nvcc` / the CUDA toolkit is
  absent. CuPy's runtime NVRTC compilation (`RawKernel`, `ElementwiseKernel`) is available
  but not used. The kernels are compositions of cuBLAS batched GEMM, cuSOLVER batched
  factorisations and CuPy element-wise/gather operations. A fused kernel would mainly cut
  launch overhead for small batches. Whether that is worth it should be decided from the
  benchmark, not assumed.

## Validation evidence

Tests are in `tests/gpu/test_gpu_gaussian.py` and `tests/gpu/test_gpu_device.py`: 68 tests,
run on an RTX 4050 Laptop GPU (compute capability 8.9) with CuPy 14.2.0, CUDA runtime 13.2 and
driver 13.0. Every numerical test runs for `device="cpu"` and `device="gpu"`. GPU cases skip
with a reason when CuPy or a device is missing.

**Reference agreement.** Results are compared with the NumPy reference backend
(`GaussianBackend` / `GaussianState`), state by state. The tests use random physical states,
batch sizes 1, 7 and 33, single-mode, multimode, permuted and all-mode gates, random angles,
efficiencies 1, 0.9 and 0.5, and noise 0 and 0.2. Largest deviations observed over seeds
0-19, relative to the largest reference entry:

| Quantity | CPU path | GPU path |
|---|---|---|
| symplectic update (means / covariances) | 4e-16 / 7e-16 | 4e-16 / 5e-16 |
| homodyne conditioning (means / covariances) | 3e-15 / 9e-16 | 3e-15 / 9e-16 |
| homodyne densities | 1e-14 | 1e-14 |
| purity | 0 | 7e-14 |
| symplectic eigenvalues | 8e-16 | 4e-14 |
| fidelity (mixed pairs) | 0 | 8e-11 |

The GPU fidelity deviation comes from the general (non-Hermitian) eigensolver in the BBP
formula, which is ill-conditioned for nearly pure states (see `GaussianState.fidelity`). The
test tolerances are rtol 1e-11 (fidelity 1e-8), at least 100 times the observed deviations.

**Other checks:**

- GPU and CPU paths agree to rtol 1e-12 on matrices, and to rtol 1e-11 on per-state
  scalars (fidelity 1e-8).
- Both paths agree with the native C++ batch kernels
  (`native_gaussian.batch_apply_symplectic` / `batch_condition_homodyne`).
- float32 is explicit and within 1e-4 / 1e-5 of float64 on the test states.
- Invalid input raises `ValueError`.

## Performance

`benchmarks/gpu/bench_gpu_gaussian.py` compares NumPy, native C++/OpenMP and GPU on identical
batched tasks. It times kernel and transfer separately; methodology is in
`benchmarks/gpu/README.md`. **No performance numbers are quoted here yet.** The development
machine was shared during implementation, so only the `--smoke` configuration was run, to
check that the script works. In that configuration, very small batches were dominated by a
fixed per-call GPU overhead (kernel launches, plus one synchronisation per validity check).
The GPU only pays off for large `B * n^2`.
