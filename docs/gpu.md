# GPU acceleration (optional, CuPy)

`photographiq.gpu` runs **batched Gaussian kernels** and **batched Fock local gates** on an
NVIDIA GPU through
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
- **Fock kernels on single small states and whole patterns.** See
  [Fock kernels](#fock-kernels-photographiqgpufock): only batches and very large
  states benefit, and patterns are dominated by Python preparation and measurement.
- **CUDA-Q Dynamics.** CUDA-Q ships Linux wheels only and cannot be installed on native
  Windows. `photographiq.cudaq_dynamics` is a Hamiltonian-level adapter, not a backend,
  and its CUDA-Q path has not been run; see [cudaq.md](cudaq.md).
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

## Fock kernels (`photographiq.gpu.fock`)

Two kernels apply local gates to batches of Fock states, using the native Fock engine's
basis. Pure states are `(B, D)` and density matrices `(B, D, D)`, with
`D = C(modes + cutoff - 1, modes)`. A single state is a batch of one.

```python
from photographiq import gpu, native

F = native.require("fock")                       # gate matrices come from the native engine
U = F.gaussian_unitary(S, d, cutoff)             # (S_k, S_k), S_k = C(k + cutoff - 1, k)
res = gpu.apply_operator(psi, modes, cutoff, [0, 1], U)            # psi -> P U P psi
res = gpu.apply_operator(rho, modes, cutoff, [2], U1)              # rho -> U rho U^dag
res = gpu.apply_phase(psi, modes, cutoff, [1], kappa * np.arange(cutoff) ** 2)  # Kerr
res.states, res.execution        # same device / dtype / on_device contract as above
```

**What runs on the GPU.**

- **Basis.** Truncation keeps the **total** photon number (`sum n < cutoff`), in the native
  engine's order. The basis and every rank come from the native engine's own `basis` and
  `rank`. They are not re-implemented.
- **Gate matrix.** A k-mode gate acts on each fibre (fixed occupations of the other modes)
  through its top-left block, exactly as the native `apply_pure` and `apply_mixed` do. This
  is one sparse `D x D` matrix G. Its CSR structure, and the map from each entry to `U`, are
  cached per `(modes, cutoff, local modes)`. Only the values are gathered per call.
- **`apply_operator`.** One cuSPARSE sparse-dense product per call (`G @ psi^T` for all B
  states). Density matrices take two products: `G rho G^dag = G (G rho^dag)^dag`, valid for
  any matrix.
- **`apply_phase`.** An element-wise multiply, for Kerr, rotation and cross-Kerr.
- **CPU path.** The same algorithm with `scipy.sparse`.
- **Not done.** No renormalisation and no diagnostics; the input is not modified.
- **Precision.** complex128 by default. complex64 only when `dtype="complex64"` is passed,
  and `execution.dtype` records it.

**Errors.** A batch whose trailing shape is not `(D,)` or `(D, D)` for the given
`(modes, cutoff)` raises `ValueError` naming the expected dimension. So does a missing
batch axis, an operator that is not `(S_k, S_k)`, or repeated, out-of-range or non-integer
local modes. Without CuPy or a device, `device="gpu"` raises.

**Validation.** `tests/gpu/test_gpu_fock.py` compares against the native kernels
`apply_pure`, `apply_mixed`, `phase_pure` and `phase_mixed`, and against whole
`NativeFockBackend` / `NativeMixedFockBackend` gate sequences.

- Gates: displacement, squeezing, beam splitter, CZ, Kerr, cross-Kerr, and cubic phase
  (truncated-generator and projected).
- Random complex operators and non-Hermitian matrices, to test the index maps themselves.
- Shapes `(modes, cutoff)`: `(2, 8)`, `(3, 7)`, `(4, 6)`, and in the error survey also
  `(2, 20)`, `(3, 12)`, `(5, 8)`. Local modes in forward and reversed order.

Largest deviation over seeds 0-9, relative to the largest reference entry:

| Path | complex128 | complex64 |
|---|---|---|
| CPU (scipy) | 1.3e-15 | 4.4e-7 |
| GPU (cuSPARSE) | 7.2e-16 | 2.4e-7 |

Tolerances are 1e-12 for complex128 and 1e-5 for complex64. Norms and traces equal the
native engine's to 3.3e-16. For the beam splitter, Kerr and cross-Kerr, which conserve
photon number, norm and trace stay 1 to 1e-13. Density matrices stay Hermitian to 1e-14.

### Where the native Fock engine spends its time

Measured on 2026-11-11 (shared i5-12450HX, system load 40-80 %):

- **Pure-state gate.** About 30-100 µs of Python per gate (state copy, renormalisation,
  diagnostics). The rest is the C++ kernel, which scales with D: 0.06 ms at D = 715,
  3-5 ms at D = 42504 (5 modes, cutoff 20). That kernel rebuilds its fibre index maps
  (`make_fibers`) on every call, so on one large state the cached maps on the GPU already
  win, by 3.6x.
- **Two-mode gate matrix.** Building it can dominate: `gaussian_unitary` takes 4.6 ms at
  cutoff 30, against 0.28 ms for applying it on 2 modes.
- **Density-matrix gate.** The `apply_mixed` kernel dominates: 22 ms for a CZ at D = 210,
  28 ms at D = 816.
- **Whole patterns** (`bench_fock.py` wire and network patterns). Pure ones are dominated by
  Python state preparation (`_ranks`, dictionary amplitude maps), measurement projection and
  gate-matrix construction. Mixed ones are dominated by `apply_mixed`. The kernels here do
  not accelerate patterns: one state at a time is the wrong shape for a GPU.

### Measured speedups and slowdowns

`benchmarks/gpu/bench_gpu_fock.py`, run on 2026-11-11 (RTX 4050 Laptop GPU, CuPy 14.2.0).

- **Load.** System CPU load was 82 % at the start and 25 % at the end. Native timings are
  inflated by an unknown, varying amount, so treat factors below about 2 as noise.
- **Columns.** Times are median ms. *native* is a loop over B native kernel calls.
  *kernel* has the batch on the device. *e2e* is NumPy in and NumPy out. Factors are
  native time divided by GPU time (< 1 means slower).

| States | modes, cutoff (D) | B | Gate | native | NumPy | GPU kernel c128 | GPU e2e c128 | GPU kernel c64 |
|---|---|---|---|---|---|---|---|---|
| pure | 2, 30 (465) | 1 | displace | 0.050 | 0.084 | 1.09 (**0.05x**) | 1.28 (**0.04x**) | 0.95 (0.05x) |
| pure | 2, 30 (465) | 256 | cz | 101 | 201 | 14.9 (6.8x) | 17.8 (5.7x) | 9.3 (11x) |
| pure | 6, 10 (5005) | 1 | cz | 0.66 | 1.0 | 0.96 (**0.68x**) | 0.89 (**0.74x**) | 1.27 (0.52x) |
| pure | 6, 10 (5005) | 256 | cz | 368 | 219 | 6.95 (53x) | 27.9 (13x) | 7.7 (48x) |
| pure | 4, 20 (8855) | 1 | displace | 0.88 | 1.2 | 1.28 (**0.69x**) | 1.13 (**0.78x**) | 1.73 (0.51x) |
| pure | 4, 20 (8855) | 256 | displace | 392 | 237 | 11.1 (35x) | 49.5 (7.9x) | 7.0 (56x) |
| pure | 5, 20 (42504) | 1 | displace | 3.7 | 3.5 | 1.04 (3.6x) | 1.46 (2.6x) | 1.71 (2.2x) |
| pure | 5, 20 (42504) | 1 | cz | 7.3 | 30 | 2.04 (3.6x) | 2.71 (2.7x) | 8.8 (0.8x) |
| pure | 5, 20 (42504) | 32 | cz | 608 | 599 | 23.9 (25x) | 50.7 (12x) | 17.9 (34x) |
| pure | 5, 20 (42504) | 256 | displace | 1080 | 1090 | 26 (41x) | 199 (5.4x) | 5.2 (210x) |
| mixed | 3, 10 (220) | 1 | displace | 1.37 | 1.0 | 0.78 (1.7x) | 0.97 (1.4x) | 1.22 (1.1x) |
| mixed | 3, 16 (816) | 1 | cz | 91 | 405 | 26 (3.5x) | 34.7 (2.6x) | 9.5 (9.6x) |
| mixed | 3, 16 (816) | 8 | cz | 293 | 2140 | 222 (1.3x) | 273 (1.1x) | 25 (12x) |
| mixed | 4, 12 (1365) | 1 | displace | 75 | 227 | 8.7 (8.7x) | 35 (2.1x) | 3.2 (24x) |
| mixed | 4, 12 (1365) | 8 | cz | 1010 | 5230 | 292 (3.5x) | 398 (2.5x) | 49 (20x) |

All 162 cases and the H2D/D2H rows are in
`benchmarks/gpu/results/gpu_fock_20261111-232613.json`.

**Summary by candidate:**

- **(a) Batched pure states.** Justified. With B >= 32 every measured case is faster on
  the GPU: 1.7-150x kernel-only and 1-13x end to end in complex128. Transfers are a large
  part of the end-to-end time, for example 51 ms H2D and 92 ms D2H for B = 256, D = 42504.
  Keep the batch on the device across a gate sequence with `on_device=True`.
- **(b) One large pure state.** Benefit only from D ≈ 40 000: 3.6x kernel and 2.6x end to
  end at D = 42504. At D <= 9000 it is about break-even (0.5-1.9x). At D = 465 it is 4-20x slower,
  because the cost is a fixed 0.4-1 ms per call (launches and the cuSPARSE descriptor).
  Below D ≈ 10^4, use the native engine.
- **(c) Density matrices.** The single-matrix gain is moderate in complex128: 1.7-8.7x
  kernel and 0.9-2.9x end to end. The complex128 cuSPARSE product runs far below even the
  GPU's reduced FP64 rate on the dense two-mode blocks. complex64 reaches 10-46x at
  B = 8 (one small case was 0.5x) where single-precision accuracy (about 2e-7 per gate) is
  acceptable.
- **Phase gates (`apply_phase`).** These are element-wise. The NumPy path is already 10-30x
  faster than the native loop: native `phase_*` rebuilds the fibre maps on every call.
  The GPU beats NumPy only with the batch already on the device. End to end it is slower than
  NumPy, for example 183 ms against 65 ms at B = 256, D = 42504.
- **complex64 is not always faster.** For some small cases it was slower than complex128 in
  this run (mixed D = 210, B = 1, CZ: 17.6 ms against 6.3 ms). This is attributed to the
  shared machine.

**Alternatives measured and not shipped:**

- **Grouped gather + batched GEMM,** with fibres grouped by size as in the native kernel.
  Faster than cuSPARSE only for dense two-mode blocks at large B (up to 4.6x for a CZ on
  2 modes at cutoff 30, B = 1024). 2-7x slower at small B, because it needs one launch
  group per fibre size.
- **Dense G.** Fast in complex64 for D <= 1000. 3-17x slower in complex128 for sparse gates.

The first per-structure call builds and uploads the maps. It is reported as `first_call_s`
(up to 0.42 s here). Peak CuPy pool usage reached 1.1 GB for 8 density matrices at
D = 1365.

### Permanents, hafnians and torontonians on the GPU (Phase 3.3): not implemented

Measured on 2026-11-11 with a throwaway script that is not in the repository. System load
was 30 % at the start and 99 % at the end.

- **Native side.** `photographiq._native.sampling` called once per matrix, since there is
  no batched entry point.
- **GPU side.** Straightforward batched CuPy formulations, each checked against native
  first:
  - Ryser over all column subsets as one batched GEMM;
  - the power-trace hafnian, vectorised over sign vectors;
  - the torontonian by batched `det`, grouped by subset size.

The factor is the native loop's time divided by the GPU kernel time, both complex128:

| Function | n | B | native loop | GPU kernel | factor | GPU rel. error |
|---|---|---|---|---|---|---|
| permanent | 4 | 10000 | 292 ms | 6.5 ms | 45x | 3e-15 |
| permanent | 8 | 10000 | 962 ms | 53 ms | 18x | 4e-14 |
| permanent | 12 | 500 | 187 ms | 77 ms | 2.4x | 5e-13 |
| permanent | 16 | 20 | 30 ms | 23 ms | 1.3x | 2e-12 |
| permanent | 20 | 2 | 48 ms | 146 ms | **0.33x** | 1e-11 |
| permanent | 24 | 1 | 355 ms | 1.74 s | **0.20x** | 1e-11 |
| hafnian | 8 | 5000 | 194 ms | 24 ms | 8x | 4e-16 |
| hafnian | 16 | 100 | 150 ms | 26 ms | 5.9x | 2e-15 |
| hafnian | 24 | 2 | 182 ms | 43 ms | 4.3x | 1e-14 |
| torontonian | 8 | 500 | 396 ms | 31 ms | 13x | 1e-12 |
| torontonian | 12 | 20 | 119 ms | 42 ms | 2.9x | 5e-12 |
| torontonian | 16 | 1 | 160 ms | 189 ms | **0.85x** | 7e-14 |

For permanents and hafnians, n is the matrix size. For torontonians, n is the number of
modes: the matrix is `2n x 2n`.

**Verdict: not justified, so nothing ships.**

- **Single large matrices lose.** For one large permanent (n = 20-24) the GPU is 3-5x
  slower than native's Gray-code Ryser. The subset-GEMM formulation does n times more work,
  and the GPU's FP64 rate is low. complex64 is still slower at n = 24 (0.73 s against
  0.36 s) and has relative error 4e-3.
- **Batched gains measure native overhead.** The batched gains appear only for large
  batches of small matrices, and they mostly measure per-call overhead in the native path:
  - an n = 1 permanent costs 34 µs per call with 12 OpenMP threads and 7 µs with 1;
  - an n = 8 permanent (about 2k complex multiply-adds) costs 140 µs single-threaded.

  A native batched entry point, or a leaner small-n kernel, should be tried before a GPU
  path.
- **No current workload batches equal-size matrices.** `output_distribution`, `sample` and
  `pnr_distribution` evaluate patterns with different repetitions and submatrix sizes, so
  no current PhotoGraphiQ workload issues large batches of same-size matrices.
