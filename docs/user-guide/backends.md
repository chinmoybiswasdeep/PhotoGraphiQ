# Backends: choose and compare

A *backend* executes a [Pattern](patterns.md) trajectory: it prepares states, applies gates,
samples or postselects measurements and returns a conditional state. All backends implement
the same contract and the same conventions ($\hbar=2$, interleaved quadratures, vacuum
$V=I$, total-photon Fock cutoff), so a pattern runs unchanged on any backend that supports
its operations. The capability data below comes from the backend registry
(`photographiq.registry`), not from prose: query it yourself with
`photographiq.describe_backend(name)`.

## At a glance

| `backend=` | State | Engine | Device / precision | Install | Recommended for |
|---|---|---|---|---|---|
| `gaussian` (**default**) | mean + covariance | NumPy reference | CPU, float64 | core | Gaussian patterns, reference results |
| `native-gaussian` | mean + covariance | C++20 (Eigen, OpenMP), $O(nk)$ local updates | CPU, float64 | compiled `gaussian` component | large Gaussian patterns, many modes |
| `native-fock` | pure state, total-photon cutoff | C++20 | CPU, complex128 | compiled `fock` component | non-Gaussian pure-state work |
| `native-mixed-fock` | density matrix, total-photon cutoff | C++20 | CPU, complex128 | compiled `fock` component | loss, noisy homodyne, Kraus instruments, mixed inputs |
| `piquasso` | mean + covariance | Piquasso 8.0.1 | CPU, float64 | `[piquasso]` extra | cross-checks |
| `piquasso-fock` | pure state | Piquasso | CPU, complex128 | `[piquasso]` extra | cross-checks; default of `cutoff_convergence` and physical PhotoGraphiQML |
| `piquasso-mixed-fock` | density matrix | Piquasso | CPU, complex128 | `[piquasso]` extra | cross-checks |

`simulate` used `piquasso` as its default before Piquasso became optional; it now uses
`gaussian`. Both consume random numbers identically, so seeded trajectories agree to
floating-point rounding. `cutoff_convergence` still defaults to `piquasso-fock` for
reproducibility of recorded studies; pass `backend="native-fock"` or
`"native-mixed-fock"` to run it without Piquasso.

Not pattern backends: the CuPy kernels in `photographiq.gpu` (batched building blocks, see
[below](#gpu-kernels-cupy)) and the CUDA-Q Dynamics adapter (`photographiq.cudaq_dynamics`,
Hamiltonian level, execution unverified; `simulate(backend="cudaq-dynamics")` raises
`ValueError`). See [CUDA-Q adapter](../cudaq.md).

## Capabilities

Each backend declares the features it supports; `simulate` and `select_backend` check them
**before** any state is allocated, so an unsupported request raises instead of silently
changing representation.

| Feature | Gaussian (`gaussian`, `native-gaussian`, `piquasso`) | Pure Fock (`native-fock`, `piquasso-fock`) | Mixed Fock (`native-mixed-fock`, `piquasso-mixed-fock`) |
|---|---|---|---|
| Gaussian gates, quadratic phase | ✓ | ✓ | ✓ |
| Ideal homodyne, postselection | ✓ | ✓ | ✓ |
| Heterodyne, general-dyne | ✓ | — | — |
| Loss (thermal) | ✓ | — | ✓ |
| Noisy / inefficient homodyne | ✓ | — | ✓ |
| Fock inputs, multimode Fock, cat states | — | ✓ | ✓ |
| Photon counting, photon addition/subtraction | — | ✓ | ✓ |
| Cubic phase, Kerr | — | ✓ | ✓ |
| Cross-Kerr | — | native only | native only |
| Encoded GKP X/Z readout, rank-one instruments | — | ✓ | ✓ |
| Finite-outcome POVMs, custom Kraus instruments, mixed conditional updates | — | — | ✓ |
| Higher-order moments | — | ✓ | ✓ |
| Displacement frame (`frame=True`) | ✓ | — | — |

## Numerical accuracy and approximations

| Engine | Approximation | Validated accuracy (reference) |
|---|---|---|
| Gaussian (all three) | none beyond float64 roundoff | native vs analytic moments / `expm` symplectics: max error 6.5e-13 |
| Native Fock gates | total-photon truncation; renormalization after each operation (retained norm recorded; error above `norm_tolerance`, default 1e-3) | vs mpmath amplitudes / enlarged-space `expm`: 5e-16; one-mode squeeze + displace vs 60-digit mpmath: ≤ 1.9e-14 |
| Native mixed Fock | as above, on density matrices | mostly cross-implementation (Piquasso, Gaussian moments), plus an explicit partial-trace check of a Kraus instrument |
| Piquasso Fock / mixed Fock | as above | independent `expm` on an enlarged space, environment dilation, binomial loss closed form (mixed) |

Known limit: for **multi-mode displaced** Gaussian unitaries in the native Fock engine no
error bound has been derived (the one-mode factorization is bounded). Truncation error is
never hidden: check `state.retained_norms` and converge observables with
[`cutoff_convergence`](convergence.md). Evidence: [validation status](../validation/MASTER_VALIDATION_STATUS.md),
[independent validation matrix](../validation/independent-validation-matrix.md),
[native Fock Gaussian unitary report](../validation/native-fock-gaussian-unitary.md).

## Runtime and memory

Measured on one laptop (i5-12450HX, 12 threads) on 2026-11-12; see
[performance](../performance.md) and the
[Phase 4 report](../validation/PHASE4_MANUSCRIPT_REPRODUCIBILITY.md) for methodology.

- **Gaussian:** storage $O(n^{2})$. The NumPy reference rebuilds dense matrices
  ($O(n^{3})$ per gate), the native engine updates only the affected rows ($O(nk)$); a single
  beam splitter at $n=256$ modes is 7506× faster natively, while like-for-like batched work is
  1.2–10× faster.
- **Fock:** dimension $D=\binom{m+c-1}{m}$; pure states need $16D$ bytes, density matrices
  $16D^{2}$. Native Fock gates are a median 44× faster than Piquasso (73 cases). An 8-mode wire
  at cutoff 16 runs in 4.8 ms (native pure) vs 90 ms (Piquasso pure) and 58 ms vs 2.31 s
  (mixed).
- **Guards:** pure Fock refuses dimensions above `max_dimension` (default $10^{6}$); mixed
  Fock refuses density matrices above `max_matrix_bytes` (default 256 MB). Configure them on a
  backend instance.

```python
import photographiq as pg
from photographiq.backends.native_fock import NativeMixedFockBackend

engine = NativeMixedFockBackend(12, max_matrix_bytes=64_000_000)   # cutoff 12, 64 MB guard
result = pg.simulate(pg.non_gaussian.photon_subtraction(theta=0.2),
                     inputs={"in": pg.FockInput.number(2)}, backend=engine,
                     measurement_outcomes={"count": 1})
print(result.state.photon_number("in"))   # 1.0 on the heralded branch
```

## Inspect and select

```python
import photographiq as pg

# Which registered backends are usable here, and what each one is.
for name in pg.available_backends():
    info = pg.describe_backend(name)
    print(f"{name:20s} {info['representation']:10s} {info['implementation']:8s} "
          f"{info['precision']:10s} uses_cutoff={info['uses_cutoff']}")

# Let the planner choose: photon counting on a Fock input needs a Fock representation.
pattern = pg.non_gaussian.photon_subtraction(theta=0.2)
choice = pg.select_backend(pattern, inputs={"in": pg.FockInput.number(2)}, cutoff=8)
print(choice.explanation)

# Inefficient homodyne on a Fock state needs density matrices.
noisy = pg.Pattern(inputs=("a",)).extend([pg.Measure("a", pg.Homodyne(efficiency=0.9), "m")])
fock1 = {"a": pg.FockInput.number(1)}
try:
    pg.simulate(noisy, inputs=fock1, backend="native-fock", cutoff=6)
except NotImplementedError as error:
    print("native-fock:", error)
print("native-mixed-fock:", pg.simulate(noisy, inputs=fock1, backend="native-mixed-fock",
                                         cutoff=6, seed=3).outcomes)
```

```text
gaussian             gaussian   numpy    float64    uses_cutoff=False
piquasso             gaussian   piquasso float64    uses_cutoff=False
piquasso-fock        pure-fock  piquasso complex128 uses_cutoff=True
piquasso-mixed-fock  mixed-fock piquasso complex128 uses_cutoff=True
native-gaussian      gaussian   native   float64    uses_cutoff=False
native-fock          pure-fock  native   complex128 uses_cutoff=True
native-mixed-fock    mixed-fock native   complex128 uses_cutoff=True
'native-fock': cheapest available pure-fock backend covering [fock_input, photon_counting]
native-fock: NativeFockBackend does not support noisy_homodyne
native-mixed-fock: {'m': -2.3090229711269425}
```

The `piquasso*` rows appear only with the `piquasso` extra. `select_backend` tries
candidates in cost order (Gaussian before pure Fock before mixed Fock, native before
Piquasso) and skips Fock backends unless you pass a `cutoff`; its `rejected` field explains
every skipped candidate. Requesting a backend whose optional package is missing raises
`ImportError` naming the extra.

## GPU kernels (CuPy)

`photographiq.gpu` provides batched building blocks, not a pattern backend: Gaussian
symplectic updates, homodyne conditioning on given outcomes, purity, symplectic eigenvalues,
fidelity, and Fock local gates (`apply_operator`, `apply_phase`) on batches of states in the
native basis. Every call takes `device="gpu"` (default) or `device="cpu"`, raises instead of
falling back, and returns an `Execution` record with the device the output buffer lives on.

```python
import math
import numpy as np
from photographiq import gpu, sampling

# Sampling kernels: the native C++ and Python (NumPy) paths implement the same algorithm.
M = sampling.haar_unitary(8, rng=0)
print(np.isclose(sampling.permanent(M, backend="native"), sampling.permanent(M, backend="python")))

# Batched Kerr phase on 128 pure two-mode Fock states (total-photon cutoff 10, D = 55).
modes, cutoff, B = 2, 10, 128
D = math.comb(modes + cutoff - 1, modes)
rng = np.random.default_rng(0)
psi = rng.normal(size=(B, D)) + 1j * rng.normal(size=(B, D))
psi /= np.linalg.norm(psi, axis=1, keepdims=True)
device = "gpu" if gpu.available() else "cpu"
res = gpu.apply_phase(psi, modes, cutoff, [0], 0.1 * np.arange(cutoff) ** 2, device=device)
print(res.execution.device, res.states.shape, np.allclose(np.linalg.norm(res.states, axis=1), 1))
```

Precision is explicit (float64/complex128 by default; float32/complex64 only on request and
recorded in `execution.dtype`). The GPU pays off for batches (B ≥ 32) and very large single
Fock states; single small states are slower than the native CPU engine. There are no custom
CUDA kernels, no GPU pattern simulation and no GPU outcome sampling. Details and measurements:
[GPU acceleration](../gpu.md).

## Sampling kernels

Permanents, hafnians, loop hafnians and torontonians (`photographiq.sampling`) take
`backend="auto"` (native when compiled), `"native"` or `"python"`. They are independent of the
pattern backends. Threshold probabilities accept `precision=` for mpmath evaluation. See
[boson sampling and GBS](sampling.md).

## Behaviour common to all backends

- Every `simulate` call resets its backend; results are independent snapshots.
- Pure Fock reductions may be mixed, but that does not make the pure-trajectory engine a
  mixed-channel simulator.
- Native engines subclass the validated Python backends and are cross-checked against them
  and against Piquasso (`tests/native`, `tests/cross_backend`).
- Differentiable execution has a separate API: [automatic differentiation](automatic-differentiation.md).
