# Select a backend

| `backend` | Representation | Implementation | Needs |
|---|---|---|---|
| `gaussian` (default) | means/covariances | NumPy reference | core |
| `native-gaussian` | means/covariances | C++20 (Eigen, OpenMP) | compiled `gaussian` component |
| `native-fock` | pure total-cutoff Fock trajectories | C++20 | compiled `fock` component |
| `native-mixed-fock` | density matrices | C++20 | compiled `fock` component |
| `piquasso` | Gaussian gates plus exact measurement adapter | Piquasso | `photographiq[piquasso]` |
| `piquasso-fock` | pure total-cutoff Fock trajectories | Piquasso | `photographiq[piquasso]` |
| `piquasso-mixed-fock` | density matrices | Piquasso | `photographiq[piquasso]` |
| `cudaq-dynamics` | bosonic dynamics (probe only) | CUDA-Q | Linux `cudaq`; never executes patterns |

`photographiq.available_backends()` lists the backends usable in the current
environment, `describe_backend(name)` reports representation, device, precision,
approximations and capabilities, and `select_backend(pattern)` picks a backend that
supports every operation the pattern needs. A backend whose optional package is not
installed raises `ImportError` with the install command when it is requested.

The default changed from `piquasso` to `gaussian` when Piquasso became optional. Both
are exact Gaussian simulators that consume random numbers identically, so seeded
trajectories agree to floating-point rounding (checked on homodyne, heterodyne and
inefficient-homodyne patterns with feed-forward). The native engines subclass the
validated Python backends and are cross-checked against them and against Piquasso
(`tests/native`, `tests/cross_backend`).

Pass a backend instance to configure resource guards. Every simulation resets its
backend; results are independent snapshots. Capability checks run before state
preparation, so unsupported requests do not silently change representation.

Displacement frames are available only on Gaussian backends. Pure Fock reductions
may be mixed, but that does not make the pure trajectory engine a mixed-channel
simulator. Differentiable execution has a separate API in
[automatic differentiation](automatic-differentiation.md).
