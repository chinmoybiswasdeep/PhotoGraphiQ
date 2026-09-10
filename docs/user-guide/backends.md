# Select a backend

| `backend` | Representation | Typical use |
|---|---|---|
| `gaussian` | NumPy means/covariances | independent Gaussian calculations |
| `piquasso` (default) | native Gaussian gates plus exact measurement adapter | Gaussian photonic MBQC |
| `piquasso-fock` | pure total-cutoff Fock trajectories | ideal nonlinear resource experiments |
| `piquasso-mixed-fock` | density matrices | lossy/mixed non-Gaussian trajectories |

Pass a backend instance to configure resource guards. Every simulation resets its
backend; results are independent snapshots. Capability checks run before state
preparation, so unsupported requests do not silently change representation.

Displacement frames are available only on Gaussian backends. Pure Fock reductions
may be mixed, but that does not make the pure trajectory engine a mixed-channel
simulator. Differentiable execution has a separate API in
[automatic differentiation](automatic-differentiation.md).
