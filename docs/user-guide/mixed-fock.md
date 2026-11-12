# Mixed Fock evolution

Select a mixed-Fock backend and an explicit cutoff. Two backends evolve density
matrices with the same PhotoGraphiQ semantics:

| Backend | Engine | Requires |
|---|---|---|
| `piquasso-mixed-fock` | Piquasso's mixed simulator | `piquasso` |
| `native-mixed-fock` | compiled C++ engine (`photographiq._native`, `fock` component); Piquasso is never imported | a native build ([building native engines](../development/native-build.md)) |

Both implement destructive conditional counting/homodyne locally and support pure
and mixed inputs, Gaussian and nonlinear gates, thermal attenuation, noisy
homodyne and finite-outcome instruments with general Kraus operators.

```python
import photographiq as pg
pattern = pg.Pattern(inputs=(0,)).append(pg.Loss(0, transmissivity=0.6))
result = pg.simulate(pattern, inputs={0: pg.FockInput.number(2)},
                     backend="native-mixed-fock", cutoff=5)
print(result.state.probabilities)
```

The nonzero probabilities are 0.16 for vacuum, 0.48 for one photon, and 0.36 for
two photons: the binomial loss distribution. `state_vector` is unavailable for a
mixed state; inspect `density_matrix`, moments, Wigner functions or reductions.

Loss transmissivity is intensity transmission eta. Thermal photons specify the
environment occupation. Homodyne detector efficiency acts as loss on the detected
mode; the reported outcome is rescaled to incident-quadrature units, matching the
Gaussian backend. Added electronics noise is a classical convolution of the
conditional operator, not a random displacement of the surviving state.

The matrix has D² complex entries, where D=comb(modes+cutoff-1,modes). A configurable
`max_matrix_bytes` guard bounds one matrix, not all native workspaces. Start with
one or two modes. Trace drift, boundary weight and cutoff sensitivity all matter.
Fock heterodyne/general-dyne remain unsupported.

## `native-mixed-fock`

`NativeMixedFockBackend` reuses every Python code path of the Piquasso-backed
backend (preparation, renormalisation with retained-norm diagnostics, counting,
homodyne, instruments, GKP readout, ladder operations) and replaces only the state
object and the gate kernels. It accepts the same `max_matrix_bytes` guard and the
pure-backend settings, plus `cubic_phase="truncated_generator"` (default, identical
to Piquasso's truncated generator) or `"projected"` (exact projected elements, so
leakage above the cutoff appears in the retained-norm diagnostics) and
`cubic_tolerance`. It also declares `cross_kerr`, available as the backend method
`cross_kerr(u, v, kappa)`; there is no Pattern command for it.

The numerics differ from Piquasso in documented ways:

* Gaussian gates and inputs use exact Fock matrix elements of the whole Gaussian
  unitary, applied as P U P on the truncated space. Piquasso applies CZ,
  QuadraticPhase and general Gaussian transforms through a decomposition with a
  truncation after each factor. Both converge to the same state as the cutoff
  grows; at a fixed cutoff they differ at truncation level.
* A mixed `GaussianInput` is built as U (sum_k p_k |k><k|) U^dag from its Williamson
  form, with thermal weights summed until the omitted trace-norm tail is below
  1e-15. A thermal occupation needing more than 4096 terms raises
  `NotImplementedError`.
* Thermal loss uses the same Kraus decomposition as the Piquasso-backed backend,
  evaluated in C++.

`tests/native/test_native_mixed_fock_backend.py` checks that pure inputs reproduce
the pure-backend state, that noisy channels, mixed inputs, noisy homodyne, ladder
operations and counting match `piquasso-mixed-fock` (density matrices to 1e-10 at
cutoff 10 and 1e-12 at cutoff 12), that loss reproduces exact Gaussian moments, and that states stay
physical. The full derivation and measured error figures are in the
`photographiq.backends.native_fock` module docstring.

## Hardening tests

Hardening tests cover n=1 through 4 attenuation, coherent attenuation, thermal
environment dilation, independent unitaries, and correlated noisy-homodyne
Schur complements. These regimes do not establish dense-state scalability or
uniform accuracy for every input. Inspect pre-normalization retained trace;
normalizing a truncated thermal tail does not turn it into physical photon loss.
