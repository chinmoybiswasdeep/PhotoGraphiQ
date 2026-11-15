# CUDA-Q Dynamics adapter

`photographiq.cudaq_dynamics` is an optional adapter at the **Hamiltonian level**. It
writes PhotoGraphiQ gates as bosonic Hamiltonians and passes them to CUDA-Q Dynamics
(`cudaq.evolve`). It is **not a pattern backend**. CUDA-Q cannot execute PhotoGraphiQ
patterns, because it has no homodyne or photon-counting conditioning. For this reason
the registry has no CUDA-Q entry, and `simulate(backend="cudaq-dynamics")` raises
`ValueError` with the name of this module.

**Status: CUDA-Q execution is UNVERIFIED.** CUDA-Q publishes Linux (x86_64/aarch64)
wheels only. The development machine runs Windows without WSL, so `to_cudaq` and
`evolve` have never run against a real CUDA-Q installation. The conventions are
validated through the NumPy/SciPy reference path against `native-fock` (see
[Validation](#validation)). Nothing on this page claims GPU simulation.

## API

```python
import numpy as np
import photographiq.cudaq_dynamics as cq

h = cq.beamsplitter(0, 1, 0.4) + cq.kerr(0, 0.1)        # BosonicHamiltonian
dims = {0: 12, 1: 12}                                     # CUDA-Q style per-mode dims
psi = np.zeros(144, complex)
psi[np.ravel_multi_index((2, 1), (12, 12))] = 1
loss = [np.sqrt(0.3) * cq.annihilate(1)]                  # loss at rate gamma = 0.3

ref = cq.reference_evolve(h, dims, [0.0, 0.5, 1.0], psi, collapse_operators=loss,
                          observables=[cq.number(0)], store_intermediate_results=True)
# Same call on CUDA-Q (Linux only): cq.evolve(h, dims, [0.0, 0.5, 1.0], psi, ...)
```

| Name | Purpose |
|---|---|
| `BosonicHamiltonian` | Sum of complex coefficients times products of ladder operators. Terms are stored normal-ordered as `{((mode, creations, annihilations), ...): coefficient}`. Supports `+`, `-`, `*` (scalars and operators, normal-ordered with [a_i, a_j^dag] = delta_ij), `**`, `dagger()`, `is_hermitian(atol)` and `word(key)`, which gives the (mode, dagger) ladder word. |
| `create`, `annihilate`, `number`, `position`, `momentum` | Building blocks. `position` = a + a^dag and `momentum` = -i(a - a^dag) (PhotoGraphiQ, hbar = 2). |
| Gate generators | `rotation`, `displacement`, `squeezing`, `beamsplitter`, `controlled_z`, `quadratic_phase`, `kerr`, `cross_kerr`, `cubic_phase` (table below). |
| `to_matrix(h, dimensions)` | Dense matrix on the tensor product of `{mode: d}`. Modes are in increasing label order, and the lowest label is the most significant index (`np.kron` order). |
| `per_mode_to_total_photon(state, dimensions, cutoff)` | Projects a per-mode vector or density matrix onto sum(n) < cutoff, in the native basis order. The result is not renormalised. |
| `total_photon_to_per_mode(state, dimensions, cutoff)` | Inverse embedding. Every local dimension must be >= cutoff. |
| `reference_evolve(...)` | NumPy/SciPy evolution. It applies `scipy.sparse.linalg.expm_multiply` to -iH (pure states) or to the Lindblad superoperator (density matrices) between consecutive times. |
| `to_cudaq(h)` | Builds the `cudaq.operators.boson` expression. `cudaq` is imported lazily. |
| `evolve(...)` | Runs `cudaq.evolve` on the `dynamics` target. It never falls back to the reference. |
| `cudaq_probe()`, `cudaq_capabilities()` | Report availability and the documented features. They are also re-exported from the compatibility shim `photographiq.backends.cudaq_dynamics`. |

Both evolution functions take `(hamiltonian, dimensions, times, initial_state, *,
collapse_operators=(), observables=(), store_intermediate_results=False)`. `evolve` also
takes `integrator=`, which is passed through to CUDA-Q. Both return
`EvolutionResult(state, times, intermediate_states, expectation_values, record)`.
`initial_state` is a vector or a density matrix in `to_matrix` order. A vector becomes a
density matrix when collapse operators are given. H must be Hermitian and
time-independent. `record["engine"]` is `"numpy-reference"` or `"cudaq"`. For CUDA-Q,
`record` holds `cudaq_version` and the `target`, `simulator`, `platform` and
`description` that `cudaq.get_target()` reports, plus `verified: False`.

## Conventions

PhotoGraphiQ uses hbar = 2, q = a + a^dag, p = -i(a - a^dag). Each gate is U = exp(-i H)
at t = 1. Each sign and scale comes from the native implementation
(`src/photographiq/backends/native_fock.py`) and is checked numerically against it.

| Gate (native method) | Native definition | Generator H | Max error vs native |
|---|---|---|---|
| `Rotate` (`rotate`) | exp(i angle n) | -angle n | 1.4e-13 |
| `Displace` (`displace`) | D(alpha), alpha = (q + i p)/2 | i(alpha a^dag - alpha* a) | 2.3e-16 |
| `Squeeze` (`squeeze`) | S = diag(e^-r, e^r) on (q, p) | i r (a^2 - a^dag^2)/2 | 3.4e-16 |
| `BeamSplitter` (`beamsplitter`) | q_u' = c q_u - s q_v | i theta (a_u a_v^dag - a_u^dag a_v) | 2.8e-16 |
| `Entangle` (`entangle`, CZ) | exp(i g q_u q_v / 2) | -g q_u q_v / 2 | 2.8e-16 |
| `QuadraticPhase` (`quadratic_phase`) | exp(i s q^2 / 4) | -s q^2 / 4 | 5.6e-16 |
| `Kerr` (`kerr`) | exp(i kappa n^2) | -kappa n^2 | 2.7e-14 |
| cross-Kerr (`cross_kerr`, backend method) | exp(i kappa n_u n_v) | -kappa n_u n_v | 3.9e-15 |
| `CubicPhase` (`cubic_phase`, `cubic_phase="projected"`) | exp(i gamma q^3 / 6) | -gamma q^3 / 6 | 6.7e-16 |

PhotoGraphiQ has no two-mode squeezing command, so the adapter has no generator for it.
The errors are max |psi_native - e^{i phi} psi_ref| over three random states that fill
the cutoff-6 total-photon space. The reference propagates on enlarged per-mode dimensions
(60 and 70 for one mode, 30 and 40 for two modes) and is then projected and
renormalised. The phase phi is a global phase. The native Gaussian gates define their
matrices only up to a global phase (the vacuum amplitude of the displacement-free part
is real and positive), so constant terms such as the +1 in q^2 = a^2 + 2 a^dag a +
a^dag^2 + 1 are not tested. Negative controls (the generator with the opposite sign)
fail with an error above 1e-3.

The native `cubic_phase="truncated_generator"` default is expm(i gamma Q^3/6), where Q is
the cutoff x cutoff truncated position matrix. `to_matrix(cubic_phase(...))` gives the
exact projected elements of q^3, which are not the same as Q^3. Both converge to the
exact gate. The tests check the default against expm of Q^3 explicitly.

CUDA-Q documents `position` = (a^dag + a)/2 and `momentum` = i(a^dag - a)/2. These are
half of PhotoGraphiQ's q and p. `to_cudaq` therefore uses only `create`, `annihilate`
and `identity`, never CUDA-Q's `position` or `momentum`.

Loss: the collapse operator sqrt(gamma) a gives the pure-loss channel with
transmissivity T = exp(-gamma t), which is what `NativeMixedFockBackend.loss(node, T)`
applies. Thermal noise is not translated.

## Truncation differences

CUDA-Q truncates each mode to a local dimension (`dimensions={mode: d}`).
PhotoGraphiQ Fock backends keep occupations with total photon number sum(n) < cutoff,
ordered by total-photon sectors with the first mode descending inside each sector (the
native basis order). The two truncations give different operators, so a gate evaluated
on per-mode dimension d is not the native gate at cutoff d. The tests use the per-mode
evolution on enlarged dimensions, project it to the total-photon space and then compare
it with native. A Hamiltonian that conserves or lowers the photon number (rotation, Kerr,
cross-Kerr, beam splitter, loss) maps the total-photon subspace into itself. For these,
d = cutoff is exact.

## Validation

`tests/cudaq_adapter/test_cudaq_adapter.py` checks the following. (The directory is not
called `tests/cudaq`. Pytest puts `tests/` on `sys.path`, so a directory with that name
would shadow the real `cudaq` package.)

- Every generator: `expm(-i H)` on enlarged per-mode dimensions, projected to the
  total-photon subspace, equals the `native-fock` gate on random states to 1e-10, at two
  dimensions (truncation convergence). A sign-flipped generator is rejected.
- Lindblad loss through `reference_evolve` (with and without a commuting rotation)
  equals `native-mixed-fock` rotation followed by `loss(T = exp(-gamma t))` (measured
  error 6e-17).
- The basis mapping round-trips, the basis equals the native `basis(modes, cutoff)` and
  each index is checked explicitly.
- Generators are Hermitian, and the normal-ordering algebra is checked.
- Without `cudaq`, `to_cudaq` and `evolve` raise `ImportError` naming the Linux-only
  wheels. A `cudaq` older than 0.16 is refused.
- Translation wiring only: a fake minimal `cudaq` module injected into `sys.modules`
  checks that `to_cudaq` emits the expected `create`/`annihilate`/`identity`/`empty`
  calls and builds the same operator algebra. This test does not exercise CUDA-Q.
- `test_real_cudaq_evolve_matches_reference` compares `evolve` with `reference_evolve`
  for unitary and lossy evolution. It runs only when `cudaq` is importable, so it was
  skipped here.

## CUDA-Q API relied on (unverified here)

These calls come from the documentation of CUDA-Q 0.16.0 (the "latest" documentation
read on 2026-11-11): the
[Dynamics guide](https://nvidia.github.io/cuda-quantum/latest/using/dynamics.html), the
[Python API](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html)
and the `cudaq.operators.boson` bindings in the CUDA-Q source. The adapter refuses
versions below 0.16.

- `cudaq.operators.boson.create(target)`, `annihilate(target)`, `identity()`, `empty()`.
  The adapter combines them with `*`, `+` and complex scalars.
- `cudaq.evolve(hamiltonian, dimensions, schedule, initial_state, collapse_operators=[...],
  observables=[...], store_intermediate_results=cudaq.IntermediateResultSave.ALL | NONE,
  integrator=None)`.
- `cudaq.Schedule(steps, ["t"])`, `cudaq.State.from_data(numpy_array)`,
  `cudaq.set_target("dynamics")` and `cudaq.get_target()` (`name`, `simulator`,
  `platform`, `description`).
- `EvolveResult.final_state()`, `intermediate_states()`, `final_expectation_values()`,
  `expectation_values()`, `ObserveResult.expectation()`, `State.to_numpy()`.
- `operator.to_matrix(dimensions)`. Before each run, `evolve` compares the CUDA-Q
  matrix of the probe sum_m w_m n_m with `to_matrix`. If CUDA-Q orders degrees
  differently, `evolve` raises `RuntimeError` instead of returning a permuted state.

The unverified parts are the following. Any of them can fail on a real installation,
and `test_real_cudaq_evolve_matches_reference` is the check for them:

- that the documented signatures and enum names match the installed CUDA-Q;
- that the state vector or density matrix CUDA-Q evolves uses the same degree ordering
  as `to_matrix` on the operator;
- that `State.from_data` accepts NumPy arrays on the `dynamics` target (the
  documentation examples use CuPy);
- the integrator accuracy (`evolve` passes `integrator=None`, so CUDA-Q chooses its
  default);
- the device that CUDA-Q reports.
