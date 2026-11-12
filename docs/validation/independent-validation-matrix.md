# Independent validation matrix: Gaussian and Fock engines

Peer-review audit, phases 2.1 and 2.2. It covers the NumPy `gaussian`, C++ `native-gaussian`,
`native-fock` and `native-mixed-fock` engines, and the Piquasso reference backends
(`piquasso`, `piquasso-fock`, `piquasso-mixed-fock`).

The new tests are in `tests/validation/` (`test_val_gaussian.py`, `test_val_fock.py`):
128 passed, 2 skipped (Piquasso has no cross-Kerr), about 30 s
(`pytest -q -p no:cacheprovider tests/validation`). Every reference value in them is computed
inside the test from first principles with NumPy, SciPy or mpmath. No PhotoGraphiQ function
is used to build a reference. The engines are only run, and their own outputs and
diagnostics are read back as the quantities under test.

## 1. Conventions, checked against the source

| Convention | Where in the source | Checked by | Result |
|---|---|---|---|
| hbar = 2, [q, p] = 2i, q = a + a^dag, p = -i(a - a^dag) | `gaussian.py:10` (`omega`); `native_fock.py` module docstring | `test_conventions_hbar2_commutator_and_vacuum` (dense Fock commutator) | consistent |
| Interleaved (q1, p1, q2, p2, ...) | `gaussian.py:10`, `states.py` (`GaussianState`) | same test, plus every gate test via `embed` | consistent |
| Vacuum covariance I; physicality V + i Omega >= 0 | `states.py:180` | Fock-moment vacuum; random-circuit physicality | consistent |
| Coherent mean (2 Re a, 2 Im a) | `states.py:356-365`; Fock `alpha = (q + i p)/2` (`fock.py:426,480`, `native_fock.py:290`) | dense Fock `<q>, <p>` of expm(a a^dag - a* a)\|0>; mpmath amplitudes on the Fock engines | consistent on all 7 backends |
| Squeezing: gate `squeeze(r)` S = diag(e^-r, e^r); `GaussianInput.squeezed(r, angle)` = R diag(e^-2r, e^2r) R^T; resource `prepare(node, r)` is p-squeezed | `gaussian.py:19`, `states.py:368`, Gaussian `prepare` (`squeezed(-r)`), `native_fock._single_mode_gaussian`, `fock.py` `pq.Squeezing(r=-squeezing)` | generator exp(r(a^2 - a^dag^2)/2); mpmath squeezed-vacuum amplitudes, including the default resource on every Fock engine | consistent |
| CZ: p_u += g q_v, p_v += g q_u, U = exp(i g q_u q_v / 2) | `gaussian.py:26`; `native_fock.entangle`; `fock.py` `GaussianTransform(passive, active)` | symplectic matrix derived from the generator; Fock expm on an enlarged space | consistent. `piquasso-*-fock` differ from P U P at truncation level (2e-5 at cutoff 10, the documented Euler-factor truncation), which is not a convention error |
| Beam splitter q_u' = c q_u - s q_v | `gaussian.py:30`, `native_fock.beamsplitter` | generator theta(a_v^dag a_u - a_u^dag a_v) on the Gaussian and Fock sides | consistent |
| Rotation exp(i phi n): q -> q cos phi - p sin phi | `gaussian.py:14`, `native_fock.rotate` | generator i phi n | consistent |
| Homodyne records q cos(theta) + p sin(theta) | `backends/gaussian.py:133`; `fock_measurements.homodyne_projection` (weights e^{-i theta n}) | Gaussian: rotate by -theta, then read q; Fock: closed-form coherent-state density; `quadrature()` on both state types | consistent |
| Inefficient homodyne: added variance (1 - eta)/eta (+ `noise`), outcome calibrated to the incident quadrature | `backends/gaussian.py:137`; native kernel; `mixed_fock.py:288` (loss eta, then rescaling by sqrt(eta)) | Gaussian: loss-beam-splitter dilation with a vacuum ancilla; mixed Fock: N(mu, 1 + (1-eta)/eta + noise) | consistent (see finding F2 about a comment) |
| Heterodyne outcome (q, p) has covariance V + I; density = Husimi Q / 4 in (q, p) units | `backends/gaussian.py:154`; `measurements.Heterodyne` docstring ("vacuum outcome covariance 2I") | 50:50 dilation with two ideal homodynes; \|<alpha\|psi>\|^2/(4 pi) from dense Fock | consistent |
| Loss X = sqrt(eta) I, Y = (1 - eta)(2 n_th + 1) I | `backends/gaussian.py:193`; native kernel; `pq.Attenuator`; Fock Kraus (`mixed_fock.loss`, native `thermal_loss`) | analytic X, Y plus beam-splitter dilation with a thermal environment (Gaussian and Fock) | consistent |
| Fock truncation keeps occupations with total photon number < cutoff | `fock.py:246` (`dimension`), native `basis` | set equality with `itertools.product` filtered by sum < c, and dim = C(m+c-1, m) | consistent |

No module encodes any of these conventions inconsistently. Section 5 lists the API and
documentation issues found during the audit.

## 2. Reference types

- **Independent analytic**: a closed form written in the test (for example mpmath coherent and
  squeezed amplitudes, X/Y loss, TMSV log-negativity, the teleportation map T(k)).
- **Independent numeric**: a first-principles computation that shares no code with the
  package, for example `scipy.linalg.expm` of the generator on an enlarged space, dense Fock
  moments, a beam-splitter dilation, precision-matrix conditioning, or trapezoidal quadrature
  of Hermite functions.
- **Defining property**: a structural identity that any correct engine must satisfy (trace 1,
  Hermitian, positive semidefinite, preserved inner products).
- **Cross-implementation**: compared with another PhotoGraphiQ backend, or with Piquasso.
  This catches disagreement between engines, but not an error they share.
- **Regression-only**: compared with stored or previously computed values, or with the
  package's own formulas.

## 3. Existing coverage before this audit, and which reference it used

| Area (existing tests) | Engines exercised | Reference type |
|---|---|---|
| `tests/analytical/test_primitives.py` vacuum/squeeze/displace/rotate, graph nullifiers | gaussian, piquasso | independent analytic (hand numbers) |
| `tests/analytical/test_primitives.py::test_beamsplitter_and_loss_agree` | gaussian vs piquasso | cross-implementation |
| `tests/analytical/test_measurements.py` homodyne Schur complement | gaussian | analytic, but the test repeats the same Schur formula the backend uses (weakly independent); sampling checks are statistical |
| `tests/analytical/test_channels.py` compiled channels | analysis layer | regression-only (expected values are `photographiq.gaussian` matrices and `wire_channel`) |
| `tests/analytical/test_gaussian_diagnostics.py`, `tests/native/test_native_gaussian_diagnostics.py` | GaussianState, native functions | independent analytic (BBP oracle, TMSV, thermal closed forms) plus cross-implementation (numpy, Piquasso fidelity) |
| `tests/native/test_native_gaussian_backend.py` | native-gaussian | cross-implementation with the `gaussian` backend (same formulas); one tiny-variance closed form |
| `tests/protocols/*` | gaussian, piquasso, fock | cross-implementation (Gaussian vs Fock postselection), closed-form CZ pair, ensemble teleportation (statistical) |
| `tests/cross_backend/test_cross_backend_agreement.py` | all 7 | cross-implementation with the `gaussian` backend, converted to the Fock basis by Piquasso; closed-form cat counting and homodyne densities |
| `tests/native/test_native_fock_kernels.py` | native C++ kernels (not the backend classes) | independent analytic/numeric: coherent and squeezed closed forms, expm of D and S, Heisenberg action, thermal-loss closed forms, cubic by direct integration |
| `tests/native/test_native_fock_backend.py`, `test_native_mixed_fock_backend.py` | native-fock, native-mixed-fock | mostly cross-implementation (Piquasso, `gaussian` moments); explicit partial trace for one Kraus instrument |
| `tests/mixed_fock/test_independent_unitaries.py`, `test_loss_channels.py` | **piquasso-mixed-fock only** | independent numeric (expm on cutoff+4, environment dilation), binomial closed form |
| `tests/fock/*` (`reference.py` expm oracle, homodyne, cats, Wigner) | **piquasso-fock** | independent analytic/numeric |
| `tests/piquasso_reference/*` | piquasso | cross-implementation (raw Piquasso programs) |

Gaps this audit closes:

- At the backend level, the native Fock engines had no independent validation of their gates
  (only kernel-level checks and agreement with Piquasso).
- Gaussian homodyne and heterodyne conditioning were only checked against the same Schur
  formula or against other engines, never against a physical dilation. This was also true
  for inefficiency.
- The Gaussian gate matrices were never derived independently from their generators.
- Retained-norm reporting had not been validated against an exact truncation reference.

## 4. New validation matrix (measured on this machine)

Errors are absolute maximum element differences, except two kinds of row:

- `:density` and `husimi` rows give relative errors.
- `random 5-mode circuit (rel.)` gives the covariance error relative to max |V|.

`retained:*` rows compare the engine's reported retained norm (`retained_norms` /
`diagnostics[...]["retained_norm"]`) with the exact ||P U psi||^2 of the reference. A
`reference convergence` row bounds the enlarged-space reference itself: it compares D with a
larger D after projection onto the total-photon subspace.

The enlarged-space sizes for Fock gates, all at cutoff 10:

- Single-mode gates: 40 levels per mode, checked against 56.
- Two-mode gates (CZ, beam splitter, cross-Kerr): 26 levels per mode (676-dimensional expm),
  checked against 32.

The truncation effect is accounted for exactly. The engines apply P U P and renormalise, so
the reference is P U psi / ||P U psi||. Convergence to the untruncated gate is shown as
1 - F at increasing cutoff:

- displacement + CZ: 1 - F decreases monotonically over cutoffs 8, 12, 16 and 20, reaching
  2.7e-11.
- cubic phase: 1 - F reaches 5e-7 at cutoff 28 for both definitions.

**Independence caveats (please read before relying on the Piquasso rows):**

- The `piquasso` Gaussian backend runs gates, resource squeezing and loss through Piquasso.
  It **reuses the NumPy code** for `GaussianInput` preparation (`GaussianBackend.prepare`),
  for all measurements (`GaussianBackend.measure`), and for state diagnostics (`GaussianState`
  methods). Its rows for states, measurements, feed-forward and diagnostics therefore
  validate the shared NumPy path a second time. They are not an independent Piquasso
  implementation.
- `native-gaussian` has its own C++ kernels for gates, loss, homodyne and generaldyne
  conditioning and diagnostics. The rows labelled "native_gaussian fn" call those kernels
  directly.
- `native-mixed-fock` prepares pure non-Gaussian inputs (`FockSuperposition`, `FockInput`)
  through a `NativeFockBackend` helper before tensoring. Its gate, loss, measurement and
  partial-trace kernels are separate.
- Piquasso Fock rows are a secondary check only, with looser tolerances (1e-10; 1e-4 for CZ).

### Gaussian

| Operation | Backend | Reference type | Test | Max error | Tolerance |
|---|---|---|---|---|---|
| channel:loss | gaussian | independent analytic | `test_val_gaussian.py::test_loss_channel_against_xy_and_beam_splitter_dilation` | 0.0e+00 | 1e-11 |
| channel:loss | native-gaussian | independent analytic | `test_val_gaussian.py::test_loss_channel_against_xy_and_beam_splitter_dilation` | 0.0e+00 | 1e-11 |
| channel:loss | piquasso | independent analytic | `test_val_gaussian.py::test_loss_channel_against_xy_and_beam_splitter_dilation` | 1.7e-16 | 1e-11 |
| channel:thermal-loss | gaussian | independent analytic | `test_val_gaussian.py::test_loss_channel_against_xy_and_beam_splitter_dilation` | 0.0e+00 | 1e-11 |
| channel:thermal-loss | native-gaussian | independent analytic | `test_val_gaussian.py::test_loss_channel_against_xy_and_beam_splitter_dilation` | 0.0e+00 | 1e-11 |
| channel:thermal-loss | piquasso | independent analytic | `test_val_gaussian.py::test_loss_channel_against_xy_and_beam_splitter_dilation` | 1.7e-16 | 1e-11 |
| entropy:thermal | native_gaussian fn | independent numeric | `test_val_gaussian.py::test_thermal_entropy_and_native_diagnostics` | 0.0e+00 | 1e-12 |
| entropy:thermal | GaussianState (numpy) | independent numeric | `test_val_gaussian.py::test_thermal_entropy_and_native_diagnostics` | 0.0e+00 | 1e-12 |
| feed-forward wire:ensemble = T rho T^T + N | gaussian | independent analytic | `test_val_gaussian.py::test_wire_step_ensemble_is_analytic_teleportation_map` | 1.8e-15 | 1e-11 |
| feed-forward wire:ensemble = T rho T^T + N | native-gaussian | independent analytic | `test_val_gaussian.py::test_wire_step_ensemble_is_analytic_teleportation_map` | 1.8e-15 | 1e-11 |
| feed-forward wire:ensemble = T rho T^T + N | piquasso | independent analytic | `test_val_gaussian.py::test_wire_step_ensemble_is_analytic_teleportation_map` | 3.6e-15 | 1e-11 |
| feed-forward wire:postselected | gaussian | independent numeric | `test_val_gaussian.py::test_adaptive_wire_against_hand_teleportation` | 2.2e-16 | 1e-11 |
| feed-forward wire:postselected | native-gaussian | independent numeric | `test_val_gaussian.py::test_adaptive_wire_against_hand_teleportation` | 2.2e-16 | 1e-11 |
| feed-forward wire:postselected | piquasso | independent numeric | `test_val_gaussian.py::test_adaptive_wire_against_hand_teleportation` | 6.7e-16 | 1e-11 |
| feed-forward wire:postselected likelihood | gaussian | independent numeric | `test_val_gaussian.py::test_adaptive_wire_against_hand_teleportation` | 0.0e+00 | 1e-10 |
| feed-forward wire:postselected likelihood | native-gaussian | independent numeric | `test_val_gaussian.py::test_adaptive_wire_against_hand_teleportation` | 0.0e+00 | 1e-10 |
| feed-forward wire:postselected likelihood | piquasso | independent numeric | `test_val_gaussian.py::test_adaptive_wire_against_hand_teleportation` | 0.0e+00 | 1e-10 |
| feed-forward wire:sampled | gaussian | independent numeric | `test_val_gaussian.py::test_adaptive_wire_against_hand_teleportation` | 8.9e-16 | 1e-11 |
| feed-forward wire:sampled | native-gaussian | independent numeric | `test_val_gaussian.py::test_adaptive_wire_against_hand_teleportation` | 8.9e-16 | 1e-11 |
| feed-forward wire:sampled | piquasso | independent numeric | `test_val_gaussian.py::test_adaptive_wire_against_hand_teleportation` | 8.9e-16 | 1e-11 |
| feed-forward wire:sampled likelihood | gaussian | independent numeric | `test_val_gaussian.py::test_adaptive_wire_against_hand_teleportation` | 4.4e-16 | 1e-10 |
| feed-forward wire:sampled likelihood | native-gaussian | independent numeric | `test_val_gaussian.py::test_adaptive_wire_against_hand_teleportation` | 4.4e-16 | 1e-10 |
| feed-forward wire:sampled likelihood | piquasso | independent numeric | `test_val_gaussian.py::test_adaptive_wire_against_hand_teleportation` | 4.4e-16 | 1e-10 |
| gate:beamsplitter | gaussian | independent numeric | `test_val_gaussian.py::test_gaussian_gates_against_generator_symplectic` | 2.7e-15 | 1e-11 |
| gate:beamsplitter | native-gaussian | independent numeric | `test_val_gaussian.py::test_gaussian_gates_against_generator_symplectic` | 1.8e-15 | 1e-11 |
| gate:beamsplitter | piquasso | independent numeric | `test_val_gaussian.py::test_gaussian_gates_against_generator_symplectic` | 1.8e-15 | 1e-11 |
| gate:cz | gaussian | independent numeric | `test_val_gaussian.py::test_gaussian_gates_against_generator_symplectic` | 8.9e-16 | 1e-11 |
| gate:cz | native-gaussian | independent numeric | `test_val_gaussian.py::test_gaussian_gates_against_generator_symplectic` | 8.9e-16 | 1e-11 |
| gate:cz | piquasso | independent numeric | `test_val_gaussian.py::test_gaussian_gates_against_generator_symplectic` | 8.9e-16 | 1e-11 |
| gate:displacement | gaussian | independent analytic | `test_val_gaussian.py::test_displacement_shifts_mean_only` | 0.0e+00 | 1e-12 |
| gate:displacement | native-gaussian | independent analytic | `test_val_gaussian.py::test_displacement_shifts_mean_only` | 0.0e+00 | 1e-12 |
| gate:displacement | piquasso | independent analytic | `test_val_gaussian.py::test_displacement_shifts_mean_only` | 1.4e-15 | 1e-12 |
| gate:quadratic_phase | gaussian | independent numeric | `test_val_gaussian.py::test_gaussian_gates_against_generator_symplectic` | 0.0e+00 | 1e-11 |
| gate:quadratic_phase | native-gaussian | independent numeric | `test_val_gaussian.py::test_gaussian_gates_against_generator_symplectic` | 2.2e-16 | 1e-11 |
| gate:quadratic_phase | piquasso | independent numeric | `test_val_gaussian.py::test_gaussian_gates_against_generator_symplectic` | 1.8e-15 | 1e-11 |
| gate:rotation | gaussian | independent numeric | `test_val_gaussian.py::test_gaussian_gates_against_generator_symplectic` | 8.9e-16 | 1e-11 |
| gate:rotation | native-gaussian | independent numeric | `test_val_gaussian.py::test_gaussian_gates_against_generator_symplectic` | 1.1e-15 | 1e-11 |
| gate:rotation | piquasso | independent numeric | `test_val_gaussian.py::test_gaussian_gates_against_generator_symplectic` | 1.0e-15 | 1e-11 |
| gate:squeeze | gaussian | independent numeric | `test_val_gaussian.py::test_gaussian_gates_against_generator_symplectic` | 0.0e+00 | 1e-11 |
| gate:squeeze | native-gaussian | independent numeric | `test_val_gaussian.py::test_gaussian_gates_against_generator_symplectic` | 0.0e+00 | 1e-11 |
| gate:squeeze | piquasso | independent numeric | `test_val_gaussian.py::test_gaussian_gates_against_generator_symplectic` | 3.6e-15 | 1e-11 |
| heterodyne:conditional | gaussian | independent numeric | `test_val_gaussian.py::test_heterodyne_conditioning_and_density` | 4.3e-14 | 1e-11 |
| heterodyne:conditional | native-gaussian | independent numeric | `test_val_gaussian.py::test_heterodyne_conditioning_and_density` | 4.3e-14 | 1e-11 |
| heterodyne:conditional | piquasso | independent numeric | `test_val_gaussian.py::test_heterodyne_conditioning_and_density` | 4.3e-14 | 1e-11 |
| heterodyne:density | gaussian | independent numeric | `test_val_gaussian.py::test_heterodyne_conditioning_and_density` | 4.4e-16 | 1e-10 |
| heterodyne:density | native-gaussian | independent numeric | `test_val_gaussian.py::test_heterodyne_conditioning_and_density` | 4.4e-16 | 1e-10 |
| heterodyne:density | piquasso | independent numeric | `test_val_gaussian.py::test_heterodyne_conditioning_and_density` | 4.4e-16 | 1e-10 |
| heterodyne:husimi density | gaussian | independent numeric | `test_val_gaussian.py::test_heterodyne_density_is_husimi_function` | 1.3e-15 | 1e-10 |
| heterodyne:husimi density | native-gaussian | independent numeric | `test_val_gaussian.py::test_heterodyne_density_is_husimi_function` | 1.3e-15 | 1e-10 |
| heterodyne:husimi density | piquasso | independent numeric | `test_val_gaussian.py::test_heterodyne_density_is_husimi_function` | 1.3e-15 | 1e-10 |
| homodyne:conditional | gaussian | independent numeric | `test_val_gaussian.py::test_homodyne_conditioning_and_density` | 6.5e-13 | 1e-11 |
| homodyne:conditional | native-gaussian | independent numeric | `test_val_gaussian.py::test_homodyne_conditioning_and_density` | 6.5e-13 | 1e-11 |
| homodyne:conditional | piquasso | independent numeric | `test_val_gaussian.py::test_homodyne_conditioning_and_density` | 6.5e-13 | 1e-11 |
| homodyne:density | gaussian | independent numeric | `test_val_gaussian.py::test_homodyne_conditioning_and_density` | 0.0e+00 | 1e-10 |
| homodyne:density | native-gaussian | independent numeric | `test_val_gaussian.py::test_homodyne_conditioning_and_density` | 0.0e+00 | 1e-10 |
| homodyne:density | piquasso | independent numeric | `test_val_gaussian.py::test_homodyne_conditioning_and_density` | 0.0e+00 | 1e-10 |
| homodyne:inefficient+noise:conditional | gaussian | independent numeric | `test_val_gaussian.py::test_homodyne_conditioning_and_density` | 3.6e-14 | 1e-11 |
| homodyne:inefficient+noise:conditional | native-gaussian | independent numeric | `test_val_gaussian.py::test_homodyne_conditioning_and_density` | 3.6e-14 | 1e-11 |
| homodyne:inefficient+noise:conditional | piquasso | independent numeric | `test_val_gaussian.py::test_homodyne_conditioning_and_density` | 3.6e-14 | 1e-11 |
| homodyne:inefficient+noise:density | gaussian | independent numeric | `test_val_gaussian.py::test_homodyne_conditioning_and_density` | 1.4e-15 | 1e-10 |
| homodyne:inefficient+noise:density | native-gaussian | independent numeric | `test_val_gaussian.py::test_homodyne_conditioning_and_density` | 1.4e-15 | 1e-10 |
| homodyne:inefficient+noise:density | piquasso | independent numeric | `test_val_gaussian.py::test_homodyne_conditioning_and_density` | 1.4e-15 | 1e-10 |
| log-negativity TMSV | gaussian | independent analytic | `test_val_gaussian.py::test_tmsv_log_negativity` | 4.4e-16 | 1e-11 |
| log-negativity TMSV | native-gaussian | independent analytic | `test_val_gaussian.py::test_tmsv_log_negativity` | 4.4e-16 | 1e-11 |
| log-negativity TMSV | piquasso | independent analytic | `test_val_gaussian.py::test_tmsv_log_negativity` | 4.4e-16 | 1e-11 |
| log-negativity TMSV (lossy) | gaussian | independent analytic | `test_val_gaussian.py::test_tmsv_log_negativity` | 2.0e-15 | 1e-11 |
| log-negativity TMSV (lossy) | native-gaussian | independent analytic | `test_val_gaussian.py::test_tmsv_log_negativity` | 2.0e-15 | 1e-11 |
| log-negativity TMSV (lossy) | piquasso | independent analytic | `test_val_gaussian.py::test_tmsv_log_negativity` | 2.0e-15 | 1e-11 |
| mixed state:entropy | gaussian | independent analytic | `test_val_gaussian.py::test_mixed_state_spectral_diagnostics` | 0.0e+00 | 1e-10 |
| mixed state:entropy | native-gaussian | independent analytic | `test_val_gaussian.py::test_mixed_state_spectral_diagnostics` | 1.7e-14 | 1e-10 |
| mixed state:entropy | piquasso | independent analytic | `test_val_gaussian.py::test_mixed_state_spectral_diagnostics` | 1.5e-14 | 1e-10 |
| mixed state:purity | gaussian | independent analytic | `test_val_gaussian.py::test_mixed_state_spectral_diagnostics` | 0.0e+00 | 1e-10 |
| mixed state:purity | native-gaussian | independent analytic | `test_val_gaussian.py::test_mixed_state_spectral_diagnostics` | 1.7e-16 | 1e-10 |
| mixed state:purity | piquasso | independent analytic | `test_val_gaussian.py::test_mixed_state_spectral_diagnostics` | 5.6e-17 | 1e-10 |
| mixed state:symplectic eigenvalues | gaussian | independent analytic | `test_val_gaussian.py::test_mixed_state_spectral_diagnostics` | 8.9e-16 | 1e-10 |
| mixed state:symplectic eigenvalues | native-gaussian | independent analytic | `test_val_gaussian.py::test_mixed_state_spectral_diagnostics` | 8.9e-16 | 1e-10 |
| mixed state:symplectic eigenvalues | piquasso | independent analytic | `test_val_gaussian.py::test_mixed_state_spectral_diagnostics` | 3.6e-15 | 1e-10 |
| purity:coherent | gaussian | independent numeric | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 3.4e-15 | 1e-12 |
| purity:coherent | native-gaussian | independent numeric | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 3.4e-15 | 1e-12 |
| purity:coherent | piquasso | independent numeric | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 3.4e-15 | 1e-12 |
| purity:squeezed | gaussian | independent numeric | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 2.8e-15 | 1e-12 |
| purity:squeezed | native-gaussian | independent numeric | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 2.8e-15 | 1e-12 |
| purity:squeezed | piquasso | independent numeric | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 2.8e-15 | 1e-12 |
| purity:thermal | gaussian | independent numeric | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 5.6e-17 | 1e-12 |
| purity:thermal | native-gaussian | independent numeric | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 5.6e-17 | 1e-12 |
| purity:thermal | piquasso | independent numeric | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 5.6e-17 | 1e-12 |
| purity:thermal | native_gaussian fn | independent numeric | `test_val_gaussian.py::test_thermal_entropy_and_native_diagnostics` | 0.0e+00 | 1e-12 |
| purity:vacuum | gaussian | independent analytic | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 0.0e+00 | 1e-12 |
| purity:vacuum | native-gaussian | independent analytic | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 0.0e+00 | 1e-12 |
| purity:vacuum | piquasso | independent analytic | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 0.0e+00 | 1e-12 |
| quadrature q cos + p sin | gaussian | independent analytic | `test_val_gaussian.py::test_quadrature_convention_q_cos_plus_p_sin` | 0.0e+00 | 1e-14 |
| quadrature q cos + p sin | native-gaussian | independent analytic | `test_val_gaussian.py::test_quadrature_convention_q_cos_plus_p_sin` | 0.0e+00 | 1e-14 |
| quadrature q cos + p sin | piquasso | independent analytic | `test_val_gaussian.py::test_quadrature_convention_q_cos_plus_p_sin` | 0.0e+00 | 1e-14 |
| random 5-mode circuit (rel.) | gaussian | independent numeric | `test_val_gaussian.py::test_random_multimode_circuit_physicality` | 3.3e-16 | 1e-11 |
| random 5-mode circuit (rel.) | native-gaussian | independent numeric | `test_val_gaussian.py::test_random_multimode_circuit_physicality` | 4.9e-16 | 1e-11 |
| random 5-mode circuit (rel.) | piquasso | independent numeric | `test_val_gaussian.py::test_random_multimode_circuit_physicality` | 4.4e-15 | 1e-11 |
| state:coherent | gaussian | independent numeric | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 1.6e-15 | 1e-11 |
| state:coherent | native-gaussian | independent numeric | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 1.6e-15 | 1e-11 |
| state:coherent | piquasso | independent numeric | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 1.6e-15 | 1e-11 |
| state:resource p-squeezed | gaussian | independent analytic | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 0.0e+00 | 1e-12 |
| state:resource p-squeezed | native-gaussian | independent analytic | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 0.0e+00 | 1e-12 |
| state:resource p-squeezed | piquasso | independent analytic | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 5.6e-17 | 1e-12 |
| state:squeezed | gaussian | independent numeric | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 7.8e-16 | 1e-11 |
| state:squeezed | native-gaussian | independent numeric | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 7.8e-16 | 1e-11 |
| state:squeezed | piquasso | independent numeric | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 7.8e-16 | 1e-11 |
| state:thermal | gaussian | independent numeric | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 8.9e-16 | 1e-11 |
| state:thermal | native-gaussian | independent numeric | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 8.9e-16 | 1e-11 |
| state:thermal | piquasso | independent numeric | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 8.9e-16 | 1e-11 |
| state:vacuum | gaussian | independent analytic | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 0.0e+00 | 1e-11 |
| state:vacuum | native-gaussian | independent analytic | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 0.0e+00 | 1e-11 |
| state:vacuum | piquasso | independent analytic | `test_val_gaussian.py::test_vacuum_coherent_squeezed_thermal_states` | 0.0e+00 | 1e-11 |

### Fock

| Operation | Backend | Reference type | Test | Max error | Tolerance |
|---|---|---|---|---|---|
| channel:loss (dilation) | native-mixed-fock | independent numeric | `test_val_fock.py::test_loss_matches_beam_splitter_dilation` | 5.7e-17 | 1e-12 |
| channel:thermal loss (dilation) | native-mixed-fock | independent numeric | `test_val_fock.py::test_loss_matches_beam_splitter_dilation` | 7.5e-16 | 1e-12 |
| convergence: 1-F at cutoff 20 (D+CZ) | native-fock | independent numeric | `test_val_fock.py::test_truncated_gate_converges_to_untruncated_gate` | 2.7e-11 | 1e-09 |
| convergence: 1-F at cutoff 20 (D+CZ) | native-mixed-fock | independent numeric | `test_val_fock.py::test_truncated_gate_converges_to_untruncated_gate` | 2.7e-11 | 1e-09 |
| convergence: cubic projected 1-F at cutoff 28 | native-fock | independent numeric | `test_val_fock.py::test_cubic_phase_both_definitions` | 4.8e-07 | 1e-06 |
| convergence: cubic projected 1-F at cutoff 28 | native-mixed-fock | independent numeric | `test_val_fock.py::test_cubic_phase_both_definitions` | 4.8e-07 | 1e-06 |
| convergence: cubic truncated_generator 1-F at cutoff 28 | native-fock | independent numeric | `test_val_fock.py::test_cubic_phase_both_definitions` | 7.3e-07 | 1e-06 |
| convergence: cubic truncated_generator 1-F at cutoff 28 | native-mixed-fock | independent numeric | `test_val_fock.py::test_cubic_phase_both_definitions` | 7.3e-07 | 1e-06 |
| cubic (expm of truncated q^3) | native-fock | independent numeric | `test_val_fock.py::test_cubic_phase_both_definitions` | 4.4e-16 | 1e-11 |
| cubic (expm of truncated q^3) | native-mixed-fock | independent numeric | `test_val_fock.py::test_cubic_phase_both_definitions` | 3.3e-16 | 1e-11 |
| cubic (projected = P V P) | native-fock | independent numeric | `test_val_fock.py::test_cubic_phase_both_definitions` | 1.8e-16 | 1e-11 |
| cubic (projected = P V P) | native-mixed-fock | independent numeric | `test_val_fock.py::test_cubic_phase_both_definitions` | 2.9e-16 | 1e-11 |
| gate:beamsplitter | native-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 6.9e-17 | 1e-12 |
| gate:beamsplitter | native-mixed-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 8.3e-17 | 1e-12 |
| gate:beamsplitter | piquasso-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.4e-16 | 1e-10 |
| gate:beamsplitter | piquasso-mixed-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.4e-16 | 1e-10 |
| gate:cross_kerr | native-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 3.9e-17 | 1e-12 |
| gate:cross_kerr | native-mixed-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 9.0e-17 | 1e-12 |
| gate:displace | native-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.1e-16 | 1e-12 |
| gate:displace | native-mixed-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 8.3e-17 | 1e-12 |
| gate:displace | piquasso-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.1e-16 | 1e-10 |
| gate:displace | piquasso-mixed-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.7e-16 | 1e-10 |
| gate:entangle | native-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.7e-16 | 1e-12 |
| gate:entangle | native-mixed-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.9e-16 | 1e-12 |
| gate:entangle | piquasso-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 2.0e-05 | 1e-04 |
| gate:entangle | piquasso-mixed-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.5e-05 | 1e-04 |
| gate:kerr | native-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.8e-16 | 1e-12 |
| gate:kerr | native-mixed-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.1e-16 | 1e-12 |
| gate:kerr | piquasso-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.7e-16 | 1e-10 |
| gate:kerr | piquasso-mixed-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.1e-16 | 1e-10 |
| gate:quadratic_phase | native-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.1e-16 | 1e-12 |
| gate:quadratic_phase | native-mixed-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.9e-16 | 1e-12 |
| gate:quadratic_phase | piquasso-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 2.2e-16 | 1e-10 |
| gate:quadratic_phase | piquasso-mixed-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.7e-16 | 1e-10 |
| gate:rotate | native-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.1e-16 | 1e-12 |
| gate:rotate | native-mixed-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 9.8e-17 | 1e-12 |
| gate:rotate | piquasso-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 8.9e-17 | 1e-10 |
| gate:rotate | piquasso-mixed-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 7.1e-17 | 1e-10 |
| gate:squeeze | native-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 3.3e-16 | 1e-12 |
| gate:squeeze | native-mixed-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 3.3e-16 | 1e-12 |
| gate:squeeze | piquasso-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 3.9e-16 | 1e-10 |
| gate:squeeze | piquasso-mixed-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 2.2e-16 | 1e-10 |
| homodyne density (Fock) | native-fock | independent analytic | `test_val_fock.py::test_fock_homodyne_density_convention` | 6.7e-16 | 1e-09 |
| homodyne density (Fock) | native-mixed-fock | independent analytic | `test_val_fock.py::test_fock_homodyne_density_convention` | 5.6e-16 | 1e-09 |
| homodyne density (Fock) | piquasso-fock (secondary) | independent analytic | `test_val_fock.py::test_fock_homodyne_density_convention` | 6.7e-16 | 1e-09 |
| homodyne density (Fock) | piquasso-mixed-fock (secondary) | independent analytic | `test_val_fock.py::test_fock_homodyne_density_convention` | 5.6e-16 | 1e-09 |
| partial trace (incl. reordering) | native-fock | independent numeric | `test_val_fock.py::test_partial_trace_by_explicit_contraction` | 0.0e+00 | 1e-14 |
| partial trace (incl. reordering) | native-mixed-fock | independent numeric | `test_val_fock.py::test_partial_trace_by_explicit_contraction` | 2.9e-17 | 1e-14 |
| photon counting: conditional state | native-fock | independent numeric | `test_val_fock.py::test_photon_counting_probability_and_conditional_state` | 1.2e-16 | 1e-13 |
| photon counting: conditional state | native-mixed-fock | independent numeric | `test_val_fock.py::test_photon_counting_probability_and_conditional_state` | 1.1e-16 | 1e-13 |
| photon counting: probability | native-fock | independent numeric | `test_val_fock.py::test_photon_counting_probability_and_conditional_state` | 2.2e-16 | 1e-14 |
| photon counting: probability | native-mixed-fock | independent numeric | `test_val_fock.py::test_photon_counting_probability_and_conditional_state` | 5.6e-17 | 1e-14 |
| physicality after 21-gate sequence | native-fock | defining property | `test_val_fock.py::test_long_sequence_physicality` | 1.3e-15 | 1e-12 |
| physicality after 21-gate sequence | native-mixed-fock | defining property | `test_val_fock.py::test_long_sequence_physicality` | 2.2e-16 | 1e-12 |
| physicality:trace/Hermitian/PSD | native-fock | defining property | `test_val_fock.py::test_gate_against_enlarged_expm` | 6.7e-16 | 1e-13 |
| physicality:trace/Hermitian/PSD | native-mixed-fock | defining property | `test_val_fock.py::test_gate_against_enlarged_expm` | 3.0e-16 | 1e-13 |
| physicality:trace/Hermitian/PSD | piquasso-fock (secondary) | defining property | `test_val_fock.py::test_gate_against_enlarged_expm` | 5.5e-16 | 1e-13 |
| physicality:trace/Hermitian/PSD | piquasso-mixed-fock (secondary) | defining property | `test_val_fock.py::test_gate_against_enlarged_expm` | 5.9e-16 | 1e-13 |
| quadrature q cos + p sin (Fock) | native-fock | independent analytic | `test_val_fock.py::test_fock_homodyne_density_convention` | 1.1e-15 | 1e-12 |
| quadrature q cos + p sin (Fock) | native-mixed-fock | independent analytic | `test_val_fock.py::test_fock_homodyne_density_convention` | 1.3e-15 | 1e-12 |
| quadrature q cos + p sin (Fock) | piquasso-fock (secondary) | independent analytic | `test_val_fock.py::test_fock_homodyne_density_convention` | 2.0e-15 | 1e-12 |
| quadrature q cos + p sin (Fock) | piquasso-mixed-fock (secondary) | independent analytic | `test_val_fock.py::test_fock_homodyne_density_convention` | 1.3e-15 | 1e-12 |
| reference convergence D=26 vs 32:beamsplitter | (reference) | reference self-check | `test_val_fock.py::test_gate_against_enlarged_expm` | 0.0e+00 | 1e-14 |
| reference convergence D=26 vs 32:cross_kerr | (reference) | reference self-check | `test_val_fock.py::test_gate_against_enlarged_expm` | 0.0e+00 | 1e-14 |
| reference convergence D=26 vs 32:entangle | (reference) | reference self-check | `test_val_fock.py::test_gate_against_enlarged_expm` | 3.6e-16 | 1e-14 |
| reference convergence D=40 vs 56:displace | (reference) | reference self-check | `test_val_fock.py::test_gate_against_enlarged_expm` | 3.1e-19 | 1e-14 |
| reference convergence D=40 vs 56:kerr | (reference) | reference self-check | `test_val_fock.py::test_gate_against_enlarged_expm` | 0.0e+00 | 1e-14 |
| reference convergence D=40 vs 56:quadratic_phase | (reference) | reference self-check | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.1e-18 | 1e-14 |
| reference convergence D=40 vs 56:rotate | (reference) | reference self-check | `test_val_fock.py::test_gate_against_enlarged_expm` | 0.0e+00 | 1e-14 |
| reference convergence D=40 vs 56:squeeze | (reference) | reference self-check | `test_val_fock.py::test_gate_against_enlarged_expm` | 2.0e-17 | 1e-14 |
| retained:beamsplitter | native-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 4.4e-16 | 1e-12 |
| retained:beamsplitter | native-mixed-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 4.4e-16 | 1e-12 |
| retained:beamsplitter | piquasso-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 8.9e-16 | 1e-10 |
| retained:beamsplitter | piquasso-mixed-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 6.7e-16 | 1e-10 |
| retained:channel:loss (dilation) | native-mixed-fock | independent numeric | `test_val_fock.py::test_loss_matches_beam_splitter_dilation` | 2.2e-16 | 1e-12 |
| retained:channel:thermal loss (dilation) | native-mixed-fock | independent numeric | `test_val_fock.py::test_loss_matches_beam_splitter_dilation` | 2.9e-15 | 1e-12 |
| retained:coherent | native-fock | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 2.2e-16 | 1e-13 |
| retained:coherent | native-mixed-fock | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 3.3e-16 | 1e-13 |
| retained:coherent | piquasso-fock (secondary) | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 1.1e-16 | 1e-10 |
| retained:coherent | piquasso-mixed-fock (secondary) | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 3.3e-16 | 1e-10 |
| retained:coherent = Q(c, |a|^2) | native-fock | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 1.1e-16 | 1e-13 |
| retained:coherent = Q(c, |a|^2) | native-mixed-fock | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 2.2e-16 | 1e-13 |
| retained:coherent = Q(c, |a|^2) | piquasso-fock (secondary) | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 2.2e-16 | 1e-10 |
| retained:coherent = Q(c, |a|^2) | piquasso-mixed-fock (secondary) | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 2.2e-16 | 1e-10 |
| retained:cross_kerr | native-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 4.4e-16 | 1e-12 |
| retained:cross_kerr | native-mixed-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.1e-16 | 1e-12 |
| retained:cubic (expm of truncated q^3) | native-fock | independent numeric | `test_val_fock.py::test_cubic_phase_both_definitions` | 1.1e-15 | 1e-11 |
| retained:cubic (expm of truncated q^3) | native-mixed-fock | independent numeric | `test_val_fock.py::test_cubic_phase_both_definitions` | 6.7e-16 | 1e-11 |
| retained:cubic (projected = P V P) | native-fock | independent numeric | `test_val_fock.py::test_cubic_phase_both_definitions` | 4.6e-13 | 1e-11 |
| retained:cubic (projected = P V P) | native-mixed-fock | independent numeric | `test_val_fock.py::test_cubic_phase_both_definitions` | 4.6e-13 | 1e-11 |
| retained:displace | native-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 0.0e+00 | 1e-12 |
| retained:displace | native-mixed-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 0.0e+00 | 1e-12 |
| retained:displace | piquasso-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.1e-16 | 1e-10 |
| retained:displace | piquasso-mixed-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.1e-16 | 1e-10 |
| retained:displaced squeezed | native-fock | independent numeric | `test_val_fock.py::test_displaced_squeezed_and_mixed_gaussian_inputs` | 2.8e-15 | 1e-12 |
| retained:entangle | native-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 8.9e-16 | 1e-12 |
| retained:entangle | native-mixed-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 5.6e-16 | 1e-12 |
| retained:entangle | piquasso-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.3e-07 | 1e-04 |
| retained:entangle | piquasso-mixed-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.3e-07 | 1e-04 |
| retained:kerr | native-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 0.0e+00 | 1e-12 |
| retained:kerr | native-mixed-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 0.0e+00 | 1e-12 |
| retained:kerr | piquasso-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 2.2e-16 | 1e-10 |
| retained:kerr | piquasso-mixed-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 0.0e+00 | 1e-10 |
| retained:number | native-fock | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 0.0e+00 | 1e-13 |
| retained:number | native-mixed-fock | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 0.0e+00 | 1e-13 |
| retained:number | piquasso-fock (secondary) | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 0.0e+00 | 1e-10 |
| retained:number | piquasso-mixed-fock (secondary) | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 0.0e+00 | 1e-10 |
| retained:quadratic_phase | native-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 0.0e+00 | 1e-12 |
| retained:quadratic_phase | native-mixed-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 2.2e-16 | 1e-12 |
| retained:quadratic_phase | piquasso-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 2.2e-16 | 1e-10 |
| retained:quadratic_phase | piquasso-mixed-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.1e-16 | 1e-10 |
| retained:resource p-squeezed | native-fock | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 1.1e-16 | 1e-13 |
| retained:resource p-squeezed | native-mixed-fock | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 1.1e-16 | 1e-13 |
| retained:resource p-squeezed | piquasso-fock (secondary) | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 1.1e-16 | 1e-10 |
| retained:resource p-squeezed | piquasso-mixed-fock (secondary) | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 1.1e-16 | 1e-10 |
| retained:rotate | native-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 2.2e-16 | 1e-12 |
| retained:rotate | native-mixed-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 0.0e+00 | 1e-12 |
| retained:rotate | piquasso-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 2.2e-16 | 1e-10 |
| retained:rotate | piquasso-mixed-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 2.2e-16 | 1e-10 |
| retained:squeeze | native-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 2.2e-16 | 1e-12 |
| retained:squeeze | native-mixed-fock | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 5.6e-16 | 1e-12 |
| retained:squeeze | piquasso-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 1.1e-16 | 1e-10 |
| retained:squeeze | piquasso-mixed-fock (secondary) | independent numeric | `test_val_fock.py::test_gate_against_enlarged_expm` | 2.2e-16 | 1e-10 |
| retained:squeezed | native-fock | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 2.2e-16 | 1e-13 |
| retained:squeezed | native-mixed-fock | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 2.2e-16 | 1e-13 |
| retained:squeezed | piquasso-fock (secondary) | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 1.1e-16 | 1e-10 |
| retained:squeezed | piquasso-mixed-fock (secondary) | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 0.0e+00 | 1e-10 |
| retained:thermal displaced squeezed | native-mixed-fock | independent numeric | `test_val_fock.py::test_displaced_squeezed_and_mixed_gaussian_inputs` | 1.4e-15 | 1e-12 |
| state:coherent | native-fock | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 6.2e-17 | 1e-13 |
| state:coherent | native-mixed-fock | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 5.7e-17 | 1e-13 |
| state:coherent | piquasso-fock (secondary) | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 7.9e-17 | 1e-10 |
| state:coherent | piquasso-mixed-fock (secondary) | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 1.1e-16 | 1e-10 |
| state:displaced squeezed | native-fock | independent numeric | `test_val_fock.py::test_displaced_squeezed_and_mixed_gaussian_inputs` | 1.8e-16 | 1e-12 |
| state:number | native-fock | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 0.0e+00 | 1e-13 |
| state:number | native-mixed-fock | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 0.0e+00 | 1e-13 |
| state:number | piquasso-fock (secondary) | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 0.0e+00 | 1e-10 |
| state:number | piquasso-mixed-fock (secondary) | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 0.0e+00 | 1e-10 |
| state:resource p-squeezed | native-fock | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 2.2e-16 | 1e-13 |
| state:resource p-squeezed | native-mixed-fock | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 4.4e-16 | 1e-13 |
| state:resource p-squeezed | piquasso-fock (secondary) | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 2.2e-16 | 1e-10 |
| state:resource p-squeezed | piquasso-mixed-fock (secondary) | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 2.2e-16 | 1e-10 |
| state:squeezed | native-fock | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 2.2e-16 | 1e-13 |
| state:squeezed | native-mixed-fock | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 3.3e-16 | 1e-13 |
| state:squeezed | piquasso-fock (secondary) | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 2.0e-16 | 1e-10 |
| state:squeezed | piquasso-mixed-fock (secondary) | independent analytic | `test_val_fock.py::test_number_coherent_squeezed_states_analytic` | 1.2e-16 | 1e-10 |
| state:thermal displaced squeezed | native-mixed-fock | independent numeric | `test_val_fock.py::test_displaced_squeezed_and_mixed_gaussian_inputs` | 2.2e-16 | 1e-12 |
| unitarity: <phi1|phi2> = <psi1|psi2> | native-fock | defining property | `test_val_fock.py::test_number_conserving_gates_preserve_inner_products` | 1.1e-16 | 1e-13 |

## 5. Findings

**F1 (API inconsistency, all backends): `backend.measure` ignores `Homodyne.angle`.**
`GaussianBackend.measure`, `NativeGaussianBackend.measure`, `PiquassoFockBackend.measure`,
`MixedFockBackend.measure` and the native Fock subclasses take the quadrature angle from a
separate `angle=0.0` argument, not from the `Homodyne` object they receive. `simulate` passes
`measurement.angle` explicitly (`simulator.py:264`), so patterns behave correctly. A direct
backend call such as `engine.measure(node, pg.Homodyne(np.pi / 2))` silently measures q
instead. Reproducer:

```python
import numpy as np, photographiq as pg
from photographiq.backends import GaussianBackend
e = GaussianBackend(); e.prepare(0, state=pg.GaussianInput.coherent(1j))  # <q>=0, <p>=2
e.measure(0, pg.Homodyne(np.pi / 2), outcome=2.0)
print(e.last_measurement["value"])  # 0.0540 = N(2; 0, 1): q was measured
# with angle passed explicitly: 0.3989 = N(2; 2, 1). Same on NativeFockBackend(30).
```

Suggested fix: default `angle=None` and fall back to `measurement.angle`, or raise when the
two disagree.

**F2 (comment only): `backends/mixed_fock.py:301`** says "Electronics noise is a classical
convolution in detected units". The code convolves in calibrated, incident-quadrature
outcome units with variance `noise`, which matches `GaussianBackend` (added variance
`(1-eta)/eta + noise`). This audit measured agreement with the closed form to 3e-16
(`test_fock_homodyne_density_convention`, eta = 0.6, noise = 0.2). The behaviour is right and
the comment is misleading.

**F3 (documented, not a defect):** `piquasso-fock` and `piquasso-mixed-fock` CZ (built as a
`GaussianTransform`, an Euler decomposition truncated after each factor) deviates from the
exact P U P by 2.0e-5 at cutoff 10, while the native engines reach 2e-16. This matches the
`native_fock.py` module docstring.

No physics or convention defect was found in `states.py`, `gaussian.py` or `backends/*.py`.

## 6. Regenerating this table

Each test module records `MEASURED[(operation, backend)] = (max error, tolerance, test)`
through `_record`, and the assertion is made in the same call. The table above was produced
by a throw-away pytest plugin, which is not part of the repository. At session finish the
plugin dumps `MEASURED` from both modules, and a short script formats the dump into the
table.
