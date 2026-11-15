# Phase 2: Independent scientific and numerical validation

Audit date 2026-11-12, local working tree of branch `upgraded`. Each section links the
detailed evidence document. "Independent" means that the reference side of a test is computed
from first principles (closed forms, `scipy.linalg.expm`, mpmath, brute-force definitions)
without calling the PhotoGraphiQ function under test.

## Conventions (verified against the source, not the docs)

| Quantity | Convention | Verified in |
| --- | --- | --- |
| Units | hbar = 2, [q, p] = 2i, q = a + a†, p = -i(a - a†) | `states.py`, `gaussian.py`, every backend |
| Ordering | interleaved (q0, p0, q1, p1, ...) | same |
| Vacuum | covariance I; coherent mean (2 Re α, 2 Im α) | same |
| Gates | CZ p_u += g q_v; homodyne records q cos θ + p sin θ | same |
| Detection | inefficient homodyne adds (1-η)/η + noise in calibrated units | Gaussian and mixed-Fock backends agree to 3e-16 |
| Fock truncation | total photon number Σ n < cutoff | native and Piquasso Fock |

All conventions agree across the NumPy, native C++ and Piquasso backends; no inconsistent
module was found. One misleading comment was corrected: the mixed-Fock noise convolution is in
calibrated, not detected, units.

## Validation results

### Native Gaussian engine

Evidence: [independent validation matrix](independent-validation-matrix.md),
`tests/validation/test_val_gaussian.py`.

Tested on `gaussian`, `native-gaussian` and `piquasso`:
- vacuum, coherent, squeezed and thermal states (moments, purity, entropy);
- symplectic gates, with S built as `expm(2ΩM)` from each generator and S Ω Sᵀ = Ω checked;
- loss and thermal loss against analytic X, Y and a beam-splitter dilation;
- homodyne (including efficiency and noise) and heterodyne conditioning against a physical dilation conditioned through the precision matrix;
- heterodyne density against the Husimi function;
- adaptive feed-forward against the analytic T(k) map;
- physicality V + iΩ ≥ 0 of random 5-mode circuits;
- log-negativity of the two-mode squeezed vacuum.

Maximum error 6.5e-13 (tolerance 1e-11). Caveat: the `piquasso` Gaussian backend shares the NumPy code for input preparation, measurement and diagnostics, so those rows re-test NumPy.

### Native Fock engine

Evidence: `tests/validation/test_val_fock.py`.

Tested on `native-fock` and `native-mixed-fock`:
- number, coherent and squeezed states against mpmath amplitudes;
- retained probability against the regularised incomplete gamma function;
- all eight gates (displacement, rotation, squeezing, beam splitter, CZ, Kerr, cross-Kerr, cubic phase) against `expm` on an enlarged per-mode space (40/56 levels for one mode, 26/32 for two modes; the reference changes by less than 4e-16 between sizes), with the truncation handled exactly as P U ψ / ‖P U ψ‖;
- photon counting and conditional states, partial traces, homodyne density, loss as a dilation;
- number-conserving gates exactly unitary;
- norm, Hermiticity and positivity after a 21-gate sequence.

Maximum error 5e-16 for gates (tolerance 1e-12). Piquasso, as a secondary check, agrees to about 1e-16, except CZ at 2.0e-5, which is its documented decomposition truncation.

### Combined Gaussian unitary (|α| = 4, r = 0.3, cutoff 80)

Evidence: [native-fock-gaussian-unitary](native-fock-gaussian-unitary.md),
`tests/native/test_native_fock_gaussian_unitary_stability.py`.

- **Reproduced:** 8.5e-5 at the reported case, and 1.4e2 at |α| = 6, cutoff 100. Squeezing alone failed at large size (7.2e-5 at size 400).
- **Root cause:** the Bargmann column recurrence amplifies rounding by about √(N+1)/√(n+1) per step. It is not an ordering or normalisation error; the same recurrence at 60 digits is exact.
- **Fix:** one-mode U = R S_r R D(β). The squeeze factor uses a Jacobi-polynomial closed form with a stable recurrence, and the internal size comes from a rigorous displacement-tail bound.
- **After the fix:** at most 1.9e-14 against 60-digit mpmath over |α| ≤ 6, |r| ≤ 1, cutoff ≤ 100 (squeezing alone up to size 800). The convention is unchanged (5.4e-14 against the old code where the old code was accurate).
- **Remaining:** displaced multi-mode Gaussian unitaries still use the Bargmann path, without a general bound. The backends never request them; a direct `gaussian_unitary` call can.

### Boson sampling

Evidence: `tests/sampling/test_sampling_boson.py`, `tests/sampling/test_sampling_linalg.py`, [sampling guide](../user-guide/sampling.md).

Validated against:
- permanents by the defining sum;
- repeated rows and columns through occupation multiplicities;
- normalisation of complete distributions;
- collision and collision-free events;
- photon-number conservation;
- exact sampling, by total-variation distance against the exact distribution.

Both real and complex interferometers are tested, and Piquasso is used as a cross-implementation check.

### Gaussian boson sampling

Evidence: [gbs-threshold-accuracy](gbs-threshold-accuracy.md),
`benchmarks/gaussian_boson_sampling/results/threshold_reference_matrix_20261111-233720.json`.

- **Matrix functions:** (loop) hafnians against the perfect-matching definitions; the PNR distribution against Piquasso, agreeing to at most 5.7e-12 relative.
- **Threshold matrix:** 382 cases covering modes 4–24, clicks 1–16, r 0.05–1.5, transmissivity 0.5–1, with and without displacement, including zero-probability patterns. The reference is an independent 40-digit mpmath implementation.
- **Bound:** it held in every case, on both backends. Its looseness (median ratio to the actual error) is 620 native and 1100 Python, ranging from 5 to 1e7.
- **Warnings:** none missed. Every case above 1e-6 warned; 37 native and 45 Python false alarms.
- **Where double precision fails** (relative error above 1e-6), in 80 of 360 cases, all at r ≤ 0.5 with many clicks. Piquasso fails in the same regime (82 of 360 cases) and returns 0.0 for 14 nonzero probabilities without a warning.
- **Adaptive precision:** `precision="auto"` recomputes in mpmath when needed. On the matrix it gives at most 3.1e-8 on the double path and 7.6e-10 on the mpmath path.

### Graph algorithms

Evidence: [cv-flow-robustness](cv-flow-robustness.md), `tests/flow`, `tests/graphix_reference`.

- **Qubit flows:** causal flow, gflow and Pauli flow are checked against Graphix 0.4 on 150 random graphs per notion, positive and negative (causal 31/119, gflow 27/123, Pauli 42/108).
- **CV flow:** the analysis now distinguishes three outcomes:
  - **impossible**, with exact rational witnesses or a combinatorial obstruction;
  - **numerically uncertain**, from a rank decision within the threshold margin;
  - **certified**, with the regime stated: ideal limit, finite-squeezing channel, or both.
- **Defaults:** on 600 adversarial graphs, 150 certified, 408 impossible and 42 unknown. `exact=True` resolves all 42: 27 certified for the ideal limit only and 14 in both regimes. The one dynamic-range case still unknown in that run is resolved by a rule added afterwards, now covered by a regression test; the 600-graph run was not repeated after that change.
- **High squeezing:** large-squeezing wires (r up to 4.5) are certified with `shift=1.5`.

### Calculus

Evidence: `tests/calculus`.

- **Qubit measurement calculus:** standardisation, signal shifting and `pattern_from_flow` (including Pauli flow), checked by exact branch simulation.
- **Local Clifford:** the 24-element group, local complementation and LC equivalence.
- **ZX:** spider fusion, identity removal, colour change, copy, Hopf and 2×2 bialgebra. Every rewrite is checked scalar-exact by tensor evaluation on random diagrams; invalid preconditions raise `RewriteError`.
- **Gaussian graphical calculus** (new; Menicucci, Flammia and van Loock, PRA 83, 042335): pure-state Z ↔ V round trips; unitary updates via the linear-fractional rule; q- and p-measurement rules with tracked means. 213 tests against direct Gaussian numerics, maximum error 2e-14. See [the user guide](../user-guide/gaussian-graphical-calculus.md).

### PhotoGraphiQML

Evidence: [photographiqml-mentpy-validation](photographiqml-mentpy-validation.md).

- **MentPy:** 0.1.0a15, from the CI's exact commit `63c3d83e`, installed in an isolated Windows venv. 34/34 reference tests and 154/154 `tests/ml` pass.
- **Experiments:** all 24 scratch-copy experiments pass, including all MentPy-dependent ones (R7–R13, R20, R21). Every non-timing value is bit-identical to the recorded JSON.
- **native-fock vs piquasso-fock** in physical MuTA: probability differences of at most 1.5e-4 at cutoff 24 and 9e-5 at cutoff 32 (truncation scale), with identical decoded bits.

### Measurement API defect found and fixed

A direct `backend.measure(node, Homodyne(θ))` call ignored θ and measured q; `simulate` was not affected. All backends now use the measurement's own angle when none is passed, and raise on a contradiction (`tests/regression/test_measure_angle.py`).

## Classification of evidence

| Class | Examples |
| --- | --- |
| Independently validated | Gaussian states, gates, channels and conditioning; Fock gates (expm), states (mpmath) and measurements; one-mode Gaussian unitaries (mpmath); permanents and hafnians (definitions); threshold probabilities (40/50-digit mpmath); Gaussian graphical calculus; ZX rewrites (tensor evaluation); qubit measurement calculus (branch simulation) |
| Cross-implementation validated | native vs NumPy Gaussian trajectories; native vs Piquasso Fock; qubit flows vs Graphix; PhotoGraphiQML vs MentPy; GPU vs CPU kernels; CUDA-Q generators vs native-fock (NumPy reference only) |
| Internally regression-tested only | registry and planner, `OpenGraph` round trips, interactive layouts, serialization round trips |
| Experimental | Fock-backed features as labelled by the project; threshold-bound calibration factor for few clicks; CUDA-Q execution path (unverified); layout quality |
| Unsupported | heterodyne/general-dyne on Fock backends, CV local complementation, general CV-ZX, arbitrary-angle physical MuTA, pattern execution on CUDA-Q |

## Remaining numerical risks

- **Threshold probabilities** at small squeezing and many clicks need `precision=` or `"auto"`. The double-precision bound is conservative, with a median looseness of 620.
- **CV determinism** for r + 2·shift ≳ 9.5 or extreme weights is `unknown` by default; the exact ideal-limit check is opt-in.
- **Multi-mode displaced Gaussian unitaries** have no stability bound.
- **Truncation:** convergence must still be demonstrated per workload. Use `cutoff_convergence` or `measurement_convergence`; both now accept native backends.
