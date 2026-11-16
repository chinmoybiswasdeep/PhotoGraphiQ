# Validation

Every result below was produced by the **NumPy reference engine on CPU**. No CUDA-Q code
has executed (see [CUDA-Q real runtime validation](../validation/CUDAQ_REAL_RUNTIME_VALIDATION.md)).

**References used:**

| Reference | Role |
|---|---|
| Gaussian backend | exact conditional states for Gaussian patterns |
| native-fock / native-mixed-fock | total-photon truncation; a different truncation from the hybrid, so agreement means agreement in the converged limit |
| piquasso-fock | an independent engine |
| closed forms | analytic results where they exist |

Every comparison is made at **every measurement boundary**, using the postselected
outcome density, and on the final conditional state.

## Tests (`tests/hybrid/`, 27 tests: 26 pass, 1 skipped without CUDA-Q)

`test_ml_forward.py` checks the optional PhotoGraphiQML hook
(`photographiqml.cv.hybrid.hybrid_moments`) against `NonGaussianProgram.fock_moments` on
native-fock: means agree to 1e-6 and variances to 1e-5. `test_hybrid.py` covers the
following.

The tests cover:

- Williamson reconstruction;
- the bridge (mixed single-mode and pure two-mode moments);
- the representation contract, with no silent renormalisation;
- the total-photon round trip;
- the approximate label on Fock→Gaussian conversion;
- quadrature closed forms;
- the number-state phase invariance and squeezed variance;
- the conditional state vs the Gaussian backend (pure and mixed);
- the binned instrument;
- sampling reproducibility and replay;
- the ensemble-statistics label;
- the loss superoperator vs the exact Kraus channel and vs the Lindblad reference;
- lossy Kerr teleportation vs native-mixed-fock;
- three-mode cubic CV-MBQC vs native-fock;
- photon counting vs native-fock;
- planner dry run and memory;
- unsupported commands refused;
- causality;
- feed-forward gain;
- explicit engine choice;
- no NumPy fallback for `engine="cudaq"`;
- the real-CUDA-Q integration test (skipped here).

## Experiments (`paper_experiments/23_hybrid_cudaq`)

| ID | Result |
|---|---|
| HYB-01 | **BLOCKED**: CUDA-Q not importable (Windows, no WSL) |
| HYB-02 | Bridge: moment errors fall geometrically with d. At the largest d the max covariance error is ≤ 2.3e-5, the lost weight ≤ 1.6e-7, and every mixed output is positive semidefinite. |
| HYB-03 | Homodyne: closed forms to < 1e-9. Gaussian conditional states (7 outcomes, pure and lossy) agree in moments to ≤ 2.3e-5. Native-fock non-Gaussian densities agree to ≤ 9e-8, with state infidelity ≤ 6e-9. |
| HYB-04 | Teleportation: the fixed outcome at d = 28 matches the Gaussian backend to 2e-7 in the mean and 3.4e-6 in the covariance. For 200 sampled shots, outcomes pass KS against the exact marginal, and every per-shot conditional state at d = 32 matches to ≤ 1.5e-4. **Recorded failure:** at d = 20 the per-shot criterion fails (max 1.1e-2, at a 3.7σ outcome); see [limitations](limitations.md#truncation-diagnostics). |
| HYB-05 | **Primary**: three-mode cubic-phase CV-MBQC (all criteria pass; details below). |
| HYB-06 | Lossy teleportation (η = 0.9 / 0.7 / 0.5 pure, and η = 0.8 with n = 0.1 thermal) vs Gaussian at d = 24: mean ≤ 7.6e-7, cov ≤ 1.2e-5, density ≤ 7.4e-8. |
| HYB-07 | Kerr-enhanced MBQC step (κ = 0.05 / 0.1 / 0.2; η = 1 and 0.85): trace distance to native(-mixed)-fock ≤ 2.7e-4, which is below 3× the native engine's own cutoff-to-cutoff change. The Kerr effect size is 0.10–0.25, at least 740× the numerical error; the relative density error is ≤ 6.7e-6. |
| HYB-08 | Photon counting after a subtraction tap on a bridged squeezed vacuum: \|ΔP\| < 1e-8 and P(k)·(1 − F) < 1e-10 at d = 28; the count distribution sums to 1. **Recorded deviation:** the first criterion (infidelity < 1e-8 at d = 20) failed for rare counts. |
| HYB-09 | Binned → ideal homodyne as O(w²) (fitted slopes 1.99 and 1.99). |
| HYB-10 | 5000 samples of a cubic-phase state: KS 0.010 < 0.023; tail mass < 1e-10; reproducible. |
| HYB-11 | **BLOCKED**: GPU performance requires HYB-01. |
| HYB-12 | CPU scaling: the planner's memory estimate was always below the measured peak allocation; the hybrid is slower than native on CPU (see [performance](performance.md)). |

## Primary acceptance experiment (HYB-05)

**Pattern:** a coherent input, a p-squeezed wire mode, and a `CubicPhaseResource` (the
bridge followed by the cubic Hamiltonian). It runs CZ, a homodyne at θ = 1.1 with
correction, inverse-SUM coupling, a q homodyne, and QP(−2γm) and Z(−γm²) feed-forward.
Three modes are live together, with two dependent measurements.

**Sweeps:** hybrid d = 10…28 (40 for γ = 0.3) against native-fock cutoffs up to 48 (72 for
γ = 0.3). Piquasso-fock is recorded where it passes its norm guard.

| Case | Native floor (1 − F) | Hybrid 1 − F at largest d | Joint density relative error |
|---|---|---|---|
| γ = 0.15, m = (0.2, −0.3) | 9.0e-8 | 6.3e-7 (d = 28) | 1.1e-5 |
| γ = 0.15, m = (−0.6, 0.5) | 7.0e-8 | 3.7e-7 (d = 28) | 9.7e-6 |
| γ = 0.15, m = (1.0, 0.0) | 1.2e-7 | 3.1e-7 (d = 28) | 1.8e-6 |
| γ = 0.3, m = (0.2, −0.3) | 4.2e-6 | 1.7e-5 (d = 40) | 3.8e-5 |

**Acceptance criteria**, declared before the run:

- 1 − F ≤ max(1e-5, 10 × native floor);
- relative density error ≤ 1e-4;
- convergence from d = 10.

All criteria pass. Two deviations are recorded in the script:

- the native cutoff sweep was extended after the native norm guard rejected low cutoffs;
- for γ = 0.3, both sweeps were extended after the 48 / 28 run failed the density criterion while the native reference itself was unconverged.
