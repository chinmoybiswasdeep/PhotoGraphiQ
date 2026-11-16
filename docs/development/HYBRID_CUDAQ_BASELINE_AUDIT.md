# Hybrid CUDA-Q CV-MBQC: baseline audit

This audit records the state of the repository before the hybrid Gaussian–Fock work began.

- **Branch:** `upgraded`.
- **HEAD:** `9dea4abf59dae4680fe0d30f6b37d367e1098f1c` ("original CV-MBQC and CV-ZX").
- **Working tree at the start:** clean (`git status --short` empty).

All changes made for this task are local, unstaged and uncommitted.

## 1. What existed

| Component | Location | Status at baseline |
|---|---|---|
| CUDA-Q Dynamics adapter | `src/photographiq/cudaq_dynamics.py` | Hamiltonian-level only. Provides generators for rotation, displacement, squeezing, beam splitter, CZ, quadratic phase, Kerr, cross-Kerr and cubic phase. Also provides `BosonicHamiltonian`, `to_matrix`, the per-mode ↔ total-photon mapping, `reference_evolve` (NumPy/SciPy, unitary and Lindblad), `to_cudaq`, `evolve` (no fallback), `cudaq_probe` and `cudaq_capabilities`. |
| Compatibility shim | `src/photographiq/backends/cudaq_dynamics.py` | Re-exports the probe functions. CUDA-Q is **not** a registered pattern backend. |
| Adapter tests | `tests/cudaq_adapter/test_cudaq_adapter.py` | 28 passed, 1 skipped (`test_real_cudaq_evolve_matches_reference`, which needs `cudaq`). Covers generators vs native-fock, Lindblad loss vs native-mixed-fock, the truncation mapping, and wiring through a fake `cudaq` module. |
| Pattern backends | `gaussian`, `native-gaussian`, `native-fock`, `native-mixed-fock`, `piquasso`, `piquasso-fock`, `piquasso-mixed-fock` | `pg.simulate(pattern, backend=..., measurement_outcomes=...)` returns conditional trajectories. The native Fock engines use a **total-photon** cutoff. |
| Pattern IR | `commands.py`, `expressions.py`, `pattern.py` | `Prepare` (squeezing, `GaussianInput`, `FockInput`, `CubicPhaseResource`, `CatResource`, GKP), Gaussian and non-Gaussian gates, `Loss`, `Measure` (`Homodyne`, `PhotonNumber`, ...), `Displace` with outcome expressions, `Output`. `Pattern.validate()` rejects dependence on a future outcome. |
| Non-Gaussian builders | `non_gaussian.py` | `photon_subtraction`, `resource_injection`, `cat_injection`, `cubic_injection` (quadratic and nonlinear feed-forward). |

## 2. Gaps for hybrid CV-MBQC

1. **No pattern execution on CUDA-Q's representation.** The adapter evolves Hamiltonians only. It has no homodyne or photon-counting instrument, no feed-forward and no pattern planner. `docs/cudaq.md` already said so explicitly.
2. **No Gaussian→Fock bridge on per-mode dimensions.** Gaussian states reached Fock form only inside the native or Piquasso engines, on a total-photon basis.
3. **No explicit representation contract** for per-mode truncated states, such as mode order, retained weight or how the state was produced.
4. **No real CUDA-Q execution.**
   - The development machine is Windows 11 with an NVIDIA RTX 4050 Laptop GPU (driver 581.86).
   - PyPI lists `cudaq` 0.9.0–0.17.0, all Linux-only wheels.
   - WSL is not installed, and installing it is a system-wide administrative change.
   - The adapter's real-CUDA-Q test has therefore never run.

## 3. Decisions taken from the audit

- The new code is an **experimental API** (`photographiq.hybrid`). It is not a registered backend: the registry, `simulate()` and the public `__all__` are unchanged.
- Gates use **truncated-generator semantics**, U = exp(−i H_d) with H_d the generator truncated to the per-mode space. This is the semantics CUDA-Q itself applies, so the NumPy reference and the CUDA-Q engine compute the same mathematical object.
- `engine="cudaq"` calls `cudaq_dynamics.evolve` and raises `ImportError` without CUDA-Q. It never substitutes NumPy.
- The validation references are the existing engines: the Gaussian backend (exact for Gaussian patterns), native-fock, native-mixed-fock and Piquasso. They are compared on the same `Pattern`, using the existing truncation mapping helpers.
- Nothing in PhotoGraphiQML (MuTA, MuTAKernel, MuTAClassifier, MuTARegressor, PhysicalMuTA, GKPBridge) is modified.
