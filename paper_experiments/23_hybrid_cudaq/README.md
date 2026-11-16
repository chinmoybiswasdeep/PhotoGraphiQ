# HYB-01–HYB-12: hybrid Gaussian–Fock CV-MBQC and the CUDA-Q Dynamics adapter

These studies validate `photographiq.hybrid`, the experimental executor documented in
[docs/hybrid-cudaq](../../docs/hybrid-cudaq/architecture.md).

Each script writes the following to `../results` and `../figures`, using the R61–R68
conventions (`hyb_support.save`):

- CSV and JSON rows;
- environment metadata;
- a figure (PDF/PNG/SVG) and a caption.

The JSON also includes `configuration`, `validation_report`, `source_revision` and
`status`, where the status is `pass`, `fail` or `blocked`.

Every number comes from the NumPy reference engine on CPU. **No CUDA-Q or GPU execution
took place** (HYB-01 and HYB-11 are blocked).

| ID | Script | Question | Reference | Status |
|---|---|---|---|---|
| HYB-01 | `hyb01_11_cudaq_status.py` | Does real CUDA-Q run and match the reference? | `tests/cudaq_adapter`, `tests/hybrid` (skip-guarded) | **blocked** (Windows, no WSL) |
| HYB-02 | `hyb02_bridge.py` | Gaussian→Fock bridge moments and weights vs dims (pure and mixed, 1–2 modes) | exact Gaussian moments, Gaussian backend | pass |
| HYB-03 | `hyb03_09_10_homodyne.py` | Homodyne densities and conditional states | closed forms, Gaussian backend, native-fock | pass |
| HYB-04 | `hyb04_06_teleportation.py` | Feed-forward teleportation, fixed and 200 sampled outcomes | Gaussian backend (exact) | pass (dims 20 per-shot: recorded failure, see the docstring) |
| HYB-05 | `hyb05_cubic_mbqc.py` | **Primary**: three-mode cubic-phase CV-MBQC, cutoff sweeps | native-fock (cutoffs ≤ 48 / 72), piquasso-fock | pass |
| HYB-06 | `hyb04_06_teleportation.py` | Noisy teleportation with pure and thermal loss | Gaussian backend | pass |
| HYB-07 | `hyb07_08_kerr_counting.py` | Kerr-enhanced MBQC step, lossless and lossy | native-fock, native-mixed-fock | pass |
| HYB-08 | `hyb07_08_kerr_counting.py` | Photon counting after a subtraction tap, bridged squeezed input | native-fock | pass (first-run criterion failed, see the docstring) |
| HYB-09 | `hyb03_09_10_homodyne.py` | Finite-resolution bins → ideal, O(w²) | ideal instrument | pass |
| HYB-10 | `hyb03_09_10_homodyne.py` | Sampling (KS) and seed reproducibility | exact CDF | pass |
| HYB-11 | `hyb01_11_cudaq_status.py` | GPU performance | — | **blocked** (requires HYB-01) |
| HYB-12 | `hyb12_cpu_scaling.py` | CPU runtime and memory vs native engines; planner memory estimate | native-fock, native-mixed-fock | pass (CPU only; hybrid is slower) |

### Mapping from the suggested experiment list

The task prompt suggested a different numbering. Each suggested experiment is covered as
follows:

| Suggested | Covered by |
|---|---|
| HYB-01 real CUDA-Q operator validation | HYB-01 (blocked) |
| HYB-02 Gaussian-to-Fock conversion | HYB-02 |
| HYB-03 Fock homodyne distribution | HYB-03, HYB-09 (bins), HYB-10 (sampling) |
| HYB-04 conditional state update | HYB-03 (Gaussian and native-fock conditional states), HYB-04 |
| HYB-05 classical feed-forward | HYB-04 (linear), HYB-05 (quadratic and nonlinear) |
| HYB-06 Gaussian MBQC equivalence | HYB-04, HYB-06 (vs the exact Gaussian backend) |
| HYB-07 cubic-phase MBQC | HYB-05 |
| HYB-08 Kerr MBQC | HYB-07 |
| HYB-09 Lindblad photon loss | HYB-06, HYB-07 (lossy), `tests/hybrid` (exact Kraus) |
| HYB-10 cutoff convergence | dimension sweeps in HYB-02, -04, -05, -06, -07, -08 |
| HYB-11 CUDA-Q performance | HYB-11 (blocked); CPU scaling in HYB-12 |
| HYB-12 end-to-end hybrid demonstration | HYB-05; `docs/hybrid-cudaq/trajectories.md` example |

Run all of them from this folder (about 10 min on a laptop CPU):

```bash
for s in hyb01_11_cudaq_status hyb02_bridge hyb03_09_10_homodyne hyb04_06_teleportation \
         hyb05_cubic_mbqc hyb07_08_kerr_counting hyb12_cpu_scaling; do python $s.py; done
```
