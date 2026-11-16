# Limitations

## Not demonstrated

- **Real CUDA-Q execution: blocked.** `engine="cudaq"` is implemented, and its tests are kept, but it has never executed (Windows, no WSL). See [CUDA-Q real runtime validation](../validation/CUDAQ_REAL_RUNTIME_VALIDATION.md).
- **GPU execution or acceleration: none.** All timings in these pages are CPU timings of the NumPy reference.
- **Not a registered backend.** `pg.simulate(..., backend=...)` does not reach the executor. Use `photographiq.hybrid.executor.run`.

## Unsupported commands (planner reports them; `run` refuses)

The planner reports these commands and `run` refuses to execute them:

- `PrepareResource` (correlated resources), `CatResource` and GKP preparations, `Signal`;
- non-homodyne and non-counting measurements (GKP readouts, ...);
- inefficient or noisy homodyne;
- cross-Kerr, which has no pattern command.

Arbitrary-angle physical GKP-MuTA is **not** provided here, because it would need the
corresponding GKP measurement instrument.

PhotoGraphiQML (MuTA, MuTAKernel, MuTAClassifier, MuTARegressor, PhysicalMuTA, GKPBridge)
is unchanged. The optional PhotoGraphiQML hook is forward inference only:
`photographiqml.cv.hybrid.hybrid_moments` returns the q and p moments of a
`NonGaussianProgram` run through the hybrid executor. There is no hybrid loss, gradient
or training path.

## Truncation diagnostics

Gates use truncated-generator semantics, which are exactly unitary on the truncated space.
As a result, `retained_weight` (bridge and input truncation) does **not** certify the
convergence of the gate dynamics. The result therefore reports two further diagnostics:

| Diagnostic | Definition | Use |
|---|---|---|
| `truncation["max_boundary_population"]` | max over operations and modes of the population with n_k ≥ d_k − 2 (as the native engines) | catches states pushed into the top levels. A `RuntimeWarning` is issued above 0.02. |
| `truncation["max_conditional_boundary"]` | max over homodynes of ‖ψ̃_top(x)‖² / p(x), the share of the outcome density carried by the measured mode's top two levels | catches **rare outcomes**. Conditioning on a low-density outcome divides the absolute truncation error of the high levels by p(x). |

The second diagnostic was added after HYB-04 showed the problem. Teleporting with 200
sampled outcomes at dims 20 gave per-shot errors up to 1.1e-2, at a 3.7σ outcome. In that
run every boundary population stayed below 4e-5. The conditional diagnostic ranks those
shots (Spearman 0.85 with the per-shot error), and dims 32 brings the worst shot to
1.5e-4. Photon counting shows the same effect: rare counts (P = 8e-5) need larger
dimensions (HYB-08).

**Practice:** converge in `dims` for the outcomes that actually occur. Do not rely on the
norm alone.

## Accuracy and cost

- Per-mode truncation grows as ∏ d_k (pure) or (∏ d_k)² (mixed). The native engines' total-photon basis is smaller at equal accuracy. On CPU, the hybrid NumPy reference is slower than native-fock for the same patterns (HYB-12, about 7× for the three-mode cubic pattern at matched accuracy). The executor's value is the CUDA-Q-compatible representation and instrument layer, not CPU speed.
- Strong cubic phases converge slowly and non-monotonically in both engines (γ = 0.3 needed native cutoff 72 and hybrid dims 40 to agree to 1e-5; HYB-05).
- Thermal loss uses the truncated a†; check convergence in d.
- Sampling uses a 4001-point grid with linear interpolation (KS-validated). Very narrow conditional densities may need a larger `grid`.
- The binned instrument uses a 64-point midpoint rule per bin; this is accurate when the bin is small compared with the density's structure.
