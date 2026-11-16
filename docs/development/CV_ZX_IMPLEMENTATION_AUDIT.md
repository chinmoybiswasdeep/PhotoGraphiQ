# CV-ZX / CV-MBQC optimisation: implementation audit

**Starting point.** Branch `upgraded`, HEAD `c8d053a8c388d075086a7b642d52eddf3b318568`. The
working tree already held the uncommitted CV-native ML work of the previous task. Six tracked
files were modified:

- `README.md`
- `docs/COMPLETE_FUNCTIONALITY.md`
- `docs/ml/guide.md`
- `docs/ml/index.md`
- `mkdocs.yml`
- `src/photographiqml/__init__.py`

The new directories were `src/photographiqml/cv/`, `tests/ml/cv_native/`,
`paper_experiments_ml/15_cv_native/`, `docs/ml/cv-native/` and `examples/ml/cv_native/`, plus
two development documents and their results. All of it was preserved.

## Existing functionality (verified in the code)

| Component | What it is | Scope limits |
|---|---|---|
| `calculus.zx.Diagram` | Scalar-exact **qubit** ZX diagrams with dense tensor semantics. Rules: spider fusion, self-loop and identity removal, colour change, Pauli-state copy, 2×2 bialgebra, Hopf, parallel-Hadamard cancellation. | The module docstring says no CV-ZX identities are implemented. Unchanged here. |
| `calculus.gaussian_graph.GaussianGraph` | Pure Gaussian states as Z = V + iU (MFvL 2011), with an exact separate mean. Unitary update Z′ = (C + DZ)(A + BZ)⁻¹. `cz`, `quadratic_phase`, `rotate`, `squeeze`, `beamsplitter`, `measure_q` (vertex deletion), `measure_p` (Schur complement), `tensor`, `from_covariance` (rejects mixed states). | Pure states only: no mixed states, loss, non-Gaussian elements or channels. |
| `calculus.cv` | Command-level view of CV patterns. `affine_form` gives exact affine feed-forward. `ideal_limit_check` separates S from N. | Its docstring states explicitly that it is *not* a CV graphical calculus. |
| `analysis.gaussian_channel` | Exact unconditional channel of a fixed-angle, affine-feed-forward Gaussian `Pattern`, by source-vector propagation. | Rejects adaptive angles and nonlinear feed-forward. Cost grows with the command count (about 1 s per 100 measurements). |
| `cvflow.certify_cv_flow`, `flow_pattern` | Booth–Markham correction-matrix certificate for a supplied order. | Weights are frozen per certificate. |
| `flows.find_cv_flow` → `FlowAnalysisResult` | Order search with backward layering, exact rational fallback and determinism checks (`ideal_limit`, `finite_squeezing` regimes). | `Model.CV_GAUSSIAN` only. |
| `compiler.Circuit` / `compile_circuit` | Gate lowering: four or five teleportation steps per single-mode symplectic; CZ followed by four Fourier steps per mode; beam splitter as six SUMs; displacement as four steps followed by `Displace`. | `CompilationTrace` gives provenance. |
| Backends | `gaussian` and `native-gaussian` (exact Schur-complement conditioning, postselection, densities); `native-fock` and `native-mixed-fock` (Kerr, cubic phase, ladder operators, photon counting, retained norms, truncation guard). | The Fock engines refuse a retained norm below 1 − 1e-3. |
| GKP | `GKPCode.encode` (finite-energy `FockInput`), `multimode_readout`, `PhysicalGKPReadout`, `GKPLogicalXYMeasurement`. | Finite energy only. |
| Untruncated GKP grid reference | `tests/validation/gkp_xy_reference.py` (from the GKP XY task). | Independent of the Fock engines. |

## Conventions

- **Quadratures:** ħ = 2, [q, p] = 2i, interleaved order, vacuum covariance I.
- **Channels:** an affine channel (X, Y, d); a symplectic map satisfies S Ω Sᵀ = Ω.
- **Graphs:** in `GaussianGraph`, Z is convention-independent.
- **Literature conversions:** Nagayoshi et al. use ħ = 1 and the opposite rotation sign; see
  `docs/theory/CV_ZX_FOUNDATIONS.md`.

## Unsupported before this work

- No typed CV process diagrams.
- No rewrite of Gaussian channels or instruments.
- No pattern ↔ diagram translation.
- No resource optimiser.
- No non-Gaussian diagram rules.

## Known numerical limitations

- Near-singular homodyne angles: θ ≈ 0 gives the feed-forward gain 1/sin θ.
- Highly squeezed resources: the e^{2r} anti-squeezed variance. Flow corrections cancel it
  exactly, but the full covariance is badly conditioned.
- Fock cutoffs for GKP and non-Gaussian states. Two-mode GKP needs cutoffs of 64 or more.

## API conflicts considered

- `calculus.zx.Diagram` and `RewriteStep` are qubit-specific. The new package uses its own
  `CVDiagram` and `RewriteStep` under `calculus.cv_diagrams`, so neither name is shadowed.
- No existing API changed. `calculus/__init__.py` is untouched; the subpackage is imported as
  `photographiq.calculus.cv_diagrams`.

## Integration points used

| Point | How it is used |
|---|---|
| `calculus.cv.affine_form` | Translating feed-forward. |
| `analysis.gaussian_channel` | Independent semantic reference. |
| `GaussianBackend` trajectories | Conditional reference. |
| `GaussianGraph` | Pure-state interoperability and measurement-rule references. |
| `flows.find_cv_flow` and `flows.cv.corrected_pattern` | Flow certificates and layered schedules. |
| `compiler.decompose_symplectic` | Resynthesis fallback. |
| Native Fock engines | Non-Gaussian and GKP semantics. |
| `GKPCode` | GKP preparation. |
| `paper_experiments/common.py` | Saving results and figures. |

## Recommended scope (followed)

The work follows the prompt's own priority: a rigorously verified Gaussian process calculus
and executable CV-MBQC optimisation first. On top of that sits a restricted non-Gaussian layer
of operator identities with proofs and Fock convergence evidence. No completeness or
universality claims are made.
