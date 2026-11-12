# Phase 5: Final scientific software release audit

Audit date 2026-11-12. Scope: the local working tree of branch `upgraded` (HEAD `f30321f` plus
unstaged changes). Detailed evidence: [Phase 1](PHASE1_RELEASE_AUDIT.md),
[Phase 2](PHASE2_SCIENTIFIC_AUDIT.md), [Phase 3](PHASE3_FEATURE_COMPLETION.md),
[Phase 4](PHASE4_MANUSCRIPT_REPRODUCIBILITY.md), and the
[master status](MASTER_VALIDATION_STATUS.md).

## 1. Executive summary

PhotoGraphiQ is a CV-MBQC platform with native C++ Gaussian, Fock and mixed-Fock engines,
boson-sampling and GBS routines, qubit and CV flow and calculus tools, interactive
visualization, and the PhotoGraphiQML learning layer.

- **Independent validation:** its core physics is checked against first-principles references (closed forms, enlarged-space `expm`, 40–60-digit mpmath, exact rational arithmetic) and against Piquasso, Graphix and MentPy.
- **CI defects found:**
  - notebook lint;
  - a Numba 0.68 cache-reload crash in Piquasso;
  - a pytest configuration that shadowed installed wheels.

  All three are fixed locally.
- **Scientific defects found:**
  - unstable one-mode Gaussian unitaries at large size;
  - a direct-measure angle bug;
  - conflated "impossible" and "uncertain" CV-flow outcomes.

  All three are fixed and tested.
- **Recommendation:** **publication-ready with limitations** (section 18). Two conditions remain: green remote CI on Linux after the maintainer pushes, and the documented limitations reflected in the manuscript.

## 2. Software architecture

Pipeline: user API → `OpenGraph` / `Pattern` IR → compiler and flow analysis → capability registry
→ engines (NumPy Gaussian; native C++ Gaussian, Fock and mixed-Fock; optional Piquasso) →
results → visualization. Direct entry points bypass the planner: `photographiq.sampling`,
`photographiq.gpu` (CuPy) and `photographiq.cudaq_dynamics` (adapter). PhotoGraphiQML builds on
the core. See figures A–H (`docs/figures/architecture/`).

## 3. Implemented capabilities

282 feature entries (2026-11-12 recount) in [COMPLETE_FUNCTIONALITY.md](../COMPLETE_FUNCTIONALITY.md):

| Status | Count |
| --- | --- |
| Implemented and validated | 168 |
| Implemented but experimental | 91 |
| Partially supported | 8 |
| Unsupported | 13 |
| Planned | 2 |

## 4. Independently validated capabilities

- **Native Gaussian:** states, gates, channels and homodyne/heterodyne conditioning; max error 6.5e-13.
- **Native Fock and mixed-Fock:** states, all gates, photon counting, homodyne, loss and Kraus instruments; max error 5e-16.
- **One-mode Gaussian unitaries:** ≤ 1.9e-14 against 60-digit mpmath.
- **Matrix functions:** permanents and (loop) hafnians against their definitions; threshold probabilities against 40/50-digit references, with a bound that holds in all 382 matrix cases.
- **Flows:** qubit flows against Graphix (360/360); CV-flow impossibility with exact rational witnesses.
- **Calculus:** ZX rewrites by tensor evaluation; Gaussian graphical calculus against Gaussian numerics (2e-14).
- **PhotoGraphiQML:** logical results against MentPy (bit-identical recorded values).

## 5. Experimental features

- Piquasso-backed Fock engines, cat resources, photon subtraction, GKP resources and synthesis (labelled experimental in the code).
- The few-click calibration factor of the threshold bound.
- Interactive layout quality.
- `photographiq.autodiff` (JAX).
- Physical PhotoGraphiQML.
- CUDA-Q execution.

## 6. Unsupported features

- Heterodyne and general-dyne on Fock backends.
- Pattern execution on CUDA-Q.
- GPU sampling kernels (measured, not justified).
- General CV-ZX and CV local complementation.
- Arbitrary-angle physical MuTA.
- Exact (non-float64) certification of the finite-squeezing channel.

## 7. Native backend assessment

**Ready.** The native engines replace Piquasso for every supported operation. Native speed on
this laptop:
- Fock gates: median 44× over Piquasso.
- Fock patterns: 7–40× over Piquasso.
- Gaussian: algorithmic gains up to 7500× at n = 256 for single gates; batched gains 1.2–10× over vectorized NumPy.

Limit: multi-mode displaced Gaussian unitaries have no stability bound. The backends never
request them; a direct `gaussian_unitary` call can.

## 8. Numerical accuracy assessment

- **Gaussian:** rounding-level.
- **Fock:** truncation must be shown per workload. `cutoff_convergence` and `measurement_convergence` support this on native backends, and the native and Piquasso engines differ only at truncation level for the CZ.
- **Threshold probabilities:** lose accuracy at small squeezing with many clicks (in double precision, 80/360 matrix cases exceed 1e-6 relative error; Piquasso fails in the same regime). PhotoGraphiQ always reports a bound, warns, and offers `precision="auto"`.
- **CV determinism:** for r + 2·shift ≳ 9.5 or extreme weights it is `unknown` unless `exact=True` is used.

## 9. GPU implementation status

| Layer | Status |
| --- | --- |
| CuPy GPU acceleration | Batched Gaussian and batched Fock (P U P) kernels, validated on an RTX 4050 against CPU and native to ≤ 7e-16. Faster for batches; slower for single small states. |
| Native CUDA kernels | None (no `nvcc` on this machine). |
| CUDA-Q integration | Hamiltonian adapter. Translation validated with a NumPy reference; execution on CUDA-Q unverified (Linux only; no WSL or Docker here). |
| CPU-only | Every feature works without a GPU. |

## 10. Benchmark findings

Source: `benchmarks/results_20261112/`; tables in [Phase 4](PHASE4_MANUSCRIPT_REPRODUCIBILITY.md).

- **Where native code is faster:** native Fock is much faster than Piquasso, and native Gaussian is much faster than the dense NumPy backend.
- **Where it is at parity or slower:** PhotoGraphiQ's permanent, hafnian and torontonian kernels are at parity with or slower than Piquasso's. The qubit flow, standardisation and simulation tools are slower than Graphix (median 18× for gflow and Pauli flow).
- **Accuracy of threshold probabilities:** about 10× worse than Piquasso in double precision for 24–32 modes.

## 11. PhotoGraphiQML compatibility

- `tests/ml`: 154/154 pass.
- MentPy reference tests: 34/34 pass in an isolated environment, with the CI's MentPy commit.
- The 24 experiments run in the isolated environment reproduce the recorded values bit for bit, and the 28 project-venv experiments match the recorded files.
- Physical execution on `native-fock` agrees with `piquasso-fock` at truncation scale, with identical decoded bits. `piquasso-fock` remains the default for reproducibility.

## 12. Manuscript reproducibility

- **Entry point:** `scripts/reproduce_manuscript.py` (fast and full modes; structured `summary.json`; dependency-aware skips; experiments run from copies).
- **Full run on 2026-11-12:** 25 of 27 steps passed.
  - The 2 failures are the 9 MentPy-dependent ML scripts (MentPy is not in the project venv; they are reproduced separately) and the schema drift between the recorded core results and the current scripts.
  - Every value common to both versions is reproduced.
- **Recommendation:** regenerate the recorded core results before submission. Heavy experiments were not re-run.

## 13. CI and platform compatibility

| Platform | Status |
| --- | --- |
| Windows 11, Python 3.11 / 3.12 / 3.13 / 3.14 | full suite green (after `numba<0.68`) |
| Linux (CI ubuntu) | not re-run: the fixes are unstaged, and the remote run needs a push by the maintainer. The two identified root causes (lint, Numba) are reproduced and fixed locally. |
| macOS | untested |

## 14. Packaging quality

- **Wheel:** 0.69 MB; only `photographiq`, `photographiqml` and `_native`.
- **sdist:** 30.2 MB. It includes the experiment suites, benchmark results and docs, so it is a complete reproducibility archive; `.vscode` and root images are now excluded.
- **Core-only install** (NumPy, SciPy and NetworkX): backends `gaussian`, `native-gaussian`, `native-fock` and `native-mixed-fock`.
- **Missing extras:** they raise errors naming the extra or wheel to install (Piquasso, mpmath, CuPy, CUDA-Q).
- **Editable install:** works; tests now import the installed package.

## 15. Remaining defects

None known that affect correctness in a validated regime. Known weaknesses:
- **Interactive views:** straight edges can pass under other nodes; outcome labels can overflow nodes; dense grids have small labels.
- **Ordering of feed-forward edges:** the 6-node flow drawing can read ambiguously.
- **sdist size.**

## 16. Release blockers

1. **Remote CI on Linux is unconfirmed** for the current tree. Push and observe the `tests` and `documentation` workflows.
2. **Recorded core experiment results** use an older schema than the current scripts. Regenerate them, or document the version that produced them.

There is no other blocker; the remaining items are documented limitations.

## 17. Final Git status

Read-only checks run at the end of the audit, 2026-11-12:

| Check | Result |
| --- | --- |
| `git branch --show-current` | `upgraded` (unchanged) |
| HEAD (`git log --oneline -1`) | `f30321f modified` at start and at end; no commit was created |
| `git diff --cached --stat` | empty: nothing staged at start or at end |
| `git diff --check` | exit 0 (only CRLF→LF normalisation notices from `.gitattributes`) |
| `git diff --stat` | 107 tracked files modified, 6855 insertions, 2597 deletions; 0 tracked files deleted |
| Untracked new files | 90: `benchmarks/results_20261112/` (47), `docs/validation/` (18), new modules `src/photographiq/cudaq_dynamics.py`, `calculus/gaussian_graph.py`, `flows/_exact.py`, `gpu/fock.py`, new tests (12), new scripts (3), two benchmark scripts with their two result files (4), `docs/cudaq.md`, `docs/user-guide/gaussian-graphical-calculus.md` |

The working tree was clean at the start (HEAD `f30321f`, equal to the reference revision). No
branch was created or switched, nothing was staged, committed or pushed, no history was
rewritten, and `.git` was not modified. All changes are local and unstaged.

## 18. Recommendation regarding publication readiness

**Publication-ready with limitations**, provided:
- the two release blockers above are cleared;
- the manuscript limits its claims to the validated capabilities (section 4) and states the limits in sections 5–8 and 10.

**Safe to claim:**
- native Gaussian, Fock and mixed-Fock engines validated against independent references, with the measured speedups over Piquasso and dense NumPy;
- CV-MBQC patterns with feed-forward, compilation and finite-squeezing channels;
- qubit and CV flow analysis, with CV certificates that state their regime;
- BS/GBS probabilities with explicit error control;
- the validation methodology;
- interactive visualization;
- PhotoGraphiQML integration validated against MentPy.

**Exclude or describe as future work:**
- CUDA-Q execution;
- GPU sampling;
- general CV-ZX;
- speed advantages for matrix functions or qubit tools;
- double-precision accuracy of many-click threshold probabilities.
