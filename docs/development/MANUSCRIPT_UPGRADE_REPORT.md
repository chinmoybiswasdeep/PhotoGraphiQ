# Manuscript upgrade report (2026-11-13)

Release hardening, documentation modernisation and manuscript-experiment expansion for the
Quantum manuscript "PhotoGraphiQ: A photonic continuous variable software platform to perform
measurement-based quantum computing". Task-by-task status: [tracker](MANUSCRIPT_UPGRADE_TRACKER.md).

All numbers below come from runs on 2026-11-13 on one machine: Windows 11, Intel i5-12450HX
(12 logical CPUs), NVIDIA RTX 4050 Laptop GPU, Python 3.12.14 unless stated. No Linux machine was
available, and nothing was pushed, so **no claim here is a remote CI result**.

## 1. Repository state

| Item | Value |
|---|---|
| Branch | `upgraded` (checked at start and end; never switched) |
| HEAD at start | `441de9a` "modified 2" (clean working tree) |
| HEAD at end | `441de9a`: no commit was made |
| Staged changes | none at start, none at end |
| Tracked files modified | 623 (0 deleted): mostly regenerated experiment results, figures and tables, plus source, tests, docs |
| Untracked new files | the new experiment directories `paper_experiments/14_*` … `21_*`, their outputs, `legacy_reference/` archives, `docs/figures/demos/`, `docs/theory/{cluster-states,calculus,sampling}.md`, `docs/ml/guide.md`, `docs/development/MANUSCRIPT_UPGRADE_{TRACKER,REPORT}.md`, `scripts/{audit_manuscript_consistency,generate_demo_figures}.py`, `src/photographiq/_checks.py`, `src/photographiq/interactive/layouts.py`, `.github/workflows/manuscript.yml` |

The exact lists are produced by `git status --short` (section 10).

## 2. CI repair

Run [43224033322](https://github.com/chinmoybiswasdeep/PhotoGraphiQ/actions/runs/43224033322)
(`441de9a`): `tests` failed on Python 3.11-3.14; `documentation` and `ml-reference` passed. The
validation artifacts and job logs require authentication (HTTP 401), so the failures were
reproduced locally instead.

| Python | Failed gate | Root cause | Fix | Validation |
|---|---|---|---|---|
| 3.11-3.14 | mypy | mypy 2.4.1 with NumPy 2.5.4 stubs narrows `np.isscalar(x)` to a union that `np.isreal`/`np.isfinite` reject: 11 errors in 5 files (static typing, category 9). Reproduced exactly in a fresh environment with the latest dependencies | `src/photographiq/_checks.py:is_finite_real_scalar`, same semantics, used at all five call sites (`backends/gaussian.py`, `backends/native_gaussian.py`, `backends/mixed_fock.py`, `encoded.py`, `photographiqml/physical.py`) | mypy clean with mypy 2.3.1 and 2.4.1 |
| 3.11-3.14 | pytest | The pristine `441de9a` tree passes on Windows with the latest dependencies (1671 passed; the one failure is an artifact of the archive copy having no `.git`). So the remote failure is Linux-specific. By inspection: `tests/visualization/test_interactive_browser.py` skips when no Chromium browser exists, but GitHub's ubuntu runners have Google Chrome, whose sandbox cannot start under the AppArmor user-namespace restriction of Ubuntu 24.04+ (test-environment issue, category 8) | `--no-sandbox` on Linux in `run_page`; the assertion that reports Chrome's stderr is kept, and the failure is not turned into a skip | Passes on Windows. **Unverified on Linux**: the job logs could not be read, so this is the most likely cause, not a confirmed one |

Also checked and found clean: letter case of every path literal (Linux file systems are
case-sensitive), OS-specific tools, timing assertions in tests.

**Final local gates** (`experiments/validate_release.py`, Python 3.12, final tree): all passed:
pytest 1724 passed, 13 skipped, 0 failed (18 min); coverage 93.93 %; ruff check and format clean;
mypy clean (92 files); wheel and sdist build; physics evidence. The 13 skips are declared
optional cases (MentPy, Playwright and CUDA-Q not installed; backend-capability combinations;
Piquasso has no cross-Kerr).

**Python matrix**: see section 10 for the 3.11, 3.13 and 3.14 results.

**Documentation and ML-reference workflows** passed remotely at `441de9a`; locally,
`check_docs_examples.py --projects --notebooks`, `check_public_api.py` and `mkdocs build --strict`
results are in section 10.

**To verify remotely:** commit and push the working tree, then watch the `tests`, `documentation`,
`ml-reference` and new `manuscript` workflows. If `tests` still fails, download the
`validation-python-3.x` artifact (`pytest.log`) and look first at
`test_interactive_browser.py`.

## 3. Documentation

- **README** rewritten as a unified platform overview: summary, honest validation status (local
  vs remote CI), architecture figure, capabilities with executed examples, validated /
  experimental / restricted / unsupported table, GPU and CUDA-Q limits, installation, quickstart,
  validation, benchmarks, manuscript experiments, citation.
- **Installation** (`docs/getting-started/installation.md`): platform support (Linux = CI,
  Windows = verified locally, macOS untested), C++ toolchain, CPU-only and GPU extras, CUDA-Q
  status, and an installation check that runs real backend operations (executed).
- **Theory**: seven new or completed pages in `docs/theory/` (Gaussian, compilation, cluster
  states, flow, calculus, non-Gaussian, sampling) in the codebase conventions (hbar = 2), each with
  executed checks, keeping ideal and finite-squeezing relations apart.
- **Backends** (`docs/user-guide/backends.md`): all registered backends, CuPy kernels, CUDA-Q
  status and sampling kernels, from the registry.
- **PhotoGraphiQML**: `docs/ml/guide.md` (models, backends, validation) and caveats.
- **Corrected statements**: the old theory page said pure Fock states are "evolved by Piquasso"
  and that noisy Fock homodyne and GKP are not implemented; `docs/index.md` showed Piquasso under
  every engine; `docs/user-guide/non-gaussian.md` claimed noisy Fock homodyne, mixed inputs and GKP
  were unavailable; several snippets lacked imports or used undefined names; citation version was
  0.3.0; `CITATION.cff` too.
- **Historical pages** (v0.2/v0.3 reports, design notes) carry a "historical record" banner
  instead of being rewritten.
- **Registry fix found by the audit**: `native-fock` and `native-mixed-fock` now declare
  renormalisation among their approximations (they renormalise after every operation).

## 4. Interactive and publication graphics

- **Graph construction → flow → corrections → execution**: the existing `GraphView` already
  built all four stages from the real flow result, correction map and simulator trace. Added:
  stage and layout selectors, fit/reset, measurement-order stepping, hover tooltips (squeezing in
  dB, CZ formula), downloads (SVG with stable element ids, 3x PNG, self-contained HTML that
  reopens with the current positions and stage); semantic edits still go through `apply_edits`.
- **Visual language** shared by HTML and static output: CZ edges neutral solid; flow arrows solid
  blue `#0072b2`; corrections dashed orange `#d55e00`; inputs blue squares, outputs green
  diamonds, measured nodes thick dark outline (not colour alone); uncertified flow drawn grey and
  dashed.
- **Known layout weaknesses fixed**: routed edges and arrows keep 0.34 layout units from every
  non-endpoint node (same routing in Python and JS); nodes grow to fit labels; dense grids render
  labels at 10.9 px (was 8.8 px); ambiguous feed-forward arrows resolved.
- **Honesty fix**: `from_pattern` dropped constant `Displace` commands silently; it now annotates
  them and the outcomes adaptive gates depend on.
- **Browser**: `scripts/check_interactive_browser.py` 12/12 in Edge with real input events;
  Firefox, WebKit and touch untested.
- **Demonstration figures** (`docs/figures/demos/`, SVG/PDF/PNG/caption, HTML where useful):
  single-mode teleportation, two-mode and multimode clusters, compilation, adaptive feed-forward,
  flow layering, correction maps, cubic injection, GKP syndrome extraction, native execution,
  boson sampling, GBS, PhotoGraphiQML, end-to-end.

## 5. Existing experiments R1-R42

Historical outputs are preserved byte-for-byte in `paper_experiments/legacy_reference/441de9a/`
(157 files, SHA-256 manifest; figures by hash, retrievable from git) and the old catalogue in
`legacy_reference/441de9a_catalogue/`. Their own metadata says they came from branch
`manuscript-experiments` (commit 9304081, 2026-09-15) using only the NumPy and Piquasso engines;
nothing was relabelled. Full per-experiment matrix: `paper_experiments/UPGRADE_MIGRATION_MATRIX.md`.

| Group | Updated engines | Independent validation | Result (final idle run) |
|---|---|---|---|
| R1-R2 teleportation | gaussian, native-gaussian (+piquasso) | analytic channel; Fourier map; Tr N = 4e^-2r | pass; moments within 2.7 standard errors; map 6e-17 |
| R3-R10 compilation | analyser + both Gaussian engines | NumPy targets; noise oracles | pass; map errors 4e-16 to 7.8e-12 (400 random symplectics); noise reported separately |
| R11-R14 feed-forward and CV flow | both Gaussian engines, `photographiq.flows` | fixed-angle controls, exact witnesses | pass; 12/12 invalid patterns rejected; 7/7 decline-to-certify cases correct |
| R15-R18 external references | 3 Gaussian + 4 Fock engines | raw Piquasso, NumPy targets, Graphix (structure) | pass; 120 cases to 2.7e-15 |
| R19-R25 non-Gaussian | native-fock (+piquasso-fock, native-mixed-fock) | binomial law, coherent expansions, `expm`, exact phase rule | pass; 4.4e-16 (135 branches), cat fidelity ≥ 1-2e-16 |
| R26-R28 GKP | native-fock (+piquasso-fock) | engine agreement, exact readout integrals | pass; syndrome errors 9/144 (Wilson 95 % [0.033, 0.114]); no fault-tolerance claim |
| R29-R32 noise | Gaussian + mixed-Fock engines | analytic loss/thermal channels | pass; ≤ 1.9e-15 |
| R33-R34 convergence | native-fock, piquasso-fock | exact moments, `expm` gate | pass; native reaches r = 1 at cutoff 96 where piquasso stops |
| R35-R38 performance | all engines | correctness gate before timing | pass; idle timings in section 7 |
| R39-R40 serialization, seeds | Gaussian engines, native-fock | round trip, RNG contract | pass; 16/16 checks; native-gaussian samples equal NumPy |
| R41-R42 autodiff | JAX; native-fock finite differences | central differences, analytic gradients | pass; JAX vs native finite difference 6.7e-9 at cutoff 40 (native engines have no autodiff) |

Four checks added during migration failed on first run and were diagnosed before any change
(R10 conditioning, R17 non-unique flows, R28 an unfounded 1e-6 tolerance, R41 truncation at the
historical cutoff); see `paper_experiments/STATUS.md`.

## 6. PhotoGraphiQML

- `tests/ml`: 154/154 in the MentPy environment; 120 passed and 2 modules skipped (MentPy
  absent) in the project environment. MentPy reference tests 34/34.
- Manuscript suite: 59/59 experiments pass in the MentPy environment; in the project
  environment, 9 safe-suite scripts fail only with `ModuleNotFoundError: mentpy` (MentPy
  unavailable, not a failure).
- 10175 of 10349 recorded JSON values are bit-identical to the frozen evidence; the rest are
  timings, memory readings and one NaN.
- native-fock vs piquasso-fock on the manuscript scripts: decoded differences 4.6e-5 to 1.2e-4.
  At the default GKP resource Piquasso needs a cutoff of about 48 to match native-fock to 1e-4.
- `verify_evidence.py` reports provenance errors only (results produced from uncommitted, changing
  sources). Accepted evidence needs `run_all_full.py --publication` on a clean committed tree.
- Historical outputs: `paper_experiments_ml/legacy_reference/441de9a/` (394 checksummed files).

## 7. New experiments R43-R60

All pass in their publication configuration (`--full` where defined) on the idle machine.
Write-ups with acceptance criteria and limitations: `paper_experiments/NEW_EXPERIMENTS_*.md`.

| ID | Objective | Validation | Result |
|---|---|---|---|
| R43 | Native Gaussian engine correctness | `expm` symplectics, dilation conditioning, NumPy | covariance ≤ 1.6e-12 relative over 80 circuits up to 128 modes and depth 2560; symplectic residual 2e-15 |
| R44 | Native Gaussian performance | accuracy alongside timing | 3607x faster than the NumPy backend at n = 256 (200 gates), 2936x faster than Piquasso at n = 128; accuracy 3.4e-14 |
| R45 | Native Fock accuracy | mpmath, 50-digit sums, `expm`, Piquasso | worst 1.7e-15 over cutoffs 6-96; Piquasso 3.9e-6 vs native 7.2e-16 on D(4)\|c/2⟩ |
| R46 | Implemented Fock optimisations | identical outputs across variants | 1.7e-16 agreement; D = 12870: native 0.54 ms vs scipy sparse 0.74 ms |
| R47 | Native mixed-Fock | analytic channels, Gaussian moments | 4.4e-16; min eigenvalue -8.8e-16 |
| R48 | Boson sampling | definition, Glynn, Fock evolution, Piquasso | permanent n = 20 relative 2.7e-11; distributions 5e-16 |
| R49 | GBS (PNR) | exact-in-truncation Fock, Piquasso | 5.6e-16 over 5117 patterns |
| R50 | Threshold GBS | 40-digit reference | bound held 185/185; double precision > 1e-6 in 20/170 cases (Piquasso 21); `precision="auto"` within 1.9e-8 |
| R51 | CV flow finding | exact rational search | 57/57 decided graphs agree; 20/20 witnesses valid |
| R52 | CV determinism | linear Wigner model | S to 3.8e-16; noise N falls with slope -2.000 in r; never zero at finite r |
| R53 | Qubit flows vs Graphix | definition checker, branches | 89/89 agreement per notion |
| R54 | Measurement calculus | branch-wise Choi simulation, Graphix | 312 branches, ≤ 5.6e-16 |
| R55 | Local Clifford | graph-state builder, invariants | ≤ 7.8e-16; CV local complementation not implemented |
| R56 | ZX (qubit subset) | independent tensor evaluator | 171 checks ≤ 7.8e-16 |
| R57 | Interactive visualization | channel equality, SVG parsing, real browser | 20/20 checks |
| R58 | GPU Gaussian | NumPy and native on identical batches | float64 ≤ 3.8e-16, float32 ≤ 7e-7 |
| R59 | PhotoGraphiQML physical execution | piquasso-fock, logical MuTA | wide resource \|ΔP\| ≤ 9e-5; 2 Piquasso guard trips at the moderate resource recorded as data |
| R60 | End-to-end showcase | analytic S, NumPy, ensemble statistics | compiled S 5.6e-16; Tr N slope -2.0000; z-scores 1.4 and 1.9 |

**Idle-machine timings** (single-machine observations; final run 2026-11-13):

| Comparison | Measured |
|---|---|
| Native vs NumPy Gaussian backend, n = 256, 200 gates (R44) | 3607x |
| `simulate`, N = 200 line with measurements (R36) | native 0.074 s, NumPy 18.5 s, Piquasso 119 s |
| Native-gaussian vs raw Piquasso program, N = 40 (R38) | 5.8 ms vs 9.5 ms |
| native-fock vs piquasso-fock, two modes, cutoff 64 (R35) | 30.8 ms vs 0.49 s |
| GPU, n = 16, B = 16384, dense symplectic (R58) | NumPy 1412 ms, native 443 ms, GPU end-to-end 123 ms (kernel 53 ms, transfers 72 ms) |
| GPU, same size, homodyne conditioning (R58) | NumPy 404 ms, native 17.5 ms, GPU end-to-end 59 ms: native wins |

## 8. Manuscript narrative

Recommended structure (details and figure files: `paper_experiments/MANUSCRIPT_MAP.md`):

1. Introduction and architecture (Fig. 1, code-validated architecture figure).
2. Programming model: graphs, patterns, compilation, flow, feed-forward (Fig. 2: R51 + R57;
   Fig. 3: R5, R29, R52).
3. Native simulation and accuracy (Fig. 4: R43, R45; appendix R46, R47, R18-R34).
4. Photonic sampling (Fig. 5: R48-R50).
5. Performance (Fig. 6: R44, R35, R58; idle timings only).
6. End-to-end showcase (Fig. 7: R60).
7. Extensions: non-Gaussian/GKP or PhotoGraphiQML (optional Fig. 8: R23/R28 or R59).
8. Validation methodology and reproducibility (`reproduce_manuscript.py`, provenance, audit).

Main text: R5, R29, R43-R45, R48-R52, R57, R60 and the performance set. Supplement: gate sweeps,
calculus (R53-R56), GKP (R26-R28), noise (R30-R32), convergence (R33, R34), serialization,
autodiff, GPU details, R59.

## 9. Scientific limitations

- **CUDA-Q**: adapter only; no CUDA-Q execution has ever run (Linux-only wheels; no WSL/Docker).
- **CV calculus**: no general CV-ZX, no CV measurement calculus, no CV local complementation.
- **GKP**: finite-energy results with postselected or ensemble statistics; no fault-tolerance or
  threshold claim. R28's decoded residual sits about -0.05 below the ideal at every delta.
- **Threshold GBS**: double precision fails in the many-click, low-squeezing regime; the error
  bound and multiprecision path are the result.
- **Fock homodyne sampling** is slow on all Fock engines (seconds per shot at cutoff 48; 8-22 s
  per noisy mixed-Fock shot): a Python root-find per shot, not the C++ kernels.
- **Matrix functions and qubit tools** are at parity with or slower than Piquasso and Graphix
  (see `docs/validation/PHASE4_MANUSCRIPT_REPRODUCIBILITY.md`).
- **GPU** helps batched workloads only; there are no native CUDA kernels.
- **Platform**: Windows verified locally; Linux CI unconfirmed for this tree; macOS untested.
  Browser checks in Edge only.
- **Provenance**: every result was produced from uncommitted changes (`git_dirty: true`, source
  SHA-256 recorded). Accepted evidence needs a rerun on the committed tree.
- **Engineering change during the work**: the native engines are now built with
  `EIGEN_DONT_PARALLELIZE`. Eigen's own threads made small mixed-state gates up to 8.6x slower;
  the Gaussian benchmark was unchanged within noise.

## 10. Final checks

| Check | Result |
|---|---|
| `experiments/validate_release.py` (Python 3.12, final tree) | all gates passed: pytest 1724 passed / 13 skipped / 0 failed, coverage 93.93 %, ruff, format, mypy (92 files), build, physics evidence |
| Python 3.11.16 / 3.13.15 / 3.14.4 (fresh venvs, current tree installed as a wheel) | each: 1675 passed, 62 skipped, 0 failed; mypy clean. The extra skips are GPU, notebook and browser tests (no CuPy, nbclient or Playwright in those venvs). One environment still had numba 0.68.1 from before the `numba<0.68` pin and aborted in Piquasso's JIT: the known crash that the pin prevents; with the pinned dependencies it passes |
| `scripts/check_public_api.py` | 85 exports and API modules import |
| `scripts/check_docs_examples.py --projects --notebooks` | 52 PASS |
| `mkdocs build --strict` | clean |
| `scripts/audit_manuscript_consistency.py` | 12/12 checks pass |
| `paper_experiments/run_all_safe.py --all` (idle machine) | 60/60 pass (section 5 and `paper_experiments/STATUS.md`) |
| Core notebook, smoke mode (nbclient) | 16 experiments, 0 errors |
| PhotoGraphiQML suite | 59/59 (MentPy environment) |
| `scripts/reproduce_manuscript.py --mode smoke` | 16/16 steps passed (test subsets, figure check, consistency audit, 16 smoke experiments from a copy, 6 benchmark smoke runs incl. GPU) |
| `git branch --show-current` | `upgraded` |
| `git log -1` | `441de9a` (unchanged; no commit) |
| `git diff --cached --stat` | empty: nothing staged |
| `git diff --check` | exit 0 |
| `git diff --stat` | 623 tracked files modified (619 with content changes after line-ending normalisation), 0 deleted |

No branch was created or switched; nothing was staged, committed or pushed; no history was
rewritten and `.git` was not modified. `git diff --stat` does not count untracked files; the new
files are listed in section 1.
