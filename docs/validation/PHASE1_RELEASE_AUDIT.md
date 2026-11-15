# Phase 1: Release engineering, CI, installation and regression hardening

Audit date 2026-11-12. All work was done in the local working tree of branch `upgraded`. No
commit, stage or push was made, so the remote CI has **not** been re-run on these changes
(see "Unresolved blockers").

## Initial repository status

| Item | Value |
| --- | --- |
| Branch | `upgraded` |
| HEAD | `f30321f modified` (equal to the reference revision `f30321fd09140b310c3fb53e02a9b503e756855f`) |
| Working tree at start | clean: no modified, untracked or staged files |
| Machine | Windows 11 (10.0.26200), Intel i5-12450HX (12 logical CPUs), NVIDIA RTX 4050 Laptop GPU |
| Toolchain | MSVC 19.50, C++20, OpenMP, Eigen 3.4.0, CMake via scikit-build-core |
| Not available locally | Linux, WSL and Docker, so the ubuntu CI cannot be reproduced bit for bit; CUDA-Q (Linux wheels only); `gh` CLI |

## CI failures investigated (GitHub Actions, HEAD `f30321f`, 2026-11-12)

Read through the public REST API. Job logs need authentication (HTTP 403), so the failing
commands were reproduced locally.

| Workflow | Run | Result | Failing step |
| --- | --- | --- | --- |
| `tests` | 43145860723 | failure on 3.11, 3.12, 3.13, 3.14 | "Test, lint, format, type-check and build (all gates)" = `experiments/validate_release.py` |
| `documentation` | 43145860716 | failure | `python scripts/check_docs_examples.py --projects --notebooks` |
| `ml-reference` | 43145860809 | success | n/a |

## Root causes

1. **Lint and format gates (tests workflow, every Python version).**
   - `validate_release.py` runs `ruff check` and `ruff format --check` over `examples/` and `experiments/` as well as `src/` and `tests/`.
   - At `f30321f` ruff reports 22 findings: 17 in `examples/notebooks/boson_sampling_and_gbs_benchmark.ipynb`, 1 in `examples/notebooks/interactive_graph_visualization.ipynb` and 4 in `examples/tutorials/tutorials.ipynb`. They are E401, E402, F401, I001 and E702.
   - In addition, 3 notebooks are unformatted.
   - This is a confirmed, deterministic failure on every Python version.
2. **Numba 0.68 cache-reload crash (documentation workflow; also the Fock benchmark smoke test).**
   - With `numba==0.68.1` and `llvmlite==0.50.0`, which a fresh `pip install` resolved on 2026-11-12, Piquasso's `cache=True` hafnian kernels compile and cache in the first process. Every later process then aborts while loading the cache:
     `LLVM ERROR: Symbol not found: .numba.unresolved$_ZN8piquasso5_math7hafnian10hessenberg16_calc_vH_times_AB...`
   - Reproduced on Windows. Tutorial 09 on the `piquasso-fock` backend passed its first run, then failed the second and third runs with the message above. With `numba==0.67.0` and `llvmlite==0.49.0` all three runs passed.
   - The docs job first runs pytest, which populates the cache, and then runs the tutorials in fresh processes. That is exactly the failure reported for `examples/tutorials/09-photon-number.py`.
   - The same crash made `tests/native/test_fock_benchmark_smoke.py`, which times Piquasso in subprocesses, fail on Python 3.11, 3.13 and 3.14 in the local matrix below.
   - No upstream issue was found. The evidence is the reproduction above.
3. **Test environment shadowing (packaging defect, found by the version matrix).**
   - `[tool.pytest.ini_options] pythonpath = ["src"]` made pytest import the source tree instead of the installed package.
   - With a regular (non-editable) wheel install the compiled `photographiq._native` is not in `src/`, so native tests could not run against wheels.
   - CI uses an editable install, which masked the problem.

## Fixes implemented

| Fix | Files |
| --- | --- |
| Notebook lint and format: imports restructured; an unused Piquasso import replaced by `importlib.util.find_spec`; semicolon statements split; ruff-formatted | the three notebooks above |
| `numba<0.68` added to every extra that installs Piquasso (`piquasso`, `ml-experiments`, `autodiff`, `dev`), with the reproduction recorded in a comment | `pyproject.toml` |
| Documentation examples moved off Piquasso's JIT wherever the native engine supports the same physics: 20 tutorials (09–16, 21–23, 38, 39, 41–45, 48, 49), 3 projects, 2 notebooks (04, 05) and the copied docs snippets. Every printed or asserted quantity was compared: equal to roundoff, except three examples where squeezing/CZ truncation differs by at most 3.3e-9 (within the examples' own assertions). | `examples/`, `notebooks/`, `docs/tutorials/`, `docs/user-guide/` |
| `cutoff_convergence` accepts `native-fock` and `native-mixed-fock`, and `measurement_convergence` takes `backend=`. Defaults are unchanged for reproducibility. Native and Piquasso results agree to roundoff (GKP study rows identical; cubic-phase study within 1e-14). | `src/photographiq/convergence.py`, `src/photographiq/gkp_studies.py` |
| Optional reference test comparing the migrated tutorials on Piquasso and native backends | `tests/piquasso_reference/test_tutorial_backends.py` |
| Native-only smoke test: subprocess with Piquasso, Graphix, JAX, mpmath, CuPy, MentPy and CUDA-Q blocked. It runs the native Gaussian, Fock and mixed-Fock paths and checks the "install the extra" errors. | `tests/unit/test_native_only_smoke.py` |
| Removed `pythonpath = ["src"]`, so tests import the installed package | `pyproject.toml` |
| Graphix benchmark gains `--out-dir`, so a reproduction run writes nothing into the repository | `benchmarks/graphix/bench_graphix_mbqc.py` |

With Piquasso made unimportable, `check_docs_examples.py --projects --notebooks` now passes
everything except `24-automatic-differentiation.py`. `photographiq.autodiff` builds states with
the Piquasso mixed-Fock backend, and its `autodiff` extra declares Piquasso.

## Local reproduction of the gates (Windows, Python 3.12)

| Gate | Result |
| --- | --- |
| `experiments/validate_release.py` (pytest with coverage, ruff check, ruff format, mypy, build, physics evidence) | all gates passed; 1273 passed, 14 skipped; coverage 93.90 % (floor 90 %) |
| `scripts/check_public_api.py` | 85 exports checked |
| `scripts/check_docs_examples.py --projects --notebooks` | all tutorials, 10 projects and 5 notebooks passed |
| `ruff check` and `ruff format --check` on `src tests examples experiments scripts` (after the fixes) | clean |

## Python-version matrix (fresh venv, non-editable `pip install .[dev,autodiff]`, full suite against the installed wheel)

| Python | Install | Native components | Full suite |
| --- | --- | --- | --- |
| 3.11.16 | OK (after a transient PyPI index error on the first attempt) | gaussian, fock, sampling | 1644 passed, 1 failed (Numba 0.68 crash), 61 skipped; 1645 passed after `numba<0.68` |
| 3.12.14 (project venv, editable) | OK | gaussian, fock, sampling | see the gate run above |
| 3.13.15 | OK | gaussian, fock, sampling | 1644 passed, 1 failed (Numba 0.68 crash), 61 skipped; 1645 passed after `numba<0.68` |
| 3.14.4 | OK | gaussian, fock, sampling | 1644 passed, 1 failed (Numba 0.68 crash), 61 skipped; 1645 passed after `numba<0.68` |

The 61 skips are optional packages that the `dev` extra does not install (CuPy, nbclient,
MentPy, CUDA-Q) plus documented capability gaps. NumPy resolves to 2.4.x on 3.11 and 2.5.x on
3.12 and later. The advertised Python range 3.11–3.14 is supported on Windows.

## Fresh installation

- **Built-wheel install, core-only.** NumPy, SciPy and NetworkX only. Verified in the previous upgrade session; the current tree is re-verified in Phase 5.
  - Backends `gaussian`, `native-gaussian`, `native-fock` and `native-mixed-fock`.
  - Missing extras give `ImportError` messages naming `photographiq[piquasso]`, `photographiq[precision]` or the CuPy wheel.
- **Source distribution.** Built by the gate (`build`); its contents are inspected in Phase 5.
- **Editable install.** The project venv (rebuild command in `docs/development/native-build.md`).

## Unresolved blockers

- **Remote CI not re-run.** The fixes are local and unstaged, and pushing is outside this assignment's permissions. Linux behaviour is inferred from the Windows reproduction of the Numba crash and from the deterministic lint failures. A push, made by the maintainer, is required to confirm green CI on ubuntu.
- **No Linux environment locally** (no WSL, no Docker).
- **Numba upstream.** The `numba<0.68` constraint should be lifted once a Numba release fixes cached-library reloading; the reproduction steps are in the `pyproject.toml` comment.
- **Tutorial 24** needs Piquasso, by design of `photographiq.autodiff`.
