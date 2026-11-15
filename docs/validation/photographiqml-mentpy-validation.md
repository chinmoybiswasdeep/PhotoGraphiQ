# PhotoGraphiQML: MentPy reference validation (Windows, isolated environment)

Date: 2026-11-11/12 (local run). Repository HEAD: `f30321fd09140b310c3fb53e02a9b503e756855f`
(branch `upgraded`). The working tree had uncommitted changes from parallel work in
`src/photographiq/`, `cpp/` and `pyproject.toml` during this run; `src/photographiqml/`,
`tests/ml/` and `paper_experiments_ml/` had none. Package builds therefore reflect the
working tree at build time, not HEAD exactly (see "Builds").

The independent logical reference is **MentPy itself**, installed from the exact commit
the ml-reference workflow uses. No self-reference was substituted for it.

## Environment

| Item | Value |
| --- | --- |
| OS | Windows 11 Home 10.0.26200, x86-64 |
| Python | CPython 3.12.14 (uv-managed), same minor version as the CI job |
| venv | isolated, outside the repository (scratch directory); project `.venv` untouched |
| photographiq (provides `photographiq` and `photographiqml`) | 0.3.1, built non-editable from the source tree, native extension `_native.cp312-win_amd64.pyd` (MSVC, Eigen 3.4.0 from a local source dir) |
| MentPy | 0.1.0a15 from `git+https://github.com/mentpy/mentpy.git@63c3d83e495696b4491c9d376dab7e3e6cf6c863` (pip `direct_url.json` confirms the commit) |
| piquasso | 8.0.1 |
| numpy / scipy / networkx | 2.5.4 / 1.18.1 / 3.7 |
| PennyLane / galois | 0.45.1 / 0.4.11 (MentPy dependencies) |
| numba / llvmlite | **0.67.0 / 0.49.0**. The original resolve picked 0.68.1 / 0.50.0, which crashed (see Constraints); 0.67.0 matches the `numba<0.68` constraint `pyproject.toml` has since added |
| pytest | 9.2.0 |
| matplotlib / scikit-learn / graphix | 3.11.2 / 1.9.1 / 0.4 |

### Builds

- **Build A** (23:40 local): built before the `cpp/src/fock/gates.cpp` single-mode
  squeeze+displace stability change (file modified 23:48).
- **Build B** (00:17 local): incremental non-editable reinstall
  (`pip install --no-deps --force-reinstall .`) in the same isolated venv, after that change.

Results below say which build produced them. Build B reproduced every build-A result that
it re-ran **bit-for-bit**: the full `tests/ml`, R35-R38, R41, the native-fock variants and
the direct native comparison. The gates.cpp change does not alter any quantity measured here.

## Commands

All commands ran in the isolated venv (`$VENV`), never the project `.venv`.

```sh
# 1. venv + install: the ml-reference workflow steps, except non-editable
python3.12 -m venv $VENV
$VENV/Scripts/python -m pip install "./[dev]" -r tests/ml/mentpy_reference/requirements.txt \
    -Ccmake.define.FETCHCONTENT_SOURCE_DIR_EIGEN=<scratch>/deps/eigen-3.4.0 -Cbuild-dir=<scratch>/build
$VENV/Scripts/python -c "import mentpy"            # workflow guard step: passes
$VENV/Scripts/python -m pip install numba==0.67.0 llvmlite==0.49.0   # see Constraints

# 2. tests, from the repository root, forcing the installed package
python -m pytest -o pythonpath= -p no:cacheprovider tests/ml/mentpy_reference -v
python -m pytest -o pythonpath= -p no:cacheprovider tests/ml -q -rs

# 3. experiments, from a scratch copy of paper_experiments_ml/ (outside the repository)
cd <scratch>/run/paper_experiments_ml && MPLBACKEND=Agg $VENV/Scripts/python <group>/<script>.py
```

The workflow uses `pip install -e`. Here the install was non-editable on purpose, so the
tests run against the built wheel.

### Proof that the installed package is imported

- At the start of this run, the pytest config set `pythonpath = ["src"]`. With that, a plain
  `pytest tests/ml/mentpy_reference` imported `src/photographiq*` (verified: 34 passed
  against the source tree, without the native extension). `-p no:python_path` did **not**
  override this; `-o pythonpath=` did. (`pyproject.toml` has since dropped the setting.)
- A scratch pytest plugin printed `sys.modules[...].__file__` at session end. Every run
  reported here shows `photographiq`, `photographiqml`, `photographiq._native`, `mentpy` and
  `piquasso` resolving under `$VENV/Lib/site-packages`.
- Every experiment `*.metadata.json` from the scratch run records `package_import_origins`
  under `$VENV/Lib/site-packages`. The scratch copy has no sibling `src/`, so the
  `common.py` `sys.path.insert(<parent>/src)` line has no effect.

### Scratch-copy provenance change (copy only)

`publication.scientifically_relevant_files()` runs `git ls-files ... check=True` relative to
the parent of `paper_experiments_ml/`. Outside a git work tree that raises, so in the
**scratch copy only** the provenance step was disabled: `check=True` was dropped and the
function returns `[]` on a non-zero git exit. Consequences: scratch metadata has
`git_commit: null` and a fingerprint over an empty file list. The repository's
`publication.py` and every recorded file under `paper_experiments_ml/results/` are
unchanged. The `sha256` values of all 121 recorded `results/json/*.json` files were captured
before the runs.

## Test results

| Suite | Build | Result |
| --- | --- | --- |
| `tests/ml/mentpy_reference` (source tree, old `pythonpath=src`) | n/a | 34 passed |
| `tests/ml/mentpy_reference` (installed) | A | **34 passed**, 0 skipped |
| `tests/ml` (installed), numba 0.68.1 | A | **aborted at collection**: `Fatal Python error: Aborted` / `LLVM ERROR: Symbol not found` (see Constraints) |
| `tests/ml` (installed), numba 0.67.0 | A | **154 passed**, 0 skipped, 175 warnings, 188 s |
| `tests/ml` (installed; plain `pytest`, the new `pyproject.toml` has no `pythonpath`) | B | **154 passed**, 0 skipped, 173 warnings, 110 s; includes the 34 MentPy tests |

The warnings are expected: Piquasso `GaussianTransform` retained-norm `RuntimeWarning`s at
small cutoffs, plus `DeprecationWarning`s for the legacy `representation="gkp"`.

## Experiment results (scratch copy; build A, physical rows re-run on build B)

"Agreement" compares every JSON leaf of the new result with the recorded
`paper_experiments_ml/results/json/<stem>.json`. The recorded results came from Python
3.14.4, MentPy 0.1.0a15 at the same commit, and source commit `ea56bf36`. Timing leaves are
excluded.

| ID | Script | rc | Status | Key numbers (new run) | Agreement with recorded |
| --- | --- | --- | --- | --- | --- |
| R7 | 02/07_semantic_topology | 0 | pass | max symmetric difference 0 (24 configs) | 275/275 leaves identical |
| R8 | 02/08_flow_dependency_order | 0 | pass | order violations 0, correction mismatches 0 | 97/97 identical |
| R9 | 02/09_state_density_agreement | 0 | pass | max DM error 3.44e-15 (tol 2.22e-13) | 254/254 identical |
| R10 | 02/10_trainability_audit | 0 | pass | PhotoGraphiQML robust 8/8; MentPy restriction effective 0/8 (descriptive) | 93/93 identical |
| R11 | 02/11_kernel_gram_agreement | 0 | pass | state err 1.70e-15, Gram err 1.91e-14 | 174/174 identical |
| R12 | 02/12_instrument_agreement | 0 | pass | prob err 1.67e-15, cond-state 1.33e-15, trace 2.44e-15 | 87/87 identical |
| R13 | 02/13_runtime_scaling | 0 | pass (descriptive) | median run time, PhotoGraphiQML vs MentPy: 1 wire 6.7e-5 s vs 7.7e-4 s; 2 wires 1.4e-4 s vs 2.6e-3 s; 3 wires 2.0e-4 s vs 1.5e-2 s | 23/23 non-timing leaves identical; timings same order of magnitude |
| R20 | 04/20_haar_gate_learning | 0 | pass | median test infidelity 2.28e-6 (95% CI 1.09e-6 to 3.66e-6); max MentPy checkpoint err 1.78e-15 | 123/123 non-timing leaves identical |
| R21 | 04/21_ising_gate_learning | 0 | pass | median test infidelity 2.83e-6 (CI 1.50e-6 to 3.78e-6); max MentPy checkpoint err 1.55e-15 | 123/123 non-timing identical |
| R1 | 01/01_triangle_anatomy | 0 | pass | MuTA logical | 973/973 identical |
| R2 | 01/02_table_one_identities | 0 | pass | MuTA logical | 413/413 identical |
| R3 | 01/03_entangling_identity | 0 | pass | MuTA logical | 318/318 identical |
| R5 | 01/05_brute_force_agreement | 0 | pass | MuTA logical | 205/205 identical |
| R18 | 04/18_parameter_shift_vs_fd | 0 | pass | worst min FD error 1.33e-10 (tol 2.22e-8), 75 trials | 237/237 identical |
| R19 | 04/19_optimizer_convergence | 0 | pass | training | 59/59 identical |
| R23 | 04/23_classifier_verification | 0 | pass | training | 56/56 identical |
| R25 | 05/25_kernel_feature_state | 0 | pass | kernels | 52/52 identical |
| R26 | 05/26_kernel_properties | 0 | pass | kernels | 50/50 identical |
| R27 | 05/27_kernel_classification | 0 | pass | kernels | 257/257 identical |
| R35 | 08/35_capability_map | 0 | pass | physical capability audit | 102/102 identical (one `NaN` angle leaf compares unequal only because NaN != NaN) |
| R36 | 08/36_lowering_preservation | 0 | pass | physical GKP lowering, cutoff 16 | 49/49 identical |
| R37 | 08/37_pauli_frame_validation | 0 | pass | Pauli frames | 22/22 identical |
| R38 | 08/38_public_pattern_comparison | 0 | pass | piquasso-fock, cutoff 24: max err 0.0 vs independent Pattern | 29/29 identical |
| R41 | 09/41_conditional_execution_comparison | 0 | pass | piquasso-fock, 8 conditioning cases: max err 0.0 | 52/52 identical |

On build B, R35, R36, R37, R38 and R41 were re-run: rc 0, status pass, and the same
identical-leaf counts as above.

Summary: 24 scripts were run, 24 exited 0 and 24 have status `pass`. Every non-timing
result leaf is bit-identical to the recorded evidence, even though the Python minor
version, numpy patch version and machine run differ.

## native-fock vs piquasso-fock

Two checks, each run in the isolated venv on build A and again on build B, with identical
numbers on both builds. The timings below are from build B.

**(a) Manuscript scripts with `GKPPhysicalConfig(..., backend="native-fock")`.** In a
second scratch copy, R38 and R41 set `backend="native-fock"`. That backend then drives
both the PhysicalMuTA lowering and the independently assembled PhotoGraphiQ Pattern.

| Script | native-fock status | native max err vs independent native Pattern | max abs(native - recorded piquasso) decoded probability |
| --- | --- | --- | --- |
| R38 (cutoff 24) | pass | 0.0 (tol 2.22e-10) | 4.62e-5 |
| R41 (cutoff 24, 8 cases) | pass | 0.0 (tol 2.22e-10) | 4.62e-5 |

**(b) Direct PhysicalMuTA comparison.** The scratch script `native_vs_piquasso.py` used
peak width 0.9, envelope 0.9, 4 peaks and 1025 grid points, with random
0/π parameters and random analog outcomes.

| Case | Cutoff | Max abs(dP) | Max infidelity | Bits / samples identical | Time, piquasso vs native |
| --- | --- | --- | --- | --- | --- |
| 1 wire, conditional, seeds 0-2, Z and X | 24 | 5.1e-6 to 1.52e-4 | <= 1.6e-6 | yes, all | about 0.47 s vs 0.26 s (first call 1.9 s, warm-up) |
| 1 wire, conditional, seeds 0-2, Z and X | 32 | 1.2e-6 to 7.9e-5 | <= 1.3e-7 | yes, all | about 0.84 s vs 0.49 s |
| 1 wire, shots (4), seed 0 | 24 | both engines refuse with "increase cutoff": truncation norm 0.99891337 (piquasso), 0.99891343 (native) | - | - | - |
| 1 wire, shots (4), seed 0 | 32 | 2.3e-5 | 6.9e-8 | yes | 6.2 s vs 4.7 s |
| 2 wires, conditional, seed 2 | 32 | 9.0e-5 | 1.5e-7 | yes | 53.8 s vs 0.78 s |

The two engines differ only in CZ truncation: native applies the exact projected P U P,
while Piquasso truncates per Euler factor. The differences match the truncation-error
scale documented in `tests/ml/cross_layer/test_native_backend.py` (cutoff 32: 7.6e-6 /
7.9e-5 / 9.0e-5) and shrink as the cutoff rises (seed 1 Z: 1.5e-4 at cutoff 24, 1.2e-6 at
cutoff 32). Decoded bits and sampled outputs are identical in every case. At cutoff 24,
both engines raise the same truncation guard on the shots trajectory, which is consistent
behavior and not a defect.

## Constraints and incompatibilities (precise)

1. **MentPy on Windows / Python 3.12: no incompatibility.** The pinned commit installs from
   git and builds a pure-Python wheel (`mentpy-0.1.0a15-py3-none-any.whl`). The
   `import mentpy` guard step passes and all MentPy tests pass. Falling back to PyPI was not
   needed.
2. **The `ml-validation` extra (`mentpy==0.1.0a15` from PyPI) and the workflow commit
   `63c3d83e` are code-equivalent.** The PyPI wheel was downloaded but not installed, then
   diffed against the git build: the same 43 `.py` files. The only differences are CRLF/LF
   line endings and one trailing space in a docstring in `mbqc/templates.py`. Dependency
   metadata is identical except for formatting.
3. **numba 0.68.1 / llvmlite 0.50.0 crash when piquasso 8.0.1 is imported before galois
   0.4.11 and the numba on-disk cache is warm.** Minimal reproduction, run twice with the
   same `NUMBA_CACHE_DIR`: `python -c "import piquasso; import galois"`. The first run passes.
   The second exits 127 with
   `LLVM ERROR: Symbol not found: .numba.unresolved$_ZN8piquasso5_math7hafnian10hessenberg...`.
   The order `import galois; import piquasso` does not fail. This kills a full `tests/ml`
   session, because `cross_layer` imports piquasso before `mentpy_reference` imports mentpy,
   which imports galois. It does not affect the CI job: that job runs only
   `tests/ml/mentpy_reference`, where piquasso is never imported, and the main test job has
   no MentPy installed. With numba 0.67.0 / llvmlite 0.49.0, the version the project
   `.venv` already uses, three warm-cache runs pass. This is a third-party
   numba/llvmlite cache problem, not a PhotoGraphiQML defect. Unpinned resolution on
   2026-11-11 selected numba 0.68.1. `pyproject.toml` now constrains `numba<0.68` in the
   `piquasso`, `ml-experiments`, `autodiff` and `dev` extras, so the workflow's
   `pip install -e '.[dev]' -r .../requirements.txt` now resolves a passing numba. This venv
   was installed before that constraint and was downgraded by hand to the same versions.
4. The recorded `package_versions()` reports `photographiqml: null` in this environment.
   `photographiqml` ships inside the `photographiq` distribution, and no distribution named
   `photographiqml` exists. The recorded evidence (`0.2.0`) came from the earlier
   standalone PhotoGraphiQML checkout. This is informational only.

## Defects found

None in `src/photographiqml/` or `tests/ml/`. No repository source files were changed.
All 121 recorded `paper_experiments_ml/results/json/*.json` files were verified unchanged by
`sha256` after the runs.
