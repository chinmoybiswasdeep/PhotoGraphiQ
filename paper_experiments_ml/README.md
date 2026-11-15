# PhotoGraphiQML manuscript experiment suite

Reproducible experiments (R1–R48, R_PERF, and R50–R59; 59 total)
supporting a QUANTUM-journal manuscript on PhotoGraphiQML: ideal logical
MuTA (validated against MentPy), the restricted signed-X physical GKP
bridge (validated against Piquasso and independently assembled
PhotoGraphiQ patterns), and the classical-ML wrappers built on top of both.

Mirrors the organization and visual style of the PhotoGraphiQ core
[manuscript experiment suite](../paper_experiments) in the same repository
(`common.py`/`metadata.py` utility surface, rcParams, PDF+PNG figure
policy, CSV+JSON+metadata "save triple") while testing entirely different
scientific content, and adds oracle-classification (A–E, N/A), bootstrap
confidence intervals, and declared-in-advance tolerances that this suite's
broader physical/statistical scope requires beyond what PhotoGraphiQ's
core suite needed.

## Layout

```
paper_experiments_ml/
├── 01_logical_foundations/      R1-R6   ideal logical MuTA, analytic/independent oracles
├── 02_mentpy_validation/        R7-R13  independent MentPy cross-checks
├── 03_parameters_and_contracts/ R14-R17 parameter/persistence/fail-fast/reproducibility contracts
├── 04_gradients_and_training/   R18-R24 gradients, optimizers, gate learning, classifier/regressor
├── 05_kernels_and_learning/     R25-R28 Eq. 5 kernel properties and classification
├── 06_expressivity_and_diagnostics/ R29-R33 instruments, concurrence, QFI, Lie closure, Fisher spectra
├── 07_gkp_resources/            R34     finite-GKP codeword diagnostics
├── 08_physical_lowering/        R35-R38 capability audit, lowering preservation, Pauli frames
├── 09_piquasso_validation/      R39-R42 raw Piquasso / independent Pattern validation
├── 10_physical_statistics/      R43-R46, R48  decoded statistics, shot convergence, decoding, training stability
├── 11_convergence/              R47     resource-axis convergence studies
├── 12_performance/              aggregate performance figure (reproduced from saved results)
├── 13_extended/                 R50-R59 reproduction, ML, physical, and scaling studies
├── results/{csv,json,raw,logs}/ machine-readable outputs
├── figures/{pdf,png,svg}/       publication figures
├── tables/                      generated CSV+Markdown summary tables
├── common.py, metadata.py       shared utilities (see their own docstrings)
├── EXPERIMENT_INDEX.md          one row per experiment: question, oracle class, acceptance, status
├── EXPERIMENT_CONTRACTS.json    canonical machine-readable experiment contracts
├── STATUS.md                    latest real execution record
├── ISSUES_FOUND.md              genuine findings (not silently patched)
├── MANUSCRIPT_MAP.md            proposed figure/table/claim mapping + anti-claims
├── run_all_safe.py              light/deterministic subset, subprocess-isolated
├── run_all_full.py              complete suite, resumable, filterable
├── verify_evidence.py           independent schema/provenance/hash verifier
├── generate_tables.py           CSV+Markdown summary tables from saved results
├── build_notebook.py            assembles one notebook from these scripts
└── PhotoGraphiQML_Manuscript_Experiments.ipynb
```

## Running

```bash
# from the repository root, using the project's .venv
.venv/Scripts/python.exe paper_experiments_ml/run_all_safe.py     # light/deterministic subset
.venv/Scripts/python.exe paper_experiments_ml/run_all_full.py     # complete suite (heavy; physical shots take minutes)
.venv/Scripts/python.exe paper_experiments_ml/generate_tables.py  # rebuild tables/ from results/
.venv/Scripts/python.exe paper_experiments_ml/build_notebook.py   # rebuild the notebook
```

The generated manuscript notebook is also directly runnable in Google Colab. It embeds every
canonical experiment, places its mathematical contract and algorithm immediately before the code,
and follows each run with an interactive data atlas. The atlas covers every CSV cell and every
result, metadata, and raw-JSON leaf, using complete-cell heatmaps plus adaptive relationship views
(including trajectories, scatter matrices, radar profiles, violins, ECDFs, and parallel
coordinates) rather than relying only on bar charts or histograms.

Or run any single script directly, e.g.
`.venv/Scripts/python.exe paper_experiments_ml/01_logical_foundations/01_triangle_anatomy.py`.
Every script is self-contained (imports only `common`/`metadata` plus public
`photographiqml`/`photographiq`/`mentpy`/`piquasso` APIs) and can be re-run
independently.

## Conventions

- **Oracle classes** (A–E, N/A): declared in every script's docstring, its
  saved JSON, `EXPERIMENT_INDEX.md`, and figure caption/title. See that
  file's header for the class definitions.
- **Tolerances** for exact comparisons are computed by
  `common.declare_tolerance(...)` *before* the acceptance check runs, and
  recorded in the saved JSON — never chosen after inspecting results.
- **Multi-seed studies** use >=20 seeds for logical gate-learning claims
  (R20, R21), report individual trajectories plus median and bootstrap 95%
  CIs (`common.bootstrap_ci(..., statistic=np.median)`), and never
  cherry-pick a "representative" seed. The statistic is mandatory and is
  persisted in each CI object.
- Every experiment imports this checkout's `src/` before the public API, so
  an unrelated installed package with the same name cannot silently supply
  the implementation under test.
- **Physical/shot statistics** distinguish conditional POVM probabilities
  (Rao–Blackwell estimator) from empirical sampled-bit frequencies, each
  with its own standard error, per `docs/ml/physical/execution-modes.md`.
- Every experiment prints a short greppable summary; full evidence is in
  its saved CSV/JSON/metadata/figures, never only in console output.
- A script's pass/fail is computed from its saved metrics against its
  declared acceptance condition — never hard-coded.

See `requirements_notes.md` for the environment this suite was executed in.

## Provenance in the unified repository

This suite was moved byte-for-byte from the standalone PhotoGraphiQML repository into the
unified PhotoGraphiQ repository. The recorded evidence (`results/`, `figures/`, `tables/`,
`EXPERIMENT_MANIFEST.json`, `EXPERIMENT_CONTRACTS.json`, `STATUS.md` and the executed notebook)
was produced there, not in this tree:

- Standalone repository state at the move: commit
  `5e96e335c33dc52fe0179a949d96a4a3662fcdec`, branch `manuscript-experiments`.
- Accepted publication run: source commit `ea56bf364c71b7f6349a9fb8eb3171a9c4a96058`
  (run `ea56bf364c71-20260924T235941Z`), evidence recorded in commits
  `7be58767780c2989ad32a14f6f1582893fda01fb` and `bd9834213ef2eae4091559fdecd9a861b19d788b`.
  That evidence is verifiable only at those commits of the standalone repository.

Artifact paths in the manifest are relative to this directory, so every artifact `sha256` still
matches here. Source fingerprints, however, are keyed by repository-relative paths, so the
recorded `source_tree_fingerprint` and per-experiment fingerprints will not match in the unified
tree (they were already stale at `5e96e33`) until a fresh `run_all_full.py --publication` run.

In the unified repository the fingerprint and the source-dirty check cover
`publication.SOURCE_PATHS`: the PhotoGraphiQML-owned trees (`src/photographiqml`, `tests/ml`,
`paper_experiments_ml` minus its generated outputs, `scripts/ml`, `experiments/ml`,
`examples/ml`, `docs/ml`, `notebooks/ml`), the core package `src/photographiq`, `pyproject.toml`
and `.gitattributes`. PhotoGraphiQ's own suite and other repository files are excluded, so core
experiment runs do not invalidate this suite. `photographiq_commit` is the repository `HEAD` and
`photographiq_dirty` is the porcelain status of `src/photographiq` only; the field names are
unchanged, so the schema-v3 records stay compatible.

## Re-execution after the backend upgrade

The unified repository later gained compiled C++ engines (`native-gaussian`, `native-fock`,
sampling), `photographiq.flows` and the IR/compiler. The evidence recorded before that
upgrade is frozen in `legacy_reference/441de9a/` (byte copies of `results/` and `tables/`,
SHA-256 manifests for figures and the notebook, retrievable with
`git show 441de9a:paper_experiments_ml/<path>`). `STATUS.md` records the re-execution
against the upgraded tree and its comparison with that frozen evidence.

- **MentPy.** R7-R13, R20 and R21 import MentPy. The project `.venv` has no MentPy, so
  there these scripts stop with `ModuleNotFoundError: No module named 'mentpy'`. That is
  an unavailable reference, not a scientific failure. Run them in an environment with
  `tests/ml/mentpy_reference/requirements.txt` installed.
- **Backends.** Every physical experiment in this suite still uses `piquasso-fock` (the
  `GKPPhysicalConfig` default), so recorded physical numbers stay comparable.
  `native-fock` is the second validated engine; the comparison of the two on the
  PhotoGraphiQML physical path is core manuscript experiment PGQ-R59
  (`paper_experiments/20_photographiqml/59_photographiqml_physical.py`) and
  `tests/ml/cross_layer/test_native_backend.py`. See also `docs/ml/guide.md`.
- **Finalizer gates.** An unfiltered `run_all_full.py` ends with gates that rebuild the
  package (including the C++ extension) twice: `python -m build` inside
  `scripts/ml/validate_release.py`, and `pip install . --target`. `run_all_full.py --filter <ID>`
  runs one experiment without those gates.

Experiment IDs are suite-qualified. The PhotoGraphiQ core suite in `paper_experiments/` uses
R1–R42; this PhotoGraphiQML suite uses R1–R48, R_PERF and R50–R59. Cite them as, for example,
PGQ-R17 and PGML-R17.
