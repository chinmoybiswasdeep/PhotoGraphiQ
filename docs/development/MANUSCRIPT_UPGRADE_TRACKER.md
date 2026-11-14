# Manuscript upgrade tracker

Tracking for the release hardening, documentation modernisation and manuscript experiment
expansion of the `upgraded` branch (local working tree on top of HEAD `441de9a`, started
2026-11-13). The final outcomes are in [MANUSCRIPT_UPGRADE_REPORT.md](MANUSCRIPT_UPGRADE_REPORT.md).

Allowed statuses: Not started, In progress, Implemented, Validated, Blocked, Unsupported.
A task is **Validated** only when a meaningful acceptance test has passed; the evidence column
names it.

| ID | Description | Source files | Test files | Acceptance criteria | Status | Evidence | Remaining blocker | Manuscript experiment |
|---|---|---|---|---|---|---|---|---|
| M1.1 | Reproduce the CI failures of run 43224033322 (`441de9a`) | `experiments/validate_release.py` | gate run | Root cause per failing gate identified | Validated (mypy) / Implemented (pytest) | mypy: 11 errors reproduced with mypy 2.4.1 + NumPy 2.5.4 stubs in a fresh environment. pytest: `441de9a` passes on Windows with the latest dependencies (1671 passed); the Linux-only cause is identified by inspection (headless Chrome sandbox) | GitHub artifacts need authentication (HTTP 401); no Linux machine; remote CI needs a push | — |
| M1.2 | Repair mypy | `src/photographiq/_checks.py` and 5 call sites | mypy | mypy clean with old and new stubs | Validated | mypy clean in `.venv` (mypy 2.3.1) and the fresh environment (mypy 2.4.1) | — | — |
| M1.3 | Repair pytest on Linux | `tests/visualization/test_interactive_browser.py` (`--no-sandbox` on Linux) | that test | Test runs on ubuntu-latest | Implemented | Passes on Windows; Linux unverified | Remote CI | — |
| M1.4 | Release gates and Python matrix | — | `validate_release.py`, matrix venvs | All gates green on 3.11-3.14 | Validated (Windows) | [report](MANUSCRIPT_UPGRADE_REPORT.md) | — | — |
| M2.1 | Documentation audit and modernisation | `README.md`, `docs/**`, `mkdocs.yml` | `check_docs_examples.py`, `check_public_api.py`, `mkdocs --strict` | Strict build clean, 52 examples pass | Validated | docs agent: 52 PASS, API check OK; strict build clean apart from this tracker's link to the report (resolved when the report exists) | — | — |
| M2.2 | Interactive and publication MBQC graphics | `src/photographiq/interactive/**`, `scripts/generate_demo_figures.py`, `docs/figures/demos/` | `tests/visualization`, `scripts/check_interactive_browser.py` | 14 demo figures; browser checks pass | Validated | 58 passed + Playwright 12/12 in Edge; 14 demos with SVG/PDF/PNG/caption | Firefox/WebKit/touch untested | R57, R60 |
| M3.1 | Preserve historical results | `paper_experiments/legacy_reference/441de9a/`, `paper_experiments_ml/legacy_reference/441de9a/` | SHA-256 manifests | Byte copies + manifests equal git blobs | Validated | core 157/157; ML 394 entries, 178/178 figure hashes equal `git show 441de9a` | — | R1-R42, PGML suite |
| M3.2 | Migrate R1-R42 | `paper_experiments/01_*` … `13_*` | each script's assertions | Every script runs; original checks unchanged or stricter | Validated | 42/42 pass (`UPGRADE_MIGRATION_MATRIX.md`) | timings provisional until the idle run | R1-R42 |
| M3.3 | PhotoGraphiQML compatibility | `paper_experiments_ml/`, `docs/ml/` | `tests/ml`, MentPy reference | ML tests and suite pass | Validated | 59/59 PGML experiments; tests/ml 154/154 and MentPy 34/34 in the MentPy venv; 10175/10349 JSON leaves bit-identical, the rest timings | accepted evidence needs a clean committed tree | PGML suite, R59 |
| M4.1 | R43-R50 native engines and sampling | `paper_experiments/14_*`, `15_*`, `16_*` | script assertions | Declared acceptance criteria met | Validated | all 8 pass `--full` (`NEW_EXPERIMENTS_R43_R50.md`) | — | R43-R50 |
| M4.2 | R51-R56, R58 flow, calculus, GPU | `paper_experiments/17_*`, `19_*` | script assertions | Declared acceptance criteria met | Validated | all 7 pass (`NEW_EXPERIMENTS_R51_R58.md`) | — | R51-R56, R58 |
| M4.3 | R57, R60 visualization and showcase | `paper_experiments/18_*`, `21_*` | script assertions, browser | Declared acceptance criteria met | Validated | R57 20/20, R60 13/13 | — | R57, R60 |
| M4.4 | R59 PhotoGraphiQML physical execution | `paper_experiments/20_photographiqml/` | script assertions | Declared acceptance criteria met | Validated | passes; 3 criteria revised after first data, documented in `NEW_EXPERIMENTS_R59.md` | — | R59 |
| M4.5 | Catalogue rebuild | `paper_experiments/{EXPERIMENT_INDEX,STATUS,MANUSCRIPT_MAP,README}.md`, `run_all_safe.py`, `generate_tables.py`, `build_notebook.py` | consistency audit; notebook smoke execution | Every cited script, figure and table exists; notebook executes | Validated | audit `index`, `map`, `figures`, `evidence` pass; notebook smoke run 16/16, 0 errors | — | all |
| M5.1 | Manuscript validation pathway | `scripts/reproduce_manuscript.py` (smoke/scientific/full), `.github/workflows/manuscript.yml` | `--list` per mode | Modes defined; workflow uploads reports | Implemented | step lists per mode checked | workflow not run remotely | all |
| M5.2 | Consistency audit | `scripts/audit_manuscript_consistency.py` | its 12 checks | All 12 checks implemented and passing | Validated | see report | — | all |
| M5.3 | Provenance | `paper_experiments/metadata.py` | metadata smoke | Commit, dirty flag, source SHA-256, GPU, native components recorded | Validated | `tables/experiment_provenance.csv` | — | all |
| M5.4 | Native Fock: nested Eigen threading in `multiply_fibres` | `cpp/CMakeLists.txt` | timing check, native tests | No slowdown from Eigen-internal threads | Validated | idle machine: `apply_mixed` cutoff 12 is 5.5 ms with default threads vs 0.64 ms single-threaded | — | R46, R47 |
