# PhotoGraphiQ manuscript experiment suite

A reproducible experiment suite (R1-R42) auditing the `photographiq` 0.3.1
API as it actually exists on this branch (`manuscript-experiments`), built
for a QUANTUM-style software manuscript. Every script was written against
real source in `../src/photographiq` after reading it, verified by running
it, and distinguishes exact functionality from finite-resource/finite-Fock
approximations. See `EXPERIMENT_INDEX.md` for the full catalogue,
`STATUS.md` for the latest per-experiment run record, and `ISSUES_FOUND.md`
for genuine software observations made while building this suite.

**Single-notebook version**: `PhotoGraphiQ_Manuscript_Experiments.ipynb`
combines `common.py`, `metadata.py` and every R1-R42 script into one
executable notebook, with a leading cell that installs every dependency
(including `photographiq` itself, editable, from the local checkout) via
`pip` (bootstrapping `pip` itself via `ensurepip` first if the interpreter
lacks it, e.g. a bare `uv`-provisioned environment). It is generated from
the very same script files by `build_notebook.py` -- regenerate it after
editing any script with `python build_notebook.py`. The checked-in copy
already contains a verified full run's outputs; running it fresh takes
roughly 30-45 minutes (see the notebook's own first cell for a runtime
breakdown). `run_all_safe.py` remains the right choice for a fast,
subprocess-isolated lightweight run; the notebook is for a single linear,
shareable, already-executed narrative of the entire suite.

## 1. Purpose

Generate independent, machine-readable numerical evidence and
publication-ready figures for a manuscript about PhotoGraphiQ: core CV-MBQC
transport, circuit-to-MBQC compilation, adaptive feed-forward, Gaussian and
non-Gaussian simulation, GKP resources, validation against raw Piquasso /
independent NumPy references / Graphix structure, convergence, and
performance. No conclusions here are hard-coded -- every number is computed
from the current codebase by the scripts in this directory.

## 2. Environment setup

From the repository root:

```sh
python -m pip install -e '.[dev]'          # core + graphix + matplotlib
python -m pip install 'jax>=0.4.35'        # optional, for R41/R42
```

See `requirements_notes.md` for exact versions this suite was validated
against and what happens when an optional dependency is absent.

## 3. Optional dependencies

- `graphix==0.4` -- only R17. Skips cleanly (prints `SKIPPED`, exit 0) if absent.
- `jax>=0.4.35` -- only R41, R42. Same skip behavior.

## 4. How to run the lightweight suite

```sh
cd paper_experiments
python run_all_safe.py
```

This runs the ~24 scripts that are fast, dependency-free and use modest
Fock cutoffs, and explicitly SKIPS (with a stated reason) everything
tagged "optional dependency", "large Fock", or "long benchmark". A JSON
summary is written to `results/json/run_all_safe_summary.json`.

## 5. How to run an individual experiment

Every script is directly executable and self-contained:

```sh
python 07_non_gaussian/19_photon_subtraction_probability.py
```

Each script prints a short summary, saves CSV+JSON+a metadata companion
under `results/`, and (where applicable) a PDF+PNG figure pair under
`figures/`.

## 6. Expensive experiments

R18, R20-R23, R26-R28, R33-R34 use non-Gaussian Fock cutoffs from ~14 up to
96 and can take from under a minute to several minutes. R28 in particular
runs 16 seeds x 9 displacements at cutoff=80 (~9 minutes on the reference
machine) to get statistically meaningful ensemble syndrome statistics.

## 7. Experiments requiring Graphix

R17 only. `run_all_safe.py` always skips it regardless of availability
(policy: skip everything tagged "optional dependency"); run it directly if
you have `graphix==0.4` installed.

## 8. Experiments requiring JAX

R41, R42. Same skip policy as Graphix; run directly with `jax` installed
and `jax_enable_x64` enabled (the scripts do this themselves).

## 9. Experiments requiring large Fock cutoffs

R26-R28 (GKP) go up to cutoff 80-96 because two GKP-scale modes share one
total-photon cutoff; R33-R34 sweep up to cutoff 96 for convergence studies.
See `docs/performance.md` and R35 for the general dimension/memory formula
before increasing any cutoff further, and `ISSUES_FOUND.md` #2 for an
observed native-gate numerical-stability limit encountered while sizing R28.

## 10. Where outputs are stored

```
results/csv/      -- one CSV per experiment (machine-readable rows)
results/json/      -- JSON mirror + extra structured fields + *.metadata.json
results/raw/       -- serialized patterns, provenance text summaries
results/logs/       -- reserved for future log capture (run_all_safe currently
                       captures subprocess stdout/stderr inline instead)
figures/pdf/       -- vector figures
figures/png/       -- 300 dpi raster figures
tables/            -- aggregated manuscript tables (generate_tables.py), csv + md
```

## 11. Exact vs. approximate results

Every script's docstring states which category it falls into:

- **Exact functionality**: R1-R18, R24, R29-R31, R35, R39-R40 compare against
  analytic formulas or independent code paths and pass at (near-)machine
  precision.
- **Finite-resource approximation**: R2-R10, R23, R25, R29, R34 (MBQC
  compilation/injection always carries physical finite-squeezing noise; this
  is documented physics, not error).
- **Fock truncation approximation**: R18 (mixed-Fock cross-check), R20-R23,
  R26-R28, R33-R34 (any `piquasso-fock`/`piquasso-mixed-fock` execution).
- **Experimental functionality** (per `docs/validation/feature-matrix.md`):
  mixed Fock, cubic/Kerr synthesis, GKP, JAX autodiff/estimators (R18,
  R23, R25-R28, R32, R41-R42).
- **Unsupported / not attempted**: see section 13.

## 12. Reproducibility notes

- Every stochastic experiment fixes a seed (or explicit `measurement_outcomes`
  for Fock postselection) and records it.
- `metadata.py` records git commit/branch, Python/package versions, platform
  and CPU model; every experiment writes a `*.metadata.json` companion.
- Timing experiments (R25, R36-R38) use warm-up runs plus repeated
  `time.perf_counter` measurements and report median + IQR/std, never a
  single untrusted sample; they explicitly avoid timing import/setup cost.
- Fock experiments record cutoff, dimension, retained norm and boundary
  population wherever the backend exposes them, and use
  `common.safe_cutoff_run` to catch (not hide) expected `MemoryError`/
  norm-truncation `ValueError`s rather than disabling the safety checks.

## 13. Known unsupported experiments

None of R1-R42 were found unsupported by the current API -- every requested
experiment had a corresponding real code path once the actual source was
read (see the design notes embedded in each script's docstring for how each
was matched to the actual public/semi-public API surface, e.g. R25 executing
`synthesize_kerr`'s output directly as physical commands to isolate
synthesis error from MBQC injection noise, or R28's redesign around
ensemble/circular statistics once single-shot noise was understood).
Two minor validator-message dead-code paths and one native-gate numerical
stability limit were found and are documented, not worked around, in
`ISSUES_FOUND.md`.

## 14. Mapping from experiment IDs to manuscript figures/tables

See `MANUSCRIPT_MAP.md`.
