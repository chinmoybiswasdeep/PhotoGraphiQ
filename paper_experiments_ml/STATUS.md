# Suite execution status

## Latest execution: re-run against the upgraded tree (2026-11-13, NOT an accepted publication run)

- Repository HEAD `441de9a` (branch `upgraded`). Uncommitted working-tree changes were present
  (parallel work, including semantics-preserving refactors in `src/photographiq` and
  `src/photographiqml/physical.py` during the run), so every metadata file reports
  `source_dirty: true`. The source fingerprint changed while the runs were in progress; the
  59 results carry 10 distinct `source_tree_fingerprint` values.
- Environment: the isolated MentPy venv (CPython 3.12.14, mentpy 0.1.0a15 at `63c3d83e`,
  piquasso 8.0.1, numpy 2.5.4, scipy 1.18.1, networkx 3.7, matplotlib 3.11.2,
  scikit-learn 1.9.1). A `sitecustomize` shim made it import this working tree's `src/` and
  the project `.venv`'s compiled `photographiq._native` (components gaussian, fock,
  sampling). `package_import_origins` in every metadata file confirms `src/photographiq*`.
- Method: R1-R4 and R6-R8 ran through `run_all_full.py --filter <ID>`. The 37 light
  experiments of `run_all_safe.py` ran in that script. The other 22 ran through
  `run_all_full.py --filter <ID>`. R5 and R9 were rerun through `--filter` after a transient
  Windows process-launch failure (`git check-attr` exit 0xC0000142 under heavy machine load;
  not a code defect). The unfiltered finalizer gates were NOT run, because they rebuild the
  package with the C++ extension twice (`python -m build`, `pip install . --target`).
  `generate_tables.py`, `build_notebook.py` and `build_manifest.py` were run separately.
- Machine shared with other jobs: runtimes are provisional.

### Outcome

- **59 / 59 experiments: execution `completed`, structural status `pass`.** Scientific
  outcomes: 45 positive, 12 descriptive, 2 negative (R48 physical-training instability; R57
  fresh-shot physical selection not robust). This matches the accepted evidence.
- **Agreement with the archived evidence** (`legacy_reference/441de9a/`, produced with
  Python 3.14.4 at standalone commit `ea56bf36`): 10349 JSON result leaves were compared.
  10175 are bit-identical. 154 differ only in timing leaves. 19 differ only in PGML-R59
  tracemalloc peak bytes (14) and timing-derived log-log fits (5). One leaf is a `NaN` angle
  in R35, which compares unequal to itself. **No non-timing, non-memory scientific number
  changed**, including every physical (`piquasso-fock`) result: R17, R38, R40-R48, R56-R58.
- Regenerated tables differ from the archived tables only in `performance_summary`
  (timings). The notebook was rebuilt (245 cells, 59 experiments). The manifest was rebuilt
  (59 experiments, 17 derived artifacts).

### MentPy availability

| Environment | `run_all_safe.py` | MentPy experiments (R7-R13, R20, R21) |
| --- | --- | --- |
| project `.venv` (no MentPy) | ran 37, passed 28, failed 9, skipped 22 (644 s) | 9 failures, all `ModuleNotFoundError: No module named 'mentpy'`: **unavailable**, not scientific failures |
| MentPy venv + current-tree shim | ran 37, **passed 37**, skipped 22 (1532 s) | all pass; non-timing leaves bit-identical to the archive |

`tests/ml`: `.venv` 120 passed, 2 skipped (the two MentPy modules, 34 tests). MentPy venv
**154 passed** (includes `tests/ml/mentpy_reference` 34/34).

### Evidence verification

`verify_evidence.py --pre-status`: **invalid**, 127 errors. All of them are provenance
errors: 59 "metadata reports dirty scientific source", 34 "metadata source fingerprint
mismatch" and 34 "manifest fingerprint mismatch" (the fingerprint moved during the run).
There are no artifact-hash, schema, contract, seed, status or tolerance errors. The full
`verify_evidence.py` additionally rejects the old accepted publication summary
(`results/json/run_all_full_summary.json`, run `ea56bf364c71-20260924T235941Z`), whose
artifact hashes no longer match: 440 errors in total. An accepted record requires
`run_all_full.py --publication` on a clean, committed tree.

### Native backend (not part of this suite's evidence)

All physical experiments here use `piquasso-fock`. Scratch copies of R38, R41, R43 and R45
with `backend="native-fock"` pass their own checks (independent-Pattern error 0.0). Their
decoded quantities differ from the `piquasso-fock` values by at most 4.6e-5, 4.6e-5, 9.5e-5
and 1.2e-4 respectively (cutoff 24, width/envelope 0.9). The systematic comparison is
core experiment PGQ-R59 (`paper_experiments/NEW_EXPERIMENTS_R59.md`).

## Previously accepted publication evidence (archived)

Run `ea56bf364c71-20260924T235941Z`, source commit `ea56bf36` of the standalone
repository, 59/59 hash-resumed, accepted `True`. Its records are frozen in
`legacy_reference/441de9a/` (`STATUS.md`, results, tables, manifest, figure hashes).

## Capability boundary

Arbitrary-angle physical XY measurements, general physical MuTA, native CVMuTA,
loss/detector-noise models, soft adaptive flow decoding, and physical MuTA kernels remain
unsupported. No experiment is described as quantum advantage or fault tolerance.
