# Legacy reference outputs (PhotoGraphiQML suite, commit 441de9a)

This directory freezes the PhotoGraphiQML manuscript-experiment evidence exactly as it
existed in the unified repository at commit **441de9a8811cefd676271efa0b65985d041666a8**
(branch `upgraded`, "modified 2"), BEFORE the suite was re-executed against the upgraded
software (native C++ `native-fock` engine, `photographiq.flows`, IR/compiler, sampling).
Do not modify anything in this directory.

## Contents

| Path | What |
|---|---|
| `results/` | byte copy of `paper_experiments_ml/results/` (csv, json, raw, logs, validation; 372 files) |
| `tables/` | byte copy of `paper_experiments_ml/tables/` (16 files) |
| `STATUS.md`, `EXPERIMENT_MANIFEST.json`, `EXPERIMENT_CONTRACTS.json`, `EXPERIMENT_INDEX.md` | byte copies of the suite records at 441de9a |
| `figures_manifest.json` | path, SHA-256 and size of the 177 files under `paper_experiments_ml/figures/` (pdf, png, svg) and of the executed notebook `PhotoGraphiQML_Manuscript_Experiments.ipynb`; these were NOT copied (13 MB) |
| `MANIFEST.sha256` | SHA-256 of every file in this directory except itself (`sha256sum -c MANIFEST.sha256` from here) |

Line endings: the copies are of the Windows working tree (CRLF text files where git
normalises to LF). Every copied file was checked to be identical to
`git show 441de9a:paper_experiments_ml/<path>` after CRLF->LF normalisation (392/392), and
every entry of `figures_manifest.json` matches the committed blob (178/178; PNG/PDF
byte-identical, SVG/ipynb identical modulo CRLF).

## Provenance recorded by the files themselves

All 59 `results/json/*.metadata.json` record: source commit
`ea56bf364c71b7f6349a9fb8eb3171a9c4a96058` of the **standalone** PhotoGraphiQML repository
(branch `manuscript-experiments`, `source_dirty: false`), Python 3.14.4,
Windows-11-10.0.26200-SP0, photographiq 0.3.1 (sibling checkout at `6d49da0`),
mentpy 0.1.0a15, piquasso 8.0.1, numpy 2.5.3, scipy 1.18.1. Accepted run
`ea56bf364c71-20260924T235941Z` (see `STATUS.md`). Physical experiments used the
Piquasso-backed `piquasso-fock` backend only; nothing here is a result of PhotoGraphiQ's
native C++ `native-fock` engine.

## Retrieving the historical figures

```sh
git show 441de9a:paper_experiments_ml/figures/png/R38_public_pattern_comparison.png > R38.png
```

and check it against `figures_manifest.json` (`sha256sum`).
