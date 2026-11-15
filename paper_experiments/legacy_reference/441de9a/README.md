# Legacy reference outputs (historical suite, commit 441de9a)

This directory freezes the manuscript-experiment outputs (R1-R42) exactly as
they existed at source commit **441de9a8811cefd676271efa0b65985d041666a8**
(branch `upgraded`, commit message "modified 2"), BEFORE the experiments were
migrated to the upgraded software (native C++ backends, `photographiq.flows`,
IR/compiler, sampling). Do not modify anything in this directory.

## Contents

| Path | What |
|---|---|
| `results/` | byte copy of `paper_experiments/results/` from the working tree at 441de9a |
| `tables/` | byte copy of `paper_experiments/tables/` |
| `figures_manifest.json` | path, SHA-256 and size of each of the 92 files under `paper_experiments/figures/` (pdf + png); figures were NOT copied (15 MB) |
| `MANIFEST.sha256` | SHA-256 of every copied file plus `figures_manifest.json` (`sha256sum -c MANIFEST.sha256` from this directory) |

Line endings: the copies are of the Windows working tree, which has CRLF text
files (`core.autocrlf=true`); the git blobs at 441de9a are LF (`.gitattributes`
`eol=lf`). After CRLF->LF normalisation every copied file was checked to be
identical to `git show 441de9a:paper_experiments/<path>`; every figure's
SHA-256 in `figures_manifest.json` matches the committed blob byte-for-byte
(binary files).

## Provenance of these outputs (as recorded by the files themselves)

All 44 `results/json/*.metadata.json` files record the same environment:

- `git_commit`: `930408142cfd2dbc380976c28aa009d23c44babe`, `git_branch`:
  `manuscript-experiments`, `git_dirty`: `true` (i.e. they were produced by
  uncommitted changes on top of 9304081 on the `manuscript-experiments` branch
  and later committed into this tree)
- timestamps 2026-09-15T01:04:04Z to 2026-09-15T02:17:42Z
- Python 3.12.14, Windows-11-10.0.26200-SP0, CPU "Intel64 Family 6 Model 151 Stepping 2, GenuineIntel"
- numpy 2.5.3, scipy 1.18.1, networkx 3.6.1, matplotlib 3.11.1,
  piquasso 8.0.1, photographiq 0.3.1, graphix 0.4, jax 0.11.1

`STATUS.md` at 441de9a describes the same environment ("`manuscript-experiments`
branch, python 3.12.14, numpy 2.5.3, scipy 1.18.1, piquasso 8.0.1, graphix 0.4,
jax 0.11.1") and reports 42/42 implemented/executed/passed.

The metadata does NOT record native-extension components, a source
fingerprint or a GPU; those fields were added later. Nothing here should be
read as a result of the native C++ backends (`native-gaussian`, `native-fock`,
`native-mixed-fock`). The scripts at 441de9a (unchanged since they were added
in 362da97, whose source tree has no native backend modules) call only the
NumPy `gaussian` backend and the Piquasso-backed `piquasso`, `piquasso-fock`
and `piquasso-mixed-fock` backends. Where STATUS.md/ISSUES_FOUND.md at 441de9a
speak of a "native two-mode Fock GaussianTransform", that is Piquasso's own
gate inside `piquasso-fock`, not PhotoGraphiQ's later C++ engine.

## Retrieving the historical figures

```sh
git show 441de9a:paper_experiments/figures/pdf/R1_wire_teleportation.pdf > R1_wire_teleportation.pdf
git show 441de9a:paper_experiments/figures/png/R8_multimode_showcase.png > R8_multimode_showcase.png
```

The exact historical file names are listed in `figures_manifest.json` (at 441de9a some PNGs exist
only as `*_reference_style.png`, e.g. R1 and R2). Check each retrieved file against its SHA-256
there (`sha256sum`). `MANIFEST.sha256` itself is verified with `sha256sum -c MANIFEST.sha256`
from this directory (157 files OK on 2026-11-13).
