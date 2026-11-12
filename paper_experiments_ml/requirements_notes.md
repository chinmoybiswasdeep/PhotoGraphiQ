# Environment this suite was executed in

Recorded at suite-authoring time (2026-09-16/17); every experiment's own
`<name>.metadata.json` additionally records the exact git commit/branch and
package versions at the moment it ran, so this file is a human-readable
summary, not the authoritative record.

- Platform: Windows 11 Home, `Windows-11-10.0.26200-SP0`
- Python: 3.14.4 (project `.venv`)
- Repository: `PhotoGraphiQML`, branch `manuscript-experiments`
- Sibling checkout: `PhotoGraphiQ` at `photographiq==0.3.1` (editable
  install from `../PhotoGraphiQ`), matching PhotoGraphiQML's pinned
  `photographiq>=0.3.1,<0.4` dependency and the audited commit
  `db07f9f9bf47da841bfa6b206562c5a3ffb121d3` referenced in
  `docs/physical/architecture.md`

| Package | Version |
|---|---|
| photographiqml | 0.2.0 (editable) |
| photographiq | 0.3.1 (editable) |
| mentpy | 0.1.0a15 (editable, `.references/mentpy`) |
| piquasso | 8.0.1 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| networkx | 3.6.1 |
| matplotlib | 3.11.1 |
| scikit-learn | 1.9.1 |

## Notes on running this suite elsewhere

- All four packages above (`photographiqml`, `photographiq`, `mentpy`,
  `piquasso`) are required; `photographiq` and `mentpy` are editable
  installs from local sibling checkouts in this environment, not PyPI
  releases — reproducing this suite elsewhere requires either those same
  checkouts or pinned releases matching the versions above.
- Physical (GKP/Piquasso-backed) experiments are the expensive part of this
  suite: `physical-shots` mode at `cutoff=40` (needed once `shots>1`, per
  `docs/physical/evidence.json`) and 2-mode Fock simulations at `cutoff>=24`
  each take from several seconds to a few minutes on this host. `R44`,
  `R47`, and `R48` are the heaviest scripts (several minutes each);
  `run_all_full.py` should be expected to take on the order of 10-20
  minutes total on comparable hardware. `run_all_safe.py` skips or lightens
  these where noted in its own script.
- No GPU is used or required; all backends are CPU (`piquasso-fock`,
  NumPy).
- `.references/mentpy` is a local editable checkout used only for the
  optional `validation` extra; core PhotoGraphiQML execution does not
  import MentPy.
