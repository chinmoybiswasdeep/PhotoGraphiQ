# Suite execution status

## Accepted publication evidence

- Accepted: `True`
- Run identifier: `ea56bf364c71-20260924T235941Z`
- Source commit: `ea56bf364c71b7f6349a9fb8eb3171a9c4a96058`
- Source-tree fingerprint: `fa50e5937a17ad6876583a018efc559f8b2bc42e9a1aa830a2ac09a0478cdf60`
- Evidence commit: `7be58767780c2989ad32a14f6f1582893fda01fb`
- Branch: `manuscript-experiments`
- Started: `2026-09-24T23:59:41.144971+00:00`
- Completed: `2026-09-25T00:13:07.178785+00:00`
- Experiments completed or hash-resumed: `59` / `59`
- Executed in this run: `0`
- Hash-resumed: `59`
- Failed: `0`
- Skipped: `0`
- Manifest SHA-256: `d721ee870de72875d563138c2f9b08f3742b019d78bbabc96ac4bee10f702d60`

## Scientific outcomes

- descriptive: `12`
- negative: `2`
- positive: `45`

Structural pass means execution and declared checks passed; it is not a positive scientific finding.
R48 retains the observed physical-training instability as a negative result.

## Quality gates

- `generate_tables`: `pass` (0.4 s; log `results/logs/publication_generate_tables.log`)
- `build_notebook`: `pass` (0.5 s; log `results/logs/publication_build_notebook.log`)
- `build_manifest`: `pass` (11.2 s; log `results/logs/publication_build_manifest.log`)
- `verify_evidence_pre_status`: `pass` (12.4 s; log `results/logs/publication_verify_evidence_pre_status.log`)
- `repository_validation`: `pass` (150.9 s; log `results/logs/publication_repository_validation.log`)
- `notebook_validation`: `pass` (0.4 s; log `results/logs/publication_notebook_validation.log`)
- `clean_install`: `pass` (6.3 s; log `results/logs/publication_clean_install.log`)
- `installed_import_smoke`: `pass` (2.9 s; log `results/logs/publication_installed_import_smoke.log`)

## Capability boundary

Arbitrary-angle physical XY measurements, general physical MuTA, native CVMuTA, loss/detector-noise models, soft adaptive flow decoding, and physical MuTA kernels remain unsupported. No experiment is described as quantum advantage.
