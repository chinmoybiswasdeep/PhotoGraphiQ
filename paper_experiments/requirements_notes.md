# Requirements notes

This suite runs inside the same environment as the rest of the repository;
it adds no new dependencies beyond what `pyproject.toml` already declares.

## Core (always required)

`numpy`, `scipy`, `networkx`, `matplotlib`, `piquasso==8.0.1`, and
`photographiq` itself (installed in editable/source form, e.g.
`python -m pip install -e '.[dev]'` from the repository root).

## Optional, feature-gated

| Package | Used by | Behavior if missing |
|---|---|---|
| `graphix==0.4` | R17 | Script prints `SKIPPED` and exits 0. `run_all_safe.py` always skips it (marked "optional dependency"). |
| `jax>=0.4.35` (with `jax_enable_x64`) | R41, R42 | Same skip behavior. |

Both were installed and detected in the environment this suite was built
and validated on (`graphix 0.4`, `jax 0.11.1`); `common.HAS_GRAPHIX` /
`common.HAS_JAX` report their availability at import time.

## Environment actually used to build/validate this suite

Recorded by `metadata.py` and embedded as a companion
`<experiment>.metadata.json` next to every result:

- Python 3.12.14 (Windows)
- numpy 2.5.3, scipy 1.18.1, networkx 3.6.1, matplotlib 3.11.1
- piquasso 8.0.1, graphix 0.4, jax 0.11.1
- photographiq 0.3.1 (this checkout)

## How to check your own environment

```sh
python -c "import paper_experiments.metadata as m, json; print(json.dumps(m.collect(), indent=2))"
```

(run from the repository root), or simply run any experiment script --
every one writes its own `*.metadata.json` companion file.

## Large-Fock / long-benchmark scripts (excluded from `run_all_safe.py`)

These are not "optional dependencies" but are excluded from the lightweight
suite for cost reasons; see `EXPERIMENT_INDEX.md` for the full breakdown.
They need nothing beyond the core dependencies above -- just more wall time
and, for a few, more RAM (a handful of megabytes at worst for the cutoffs
used here; see `docs/performance.md` and R35 for the general scaling
formula before increasing any cutoff further).
