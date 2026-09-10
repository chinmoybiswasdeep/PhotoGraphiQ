# Contributing to PhotoGraphiQ

Use Python 3.11–3.14 in a virtual environment. Clone the repository and install:

```sh
python -m pip install -e '.[dev,docs,autodiff]'
```

Create a focused branch from the current development baseline. Explain the
user-visible result in your PR, record public API changes in CHANGELOG.md, and
keep numerical behavior changes separately reviewable from documentation edits.

## Checks

```sh
python -m pytest
python -m pytest tests/analytical tests/piquasso_reference
python -m pytest tests/graphix_reference
python -m pytest tests/mixed_fock tests/compilation tests/gkp tests/autodiff
python -m ruff check src tests examples experiments scripts
python -m ruff format --check src tests examples experiments scripts
python -m mypy src/photographiq
python -m mkdocs build --strict
python scripts/check_public_api.py
python scripts/check_docs_examples.py --projects --notebooks
python -m build
```

`python experiments/validate_release.py --output .validation` preserves release
logs and numerical evidence. JAX is optional: its tests skip when unavailable;
the dedicated documentation/extension workflow installs it and runs them.

## Add a backend, gate or measurement

Implement BaseBackend's contract and explicit capabilities. Validate the entire
request before native preparation where possible, maintain node labels through
destructive measurements, and return independent snapshots. New mixed-state
paths need positivity/trace tests and allocation guards.

For a gate, specify the unitary and hbar convention, backend support, direct
numerical reference, any compilation rule and figure glyph. For a measurement,
specify units, probability versus density, conditional update, destructive
semantics, noise model and dependency behavior. A compilation rule needs trace
coverage and resource/error accounting. Add public docstrings and a tutorial.

## Numerical and structural validation

Use independent analytical evidence for physics changes. Comparing a wrapper
against the same implementation alone is insufficient. Keep separate budgets
for floating-point algebra, Monte Carlo sampling, quadrature integration, Fock
truncation, synthesis and finite-energy resources. Preserve failing regimes and
explain any tolerance change. State fidelity alone does not certify high moments.

Graphix is a structural reference. Compare labelled edges, I/O, order and
correction support with explicit label maps. Never import qubit parity/GF(2)
rules as continuous-variable physics. A causal DAG is not a CV-flow certificate.

## Add a tutorial or notebook

Place executable source in examples/tutorials and include it from its Markdown
page. Explain goal, theory, resources, output interpretation, validation and
exercises. Run `python scripts/check_docs_examples.py --write-outputs` to refresh
captured outputs. Keep notebooks small and deterministic; document costly studies
as optional refinements. Build strictly to catch broken internal links.

## Research artifacts and licensing

Do not overwrite historical numerical evidence. Changes to manuscript claims
require rerunning the relevant experiments. Store new large logs and generated
builds as CI/release artifacts. Original software is MIT licensed; bundled papers
and the Quantum LaTeX class retain their respective copyrights and licenses.
