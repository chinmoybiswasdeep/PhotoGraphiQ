# Contributing

Create a virtual environment with Python 3.11 or newer and install `pip install -e '.[dev]'`.
Before proposing a change run:

```sh
python -m pytest
python -m ruff check src tests examples experiments
python -m ruff format --check src tests examples experiments
python -m mypy src/photographiq
python -m build
```

The checked type environment uses Python 3.12 because recent NumPy stubs use its
syntax. Runtime CI also covers 3.11 and 3.13. Add an independent analytical test
for physics changes. A test against the same backend alone is insufficient.
Never import qubit parity or GF(2) rules as continuous-variable rules. New
measurements must specify outcome units, state update, supported backends and
destructive/non-destructive semantics. Do not silently approximate unsupported
physics. Record approximations and numerical tolerances in the documentation.

Changes to the paper's numerical claims require rerunning `experiments/reproduce.py`.
The PDF references bundled upstream PDFs for study; their copyrights remain with
their authors. Do not relicense them under the repository's MIT license.
