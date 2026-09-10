# Installation

Use Python 3.11–3.14. Source installation is the documented distribution path;
this documentation does not assume that a PyPI release exists.

```sh
git clone https://github.com/chinmoybiswasdeep/PhotoGraphiQ.git
cd PhotoGraphiQ
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell or
`source .venv/bin/activate` on macOS/Linux. During development, the v0.3 source
branch is `feature/v0.3`; use the published release tag once one exists.

```sh
python -m pip install .
python -c "import photographiq as pg; print(pg.__version__)"
```

Piquasso is a core dependency, pinned to 8.0.1 because native instructions and
conventions are explicitly tested. Matplotlib, Graphix and JAX are optional.

| Purpose | From the cloned repository |
|---|---|
| Plotting | `python -m pip install '.[visualization]'` |
| Graphix structural comparisons | `python -m pip install '.[validation]'` |
| Automatic differentiation | `python -m pip install '.[autodiff]'` |
| Edit and test | `python -m pip install -e '.[dev]'` |
| Build documentation/notebooks | `python -m pip install -e '.[docs]'` |

An editable install follows local source changes. Quote extras in all shells.
Optional dependencies may support fewer hardware/platform combinations than the
core; the differentiable examples use CPU JAX with 64-bit precision enabled.

Run `python examples/first_cluster.py` from the repository root. If importing
fails, check `python -m pip show photographiq` using the same interpreter that runs
the script. Build documentation with `python -m mkdocs serve` after installing docs.
