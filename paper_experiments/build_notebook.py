"""Combine every experiment script in paper_experiments/ into one notebook.

Reads common.py, metadata.py and all R1-R42 scripts (in numeric order),
strips the per-script ``sys.path.insert(...)`` boilerplate (replaced by one
shared setup cell), and writes a single executable Jupyter notebook with a
pip-install cell for every dependency. Regenerate after editing any script:

    python build_notebook.py
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "PhotoGraphiQ_Manuscript_Experiments.ipynb"

# One shared sys.path.insert line appears near the top of every script;
# the notebook's setup cell does this once for the whole kernel instead.
SYS_PATH_LINE = re.compile(
    r"^sys\.path\.insert\(0, str\(Path\(__file__\)\.resolve\(\)\.parents\[1\]\)\)\n",
    re.MULTILINE,
)

DOCSTRING_TITLE = re.compile(r'^"""(.+?)\n\n(.*?)"""', re.DOTALL)


def strip_boilerplate(source: str) -> str:
    return SYS_PATH_LINE.sub("", source)


def title_and_body(source: str):
    match = DOCSTRING_TITLE.match(source)
    if not match:
        return None, source
    title, body = match.group(1).strip(), match.group(2).strip()
    return title, body


# Matches the module docstring plus a following `from __future__ import
# annotations` line, which (per Python's grammar) must stay module/cell-level
# statements preceding any other code -- they cannot be indented inside a
# try block, unlike everything else in these scripts.
FUTURE_IMPORT_HEADER = re.compile(r'^(""".*?"""\n)(\nfrom __future__ import annotations\n)?', re.DOTALL)


def guard_optional_dependency(source: str) -> str:
    """Make an optional-dependency script's ``sys.exit(0)`` skip harmless in-notebook.

    Standalone, ``sys.exit(0)`` cleanly aborts just that script (and is what
    ``run_all_safe.py`` relies on). Inside one shared notebook kernel, an
    uncaught SystemExit is reported as a cell error, and nbclient/Jupyter's
    "Run All" stops there by default -- which would abort every later
    experiment in an environment simply missing graphix or jax. Wrapping the
    rest of the cell body in try/except SystemExit preserves the "skip
    cleanly" behavior without blocking subsequent cells. The docstring and
    any ``from __future__ import annotations`` must stay outside the try
    block: Python requires future-imports to be the first statement.
    """
    if "sys.exit(0)" not in source:
        return source
    match = FUTURE_IMPORT_HEADER.match(source)
    header, body = (source[: match.end()], source[match.end() :]) if match else ("", source)
    indented = "\n".join("    " + line if line.strip() else line for line in body.splitlines())
    return header + "try:\n" + indented + "\nexcept SystemExit:\n    pass\n"


def script_cells(path: Path):
    source = strip_boilerplate(path.read_text(encoding="utf-8"))
    title, description = title_and_body(source)
    heading = f"## {title}" if title else f"## {path.stem}"
    md = heading + (f"\n\n{description}" if description else "")
    return [nbf.v4.new_markdown_cell(md), nbf.v4.new_code_cell(guard_optional_dependency(source))]


def main():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# PhotoGraphiQ manuscript experiment suite (R1-R42)\n\n"
            "One executable notebook combining every script in `paper_experiments/` "
            "(`common.py`, `metadata.py`, and experiments R1-R42, in order). Each "
            "experiment's own module docstring is reproduced as its section heading.\n\n"
            "**Runtime**: running the whole notebook top-to-bottom executes every "
            "experiment, including the large-Fock and long-benchmark ones excluded "
            "from `run_all_safe.py` (see `README.md`) -- expect roughly 30-45 minutes "
            "in total, dominated by R28 (~9 min), R36 (~4 min) and the cutoff=24-96 "
            "non-Gaussian/GKP scripts (R18, R20-R23, R26-R28, R33-R34).\n\n"
            "**Isolation note**: `run_all_safe.py` runs each script in its own "
            "subprocess so failures/warnings can't leak between experiments; this "
            "notebook instead runs everything in one kernel for a single linear "
            "narrative. Every script's `main()` is still called immediately after its "
            "own constants/helpers are (re)defined in the same cell, so redefining a "
            "same-named helper (e.g. `independent_rotation`) in a later cell cannot "
            "affect an earlier cell's already-completed run.\n\n"
            "**Optional dependencies**: R17 (Graphix), R41-R42 (JAX) print `SKIPPED` "
            "and raise `SystemExit(0)` if their optional package is missing -- Jupyter "
            "shows this as a brief traceback-like message but the notebook keeps going. "
            "The install cell below installs both, so on a fresh environment they run "
            "for real."
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## Install dependencies"))
    cells.append(
        nbf.v4.new_code_cell(
            "# Core + optional extras (graphix, jax) + matplotlib, from the local checkout.\n"
            "# Run once per environment; safe to re-run (pip no-ops on already-satisfied requirements).\n"
            "import subprocess\n"
            "import sys\n\n"
            "try:\n"
            "    import pip  # noqa: F401\n"
            "except ImportError:\n"
            "    # Some environments (e.g. a uv-provisioned interpreter) ship no pip at\n"
            "    # all; bootstrap it from the standard library before installing anything.\n"
            "    print(\"pip not found in this interpreter; bootstrapping via ensurepip...\")\n"
            "    subprocess.run([sys.executable, \"-m\", \"ensurepip\", \"--upgrade\"], check=True)\n\n"
            "subprocess.run(\n"
            "    [sys.executable, \"-m\", \"pip\", \"install\", \"-e\", \"..[dev,autodiff]\"],\n"
            "    check=True,\n"
            ")\n"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## Setup: make `common` and `metadata` importable"))
    cells.append(
        nbf.v4.new_code_cell(
            "import sys\n"
            "from pathlib import Path\n\n"
            "PAPER_EXPERIMENTS_DIR = Path.cwd()\n"
            "if not (PAPER_EXPERIMENTS_DIR / \"common.py\").exists():\n"
            "    # Fall back to this file's location if the notebook's cwd differs\n"
            "    # from paper_experiments/ (e.g. launched from the repo root).\n"
            "    PAPER_EXPERIMENTS_DIR = Path(\"paper_experiments\").resolve()\n"
            "sys.path.insert(0, str(PAPER_EXPERIMENTS_DIR))\n\n"
            "import common\n"
            "import metadata\n\n"
            "print(\"paper_experiments dir:\", PAPER_EXPERIMENTS_DIR)\n"
            "print(\"HAS_GRAPHIX:\", common.HAS_GRAPHIX, \" HAS_JAX:\", common.HAS_JAX)\n"
        )
    )

    for path in sorted(ROOT.glob("*/*.py")):
        cells.extend(script_cells(path))

    cells.append(nbf.v4.new_markdown_cell("## Aggregate manuscript tables (`generate_tables.py`)"))
    cells.append(
        nbf.v4.new_code_cell(
            "import importlib\n"
            "import generate_tables\n"
            "importlib.reload(generate_tables)\n"
            "generate_tables.main()\n"
        )
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.validate(nb)
    OUTPUT.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
