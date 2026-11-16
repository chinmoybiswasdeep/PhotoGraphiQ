"""Build the Colab notebook: one self-contained code cell per manuscript experiment.

``notebooks/PhotoGraphiQ_Manuscript_Experiments_Colab.ipynb`` holds both suites,
``paper_experiments/`` (PhotoGraphiQ, R1-R60) and ``paper_experiments_ml/`` (PhotoGraphiQML,
R1-R59 and R_PERF). Its first cell installs PhotoGraphiQ/PhotoGraphiQML from the ``upgraded``
branch with every optional dependency; after that every experiment cell runs on its own:

* the experiment script is copied verbatim, minus its module docstring (shown as the theory
  section above the cell, from ``notebooks/manuscript_experiments_theory.md``), its ``sys.path``
  set-up and its imports of the suites' local modules;
* those local modules are inlined instead: the ``common``/``metadata`` helpers (from
  ``scripts/notebook_cell_helpers.py``) and the functions the script takes from
  ``extended_studies.py``, ``tests/validation`` or ``benchmarks``, each reduced to what the
  experiment uses, transitively; an inlined name that collides with one of the script's own
  names is renamed;
* command-line options become ``ARGV = ["--full"]``: every experiment runs its publication
  configuration.

Regenerate after editing an experiment, a helper or the theory text:

    python scripts/build_colab_notebook.py
"""

from __future__ import annotations

import ast
import functools
import io
import re
import sys
import tokenize
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "notebooks" / "PhotoGraphiQ_Manuscript_Experiments_Colab.ipynb"
THEORY = REPO / "notebooks" / "manuscript_experiments_theory.md"
HELPERS = REPO / "scripts" / "notebook_cell_helpers.py"
SUITES = {"core": REPO / "paper_experiments", "ml": REPO / "paper_experiments_ml"}
MODULES = {  # local modules the scripts import, besides common/metadata
    "extended_studies": REPO / "paper_experiments_ml" / "extended_studies.py",
    "test_val_gaussian": REPO / "tests" / "validation" / "test_val_gaussian.py",
    "test_val_fock": REPO / "tests" / "validation" / "test_val_fock.py",
    "bench_common": REPO / "benchmarks" / "bench_common.py",
    "bench_gbs": REPO / "benchmarks" / "gaussian_boson_sampling" / "bench_gbs.py",
    "threshold_reference_matrix": REPO
    / "benchmarks"
    / "gaussian_boson_sampling"
    / "threshold_reference_matrix.py",
    "r43": REPO / "paper_experiments" / "14_native_gaussian" / "43_native_gaussian_validation.py",
}
HELPER_NAMES = {"common", "metadata"}
CELL_HEADER = (
    "import importlib.util\nimport json\nimport re\nimport textwrap\nimport time\n"
    "from pathlib import Path\n\nimport numpy as np\n"
)


# --------------------------------------------------------------------------------------------
# Source units: one top-level statement with its line span, defined and used names
# --------------------------------------------------------------------------------------------


class Unit:
    def __init__(self, node: ast.stmt, lines: list[str], previous_end: int = 0):
        self.node = node
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        while first - 1 > previous_end and lines[first - 2].lstrip().startswith("#"):
            first -= 1  # keep the comment block directly above the statement
        self.start, self.end = first, node.end_lineno
        self.text = "".join(lines[self.start - 1 : self.end])
        if not self.text.endswith("\n"):
            self.text += "\n"
        self.gap = "".join(lines[previous_end : self.start - 1])  # blank lines before it
        self.defines = defined_names(node)
        self.uses = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        self.local_import = local_import(node)


def defined_names(node: ast.stmt) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in node.names}
    names = set()  # assignments and compound statements (if/try/with/for at module level)
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            names.add(child.id)
        elif isinstance(child, (ast.FunctionDef, ast.ClassDef)):
            names.add(child.name)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in child.names}
    return names


def local_import(node: ast.stmt):
    """('from', module, [names]) / ('alias', module, alias) for imports of suite-local modules."""
    if isinstance(node, ast.ImportFrom) and node.module in MODULES | dict.fromkeys(HELPER_NAMES):
        return ("from", node.module, [a.name for a in node.names])
    if isinstance(node, ast.Import):
        for a in node.names:
            if a.name in MODULES or a.name in HELPER_NAMES:
                return ("alias", a.name, a.asname or a.name)
    return None


def units_of(source: str) -> list[Unit]:
    lines = source.splitlines(keepends=True)
    units, previous_end = [], 0
    for node in ast.parse(source).body:
        units.append(Unit(node, lines, previous_end))
        previous_end = node.end_lineno
    return units


@functools.lru_cache(maxsize=None)
def module_units(module: str) -> tuple[Unit, ...]:
    return tuple(units_of(MODULES[module].read_text(encoding="utf-8")))


def join_plain(units) -> str:
    """Statements of an inlined module, PEP 8 spaced (two blank lines around defs)."""
    block = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    out, previous = [], None
    for unit in units:
        if previous is not None:
            out.append(
                "\n\n" if isinstance(unit.node, block) or isinstance(previous.node, block) else ""
            )
        out.append(unit.text)
        previous = unit
    return "".join(out)


def is_docstring(unit: Unit) -> bool:
    return (
        isinstance(unit.node, ast.Expr)
        and isinstance(unit.node.value, ast.Constant)
        and isinstance(unit.node.value.value, str)
    )


def is_path_setup(unit: Unit) -> bool:
    return unit.text.lstrip().startswith(("sys.path.", "sys.path[")) and not unit.defines - {"sys"}


def is_main_guard(unit: Unit) -> bool:
    return isinstance(unit.node, ast.If) and "__name__" in ast.unparse(unit.node.test)


def rename(text: str, mapping: dict[str, str]) -> str:
    """Rename NAME tokens (never attributes after '.') preserving the original formatting."""
    if not mapping:
        return text
    previous = None
    tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for token in tokens:
        if token.type == tokenize.NAME and token.string in mapping and previous != ".":
            begin = offsets[token.start[0] - 1] + token.start[1]
            edits.append((begin, begin + len(token.string), mapping[token.string]))
        if token.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT):
            previous = token.string
    for begin, end, new in reversed(edits):
        text = text[:begin] + new + text[end:]
    return text


# --------------------------------------------------------------------------------------------
# Helper classes (common / metadata) reduced to what a cell uses
# --------------------------------------------------------------------------------------------


def helper_members() -> dict[str, list[tuple[set[str], set[str], str]]]:
    """class name -> [(defined names, dependencies, source)] for each class-body statement."""
    source = HELPERS.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    members = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.ClassDef):
            continue
        entries = []
        for stmt in node.body:
            unit = Unit(stmt, lines)
            if is_docstring(unit):
                continue
            names = (
                defined_names(stmt)
                if not isinstance(stmt, (ast.FunctionDef, ast.ClassDef))
                else {stmt.name}
            )
            deps = set(re.findall(r"\bcommon\.(\w+)", unit.text))
            if not isinstance(stmt, (ast.FunctionDef, ast.ClassDef)):
                deps |= unit.uses  # class-body constants use earlier constants by bare name
            entries.append((names, deps, unit.text))
        members[node.name] = entries
    return members


MEMBERS = helper_members()


def helper_class(name: str, sources: list[str], needed: set[str]) -> tuple[str, set[str]]:
    """Source of ``class <name>`` holding the needed members of the given helper classes."""
    entries = [e for source in sources for e in MEMBERS[source]]
    known = set().union(*(names for names, _, _ in entries))
    wanted, queue, common_needs = set(), [n for n in needed if n in known], set()
    while queue:
        member = queue.pop()
        if member in wanted:
            continue
        wanted.add(member)
        for names, deps, text in entries:
            if member in names:
                queue += [d for d in deps if d in known]
                common_needs |= set(re.findall(r"\bcommon\.(\w+)", text))
    unknown = {n for n in needed if n not in known}
    if unknown:
        raise SystemExit(f"helper class {name}: no member(s) {sorted(unknown)} in {HELPERS.name}")
    body, previous_def = [], None
    for names, _, text in entries:  # blank line around methods, none between constants
        if names & wanted:
            is_def = text.lstrip().startswith(("def ", "@", "class "))
            if previous_def is not None and (is_def or previous_def):
                body.append("")
            body.append(text.rstrip("\n"))
            previous_def = is_def
    doc = {
        "common": "Shared manuscript helpers: only the members this experiment uses.",
        "metadata": "Environment provenance recorded next to the results.",
    }[name]
    return f'class {name}:\n    """{doc}"""\n\n' + "\n".join(body) + "\n", common_needs


# --------------------------------------------------------------------------------------------
# Inlining local modules
# --------------------------------------------------------------------------------------------


class Inliner:
    """Collects, to a fixpoint, the statements of each local module a cell needs."""

    def __init__(self):
        self.requests: dict[str, set[str]] = {}  # module -> requested top-level names
        self.aliases: dict[str, tuple[str, set[str]]] = {}  # alias -> (module, attributes)
        self.edges: dict[str, set[str]] = {}  # module -> inlined modules it imports from
        self.common: set[str] = set()
        self.metadata: set[str] = set()

    def request(self, module: str, names: set[str]) -> None:
        if module in HELPER_NAMES:
            getattr(self, module).update(names)
            return
        before = set(self.requests.get(module, set()))
        self.requests.setdefault(module, set()).update(names)
        if self.requests[module] != before:
            self._resolve(module)

    def request_alias(self, module: str, alias: str, attributes: set[str]) -> None:
        old = self.aliases.get(alias, (module, set()))[1]
        self.aliases[alias] = (module, old | attributes)
        self.request(module, attributes)

    def needed_units(self, module: str) -> list[Unit]:
        units = module_units(module)
        by_name = {n: u for u in units for n in u.defines}
        wanted, queue = [], list(self.requests[module])
        while queue:
            unit = by_name.get(queue.pop())
            if unit is None or unit in wanted:
                continue
            wanted.append(unit)
            if not unit.local_import:
                queue += [n for n in unit.uses if n in by_name]
        return sorted(wanted, key=lambda u: u.start)

    def _resolve(self, module: str) -> None:
        units = self.needed_units(module)
        code = "".join(u.text for u in units if not u.local_import)
        used = set().union(*(u.uses for u in units if not u.local_import))
        for unit in units:
            if unit.local_import is None:
                continue
            kind, target, binding = unit.local_import
            if target not in HELPER_NAMES:
                self.edges.setdefault(module, set()).add(target)
            if kind == "from":
                self.request(target, set(binding) & used)
            elif target in HELPER_NAMES:
                self.request(target, set(re.findall(rf"\b{binding}\.(\w+)", code)))
            else:
                self.request_alias(target, binding, set(re.findall(rf"\b{binding}\.(\w+)", code)))

    def order(self) -> list[str]:
        """Inlined modules, each after the modules it imports from."""
        done: list[str] = []

        def visit(module: str) -> None:
            if module not in done:
                for dependency in sorted(self.edges.get(module, ())):
                    visit(dependency)
                done.append(module)

        for module in sorted(self.requests):
            visit(module)
        return done


# --------------------------------------------------------------------------------------------
# One experiment -> one cell
# --------------------------------------------------------------------------------------------

R44_SELF = """# Each timing configuration runs in a fresh Python process (peak memory is per process), so the
# notebook cell saves its own source as the worker script those processes execute.
if "__file__" in globals():  # this is a worker process running the saved script
    HERE = Path(__file__).resolve()
else:
    import tempfile

    HERE = Path(tempfile.gettempdir()) / "R44_native_gaussian_performance_worker.py"
    HERE.write_text(get_ipython().user_ns["_ih"][-1], encoding="utf-8")
"""
R44_LOAD = '''def _load_r43():
    """R43's reference helpers, inlined above as ``r43``."""
    return r43
'''


def experiment_cell(suite: str, script: Path) -> str:
    source = script.read_text(encoding="utf-8")
    units = units_of(source)
    inliner = Inliner()
    future = any(
        isinstance(u.node, ast.ImportFrom) and u.node.module == "__future__" for u in units
    )
    rid = experiment_id(suite, script)
    kept: list[tuple[Unit, str]] = []
    for unit in units:
        if is_docstring(unit) and unit is units[0]:
            continue
        if isinstance(unit.node, ast.ImportFrom) and unit.node.module == "__future__":
            continue
        if is_path_setup(unit):
            continue
        if unit.local_import:
            kind, module, binding = unit.local_import
            if module == "extended_studies":
                continue  # handled with the main guard below
            if kind == "from":
                inliner.request(module, set(binding))
            elif module in HELPER_NAMES:
                inliner.request(module, set(re.findall(rf"\b{binding}\.(\w+)", source)))
            else:
                inliner.request_alias(
                    module, binding, set(re.findall(rf"\b{binding}\.(\w+)", source))
                )
            continue
        if "__file__" in unit.text and isinstance(unit.node, ast.Assign):
            (name,) = unit.defines
            used_elsewhere = any(
                name in u.uses for u in units if u is not unit and not is_path_setup(u)
            )
            if not used_elsewhere:
                continue
            if name == "HERE" and rid == "R44":
                kept.append((unit, R44_SELF))
            elif name == "REPO":
                kept.append(
                    (unit, "REPO = Path.cwd()  # the notebook runs from the repository clone\n")
                )
            else:
                raise SystemExit(f"{script}: cannot rewrite {unit.text.strip()!r}")
            continue
        if rid == "R44" and unit.defines == {"_load_r43"}:
            inliner.request_alias("r43", "r43", set(re.findall(r"\br43\.(\w+)", source)))
            kept.append((unit, R44_LOAD))
            continue
        if is_main_guard(unit) and "extended_studies" in source:
            study = re.search(r'run\("(R\d+)"\)', unit.text).group(1).lower()
            inliner.request("extended_studies", {study})
            kept.append((unit, f'if __name__ == "__main__":\n    {study}()\n'))
            continue
        kept.append((unit, unit.text))

    for outer, (_, attributes) in list(inliner.aliases.items()):  # r43.ref.X: nested namespaces
        for inner in [a for a in inliner.aliases if a in attributes]:
            extra = set(re.findall(rf"\b{outer}\.{inner}\.(\w+)", source))
            if extra:
                inliner.request_alias(inliner.aliases[inner][0], inner, extra)
    script_defines = set().union(*(u.defines for u, _ in kept))
    body = join_units(kept)
    reads_args = "parse_args" in body or "sys.argv" in body
    body = (
        body.replace(".parse_args()", ".parse_args(ARGV)")
        .replace(".parse_args(argv)", ".parse_args(ARGV if argv is None else argv)")
        .replace("sys.argv[1:]", "ARGV")
    )

    # Inlined modules, renamed where they collide with the script or each other.
    # Re-importing the same object under the same name (np, pg, ...) is not a collision.
    blocks, taken = [], set(script_defines) | HELPER_NAMES | set(inliner.aliases)
    imported = import_bindings(u for u, _ in kept)
    renames: dict[str, dict[str, str]] = {}
    for module in inliner.order():
        units_m = [u for u in inliner.needed_units(module) if not u.local_import]
        defines = set().union(*(u.defines for u in units_m)) if units_m else set()
        own_imports = import_bindings(units_m)
        clash = {n for n in defines & taken if own_imports.get(n, n) != imported.get(n)}
        renames[module] = {n: f"_{module}_{n}" for n in clash}
        taken |= defines - clash | set(renames[module].values())
        imported |= own_imports
    for module in inliner.order():
        units_m = inliner.needed_units(module)
        mapping = dict(renames[module])
        for unit in units_m:  # names this module imports from another inlined module
            if unit.local_import and unit.local_import[0] == "from":
                mapping |= {
                    n: renames.get(unit.local_import[1], {}).get(n, n) for n in unit.local_import[2]
                }
        code = join_plain([u for u in units_m if not u.local_import])
        mapping = {k: v for k, v in mapping.items() if k != v}
        relative = MODULES[module].relative_to(REPO).as_posix()
        blocks.append(
            f"# ---- from {relative} (only what this experiment uses) ----\n"
            + squeeze(rename(code, mapping))
        )
        for alias, (target, attributes) in inliner.aliases.items():  # after the module it wraps
            if target == module:
                names = ", ".join(f"{a}={renames[module].get(a, a)}" for a in sorted(attributes))
                blocks.append(f"{alias} = SimpleNamespace({names})\n")
    # imports from inlined modules used by the script (from X import y) follow renames too
    script_map = {
        n: renames[m][n]
        for m in inliner.order()
        for n in renames[m]
        if n in inliner.requests.get(m, set()) and n not in script_defines
    }
    body = rename(body, script_map)

    common_src, more = helper_class(
        "common",
        ["Shared", "Core" if suite == "core" else "ML"],
        inliner.common | set(re.findall(r"\bcommon\.(\w+)", "".join(blocks))),
    )
    parts = []
    if inliner.metadata:
        meta_src, meta_needs = helper_class("metadata", ["Metadata"], inliner.metadata)
        common_src, _ = helper_class(
            "common",
            ["Shared", "Core" if suite == "core" else "ML"],
            inliner.common | meta_needs | set(re.findall(r"\bcommon\.(\w+)", "".join(blocks))),
        )
        parts.append(meta_src)
    relative = script.relative_to(REPO).as_posix()
    head = [
        f"# {rid} ({'PhotoGraphiQ' if suite == 'core' else 'PhotoGraphiQML'}): {relative}",
        "# Self-contained: after the installation cell, this cell runs on its own.\n",
    ]
    if future:
        head.append("from __future__ import annotations\n")
    if "__doc__" in body:  # argparse descriptions read the (moved) module docstring
        title = (ast.get_docstring(ast.parse(source)) or rid).strip().splitlines()[0]
        head.append(f"__doc__ = {title!r}\n")
    if reads_args:
        if rid == "R44" and suite == "core":
            head.append(
                "import sys\n\n# Publication configuration; worker processes pass their own arguments.\n"
                'ARGV = sys.argv[1:] if "__file__" in globals() else ["--full"]\n'
            )
        else:
            flag = (
                '["--full"]  # publication configuration'
                if "--full" in source
                else "[]  # defaults"
            )
            head.append(f"ARGV = {flag}\n")
    imports = CELL_HEADER + ("from types import SimpleNamespace\n" if inliner.aliases else "")
    cell = "\n".join(head) + "\n" + imports + "\n\n" + common_src + "\n\n"
    if parts:
        cell += "\n\n".join(parts) + "\n\n"
    if blocks:
        cell += "\n\n".join(blocks) + "\n\n"
    cell += "# ---- experiment ----\n" + body
    compile(cell.replace("get_ipython()", "None"), relative, "exec")  # syntax check
    return squeeze(cell).rstrip() + "\n"


def import_bindings(units) -> dict[str, str]:
    """name -> imported object ('numpy', 'scipy.linalg.expm') for plain import statements."""
    bound = {}
    for unit in units:
        node = unit.node
        if isinstance(node, ast.Import):
            bound |= {
                (a.asname or a.name.split(".")[0]): a.name if a.asname else a.name.split(".")[0]
                for a in node.names
            }
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            bound |= {(a.asname or a.name): f"{node.module}.{a.name}" for a in node.names}
    return bound


def join_units(kept: list[tuple[Unit, str]]) -> str:
    """Kept units with the blank lines and comments between them, as in the script."""
    out = []
    for index, (unit, text) in enumerate(kept):
        if index:
            out.append(unit.gap if unit.gap.strip() == "" else "\n")
        out.append(text if text.endswith("\n") else text + "\n")
    return "".join(out)


def squeeze(text: str) -> str:
    return re.sub(r"\n{4,}", "\n\n\n", text)


def experiment_id(suite: str, script: Path) -> str:
    number = int(script.name.split("_", 1)[0])
    return "R_PERF" if suite == "ml" and number == 49 else f"R{number}"


def scripts(suite: str) -> list[Path]:
    root = SUITES[suite]
    found = [p for p in root.glob("[0-9]*/[0-9]*_*.py")]
    return sorted(found, key=lambda p: (int(p.name.split("_", 1)[0]), p.as_posix()))


# --------------------------------------------------------------------------------------------
# Theory text and notebook assembly
# --------------------------------------------------------------------------------------------


def theory_sections() -> dict[str, tuple[str, str]]:
    """'core:R1' -> (title, markdown) from the theory file (sections start '<!-- core:R1 -->')."""
    if not THEORY.exists():
        return {}
    text = THEORY.read_text(encoding="utf-8")
    sections = {}
    for match in re.finditer(
        r"<!-- ((?:core|ml):R\w+) -->\n(.*?)(?=\n<!-- (?:core|ml):R\w+ -->|\Z)", text, re.S
    ):
        body = match.group(2).strip()
        title = re.match(r"\*\*Title:\*\*\s*(.+)", body)
        if title:
            body = body[title.end() :].strip()
        sections[match.group(1)] = (title.group(1).strip() if title else "", body)
    return sections


def docstring_fallback(script: Path) -> tuple[str, str]:
    doc = ast.get_docstring(ast.parse(script.read_text(encoding="utf-8"))) or script.stem
    title, _, body = doc.strip().partition("\n\n")
    title = re.sub(r"^R(?:\d+|_PERF)\s*[:-]\s*", "", " ".join(title.split()))
    return title, "```text\n" + body.strip() + "\n```"


INTRO = """# PhotoGraphiQ and PhotoGraphiQML: manuscript experiments

Every experiment of both manuscripts in one notebook:

* **Part I - PhotoGraphiQ (R1-R60):** continuous-variable MBQC wires and gates, the Gaussian
  compiler, adaptive feed-forward and flow, independent Piquasso/NumPy/Graphix validation,
  non-Gaussian resources, GKP, noise, convergence, native C++ engines, boson sampling, flow and
  measurement calculus, visualization, GPU, the PhotoGraphiQML physical layer, an end-to-end
  showcase.
* **Part II - PhotoGraphiQML (R1-R59, R_PERF):** logical MuTA foundations, MentPy
  cross-validation, parameter contracts, gradients and training, quantum kernels, expressivity,
  GKP resources, physical lowering, Piquasso validation, physical statistics, convergence,
  performance and the extended studies.

**How to run.** Run the installation cell once per Colab runtime: it installs Times New Roman,
clones the `upgraded` branch and installs PhotoGraphiQ and PhotoGraphiQML from source with every
optional dependency (compiling the C++ engines takes several minutes). Each experiment then has a
theory section (why the experiment is needed, the physics, the expected outcome) and **one
self-contained code cell**: it carries the helpers it needs, so it runs on its own in any order.
Only R_PERF reads the saved results of R13, R36, R42 and R47 and draws a placeholder for any not
yet run.

**Configuration.** Every cell runs the experiment's **full publication configuration**
(`ARGV = ["--full"]`), never the quick smoke settings. The complete notebook takes many hours; the
large-cutoff Fock, GKP and timing studies dominate. Timing results are labelled provisional unless
the environment variable `PGQ_IDLE_MACHINE=1` declares an otherwise idle machine. R58 needs a GPU
runtime (*Runtime > Change runtime type > GPU*); without one it records the CPU side only.

**Outputs.** Figures are shown inline and written to `manuscript_outputs/<suite>/figures/`
(vector PDF and SVG, 600-dpi PNG; Times New Roman, white background, journal column widths);
results go to `manuscript_outputs/<suite>/results/` (CSV, JSON and an environment record per
experiment). The last cell zips everything for download.

*Generated by `scripts/build_colab_notebook.py` from the experiment scripts; do not edit by hand.*"""

INSTALL = """# Installation: run once per Colab runtime (several minutes; compiles the C++ engines).
# Outside Colab, install the repository yourself instead (from its root):
#   pip install ".[dev,autodiff,visualization,validation,precision,ml-validation,ml-experiments]"
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/chinmoybiswasdeep/PhotoGraphiQ.git"
BRANCH = "upgraded"
CLONE = Path("/content/PhotoGraphiQ")
EXTRAS = "dev,autodiff,visualization,validation,precision,ml-validation,ml-experiments"


def sh(command):
    print("$", command, flush=True)
    subprocess.run(command, shell=True, check=True)


if "google.colab" in sys.modules:
    # Times New Roman for the figures (Microsoft core fonts; the EULA is accepted non-interactively).
    sh("echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true"
       " | debconf-set-selections")
    sh("apt-get -qq update > /dev/null && apt-get -qq install -y ttf-mscorefonts-installer"
       " fonts-liberation > /dev/null")
    sh("rm -rf ~/.cache/matplotlib")
    if not CLONE.exists():
        sh(f"git clone --branch {BRANCH} --single-branch {REPO_URL} {CLONE}")
    os.chdir(CLONE)
    # PhotoGraphiQ + PhotoGraphiQML from source, with Piquasso, Graphix, MentPy, JAX, mpmath,
    # scikit-learn and the other optional dependencies the experiments use.
    sh(f'"{sys.executable}" -m pip install -q ".[{EXTRAS}]"')
print("working directory:", Path.cwd())"""

CHECK = """# Environment check: package versions, native engines, figure font and GPU.
import importlib.util
from importlib.metadata import PackageNotFoundError, version

import matplotlib
from matplotlib import font_manager

for package in ("photographiq", "numpy", "scipy", "matplotlib", "piquasso", "graphix", "mentpy",
                "jax", "mpmath", "scikit-learn", "cupy"):
    try:
        print(f"{package:>14}: {version(package)}")
    except PackageNotFoundError:
        print(f"{package:>14}: not installed")
import photographiq
import photographiqml  # noqa: F401
from photographiq import native

print("native engines:", native.AVAILABLE, list(native.components()))
for font in __import__("glob").glob("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman*.ttf"):
    font_manager.fontManager.addfont(font)
print("figure font:", font_manager.findfont(font_manager.FontProperties(family="Times New Roman")))
print("CUDA GPU (CuPy):", importlib.util.find_spec("cupy") is not None)"""

ARCHIVE = """# Archive every figure and result for download.
import shutil
import sys
from pathlib import Path

archive = shutil.make_archive("manuscript_outputs", "zip", root_dir=".", base_dir="manuscript_outputs")
print("archive:", Path(archive).resolve())
if "google.colab" in sys.modules:
    from google.colab import files

    files.download(archive)"""

PARTS = {
    "core": (
        "Part I - PhotoGraphiQ manuscript experiments (R1-R60)",
        "Continuous-variable photonic MBQC: from one-step teleportation wires to the compiled, "
        "adaptive, non-Gaussian and fault-tolerance-relevant workloads of the PhotoGraphiQ "
        "paper, each checked against an independent reference.",
    ),
    "ml": (
        "Part II - PhotoGraphiQML manuscript experiments (R1-R59, R_PERF)",
        "The learning layer: MuTA models on the logical level, their validation against MentPy "
        "and analytic oracles, training, kernels and expressivity, then the physical lowering "
        "to GKP-encoded CV MBQC and its statistics, convergence and cost.",
    ),
}
WORDS = {
    "mbqc": "MBQC",
    "gkp": "GKP",
    "gpu": "GPU",
    "non": "non-",
    "gaussian": "Gaussian",
    "fock": "Fock",
    "piquasso": "Piquasso",
    "graphix": "Graphix",
    "mentpy": "MentPy",
    "photographiqml": "PhotoGraphiQML",
}


def section(folder: str) -> str:
    text = " ".join(WORDS.get(w, w) for w in folder.split("_")[1:]).replace("- ", "-")
    return text[0].upper() + text[1:]


def main() -> None:
    theory = theory_sections()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md(INTRO),
        md("## Installation"),
        code(INSTALL),
        md("## Environment check"),
        code(CHECK),
    ]
    missing = []
    for suite in ("core", "ml"):
        title, blurb = PARTS[suite]
        cells.append(md(f"# {title}\n\n{blurb}"))
        current = None
        for script in scripts(suite):
            folder = script.parent.name
            if folder != current:
                cells.append(md(f"## {section(folder)}"))
                current = folder
            rid = experiment_id(suite, script)
            name, text = theory.get(f"{suite}:{rid}", ("", ""))
            if not text:
                missing.append(f"{suite}:{rid}")
                name, text = docstring_fallback(script)
            cells.append(md(f"### {rid} · {name}\n\n{text}"))
            cells.append(code(experiment_cell(suite, script)))
    cells += [md("# Archive the outputs"), code(ARCHIVE)]
    nb = nbf.v4.new_notebook(cells=cells)
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {"provenance": [], "toc_visible": True},
        "accelerator": "GPU",
    }
    nbf.validate(nb)
    OUTPUT.write_text(nbf.writes(nb), encoding="utf-8", newline="\n")
    print(f"Wrote {OUTPUT} ({len(cells)} cells, {OUTPUT.stat().st_size / 1e6:.1f} MB)")
    if missing:
        print(
            f"theory missing (docstring used) for {len(missing)}: {' '.join(missing)}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
