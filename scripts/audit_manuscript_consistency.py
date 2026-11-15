"""Audit that documentation, experiment catalogues, results and figures agree.

    python scripts/audit_manuscript_consistency.py            # all checks, non-zero exit on error
    python scripts/audit_manuscript_consistency.py --json out.json

Checks (each reports its findings; an empty list means it passed):

 1. api            documented public API exists (``scripts/check_public_api.py`` logic)
 2. examples       code paths cited in docs (examples/, scripts/, benchmarks/, paper_experiments*/) exist
 3. index          every experiment script in an index exists, and every script is indexed and
                   listed in its suite's ``run_all_safe.py``
 4. map            every ``R<id>`` / figure / table cited by ``MANUSCRIPT_MAP.md`` exists
 5. metadata       every results JSON has provenance (companion ``.metadata.json`` or embedded key)
 6. evidence       every experiment reported as passed in ``STATUS.md`` has a results file
 7. figures        every generated figure belongs to an experiment ID that has a script
 8. backends       every registered backend is documented; removed backends are not advertised
 9. labels         every feature entry in ``COMPLETE_FUNCTIONALITY.md`` carries a status label
10. links          relative Markdown links resolve
11. suites         both suites have runnable ``run_all_safe.py`` whose scripts exist
12. conventions    no page states a hbar convention other than hbar = 2

The audit reads files only; it does not run experiments.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = ("paper_experiments", "paper_experiments_ml")
DOC_GLOBS = ("README.md", "docs/**/*.md", "paper_experiments/*.md", "paper_experiments_ml/*.md")
# Historical records may quote paths or conventions that no longer exist; they are kept verbatim.
HISTORICAL = re.compile(
    r"(legacy_reference|release-hardening-report|hardening-audit|v0\.3-|merged-integration-report|"
    r"CHANGELOG|ISSUES_FOUND|UPGRADE_IMPLEMENTATION_PLAN|docs/validation/PHASE)"
)


def _docs() -> list[Path]:
    seen: dict[Path, None] = {}
    for pattern in DOC_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file() and "site" not in path.parts:
                seen[path] = None
    return list(seen)


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _historical(path: Path) -> bool:
    return bool(HISTORICAL.search(_rel(path)))


def check_api() -> list[str]:
    import photographiq as pg

    problems = [f"missing export {n}" for n in pg.__all__ if not hasattr(pg, n)]
    for page in (ROOT / "docs" / "api").glob("*.md"):
        for line in page.read_text(encoding="utf-8").splitlines():
            if line.startswith("::: "):
                try:
                    importlib.import_module(line[4:].strip())
                except Exception as exc:  # report every unimportable documented module
                    problems.append(f"{_rel(page)}: {line[4:].strip()}: {exc}")
    return problems


CODE_PATH = re.compile(
    r"(?<![\w/.-])((?:examples|scripts|benchmarks|paper_experiments_ml|paper_experiments)"
    r"/[\w./-]+\.(?:py|ipynb|json|csv|md))"
)


def check_examples() -> list[str]:
    problems = []
    for doc in _docs():
        if _historical(doc):
            continue
        for match in CODE_PATH.finditer(doc.read_text(encoding="utf-8")):
            ref = match.group(1).rstrip(".")
            if "*" in ref or "<" in ref or "{" in ref:
                continue
            if not (ROOT / ref).exists() and not (doc.parent / ref).exists():
                problems.append(f"{_rel(doc)}: missing {ref}")
    return sorted(set(problems))


def _run_all_safe_scripts(suite: str) -> list[str]:
    tree = ast.parse((ROOT / suite / "run_all_safe.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SCRIPTS" for t in node.targets
        ):
            # Each entry is a script path string or a tuple whose first element is that path;
            # the other elements may be names, so read only the leading string literal.
            paths = []
            for item in getattr(node.value, "elts", []):
                first = item.elts[0] if isinstance(item, (ast.Tuple, ast.List)) else item
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    paths.append(first.value)
            return paths
    return []


SCRIPT_REF = re.compile(r"`(\d{2}_[\w]+/[\w.]+\.py)`")


def _experiment_scripts(suite: str) -> list[str]:
    base = ROOT / suite
    return sorted(
        p.relative_to(base).as_posix()
        for p in base.glob("[0-9][0-9]_*/*.py")
        if re.match(r"\d+_", p.name) and "legacy_reference" not in p.parts
    )


def check_index() -> list[str]:
    problems = []
    for suite in SUITES:
        index = ROOT / suite / "EXPERIMENT_INDEX.md"
        cited = set(SCRIPT_REF.findall(index.read_text(encoding="utf-8")))
        listed = set(_run_all_safe_scripts(suite))
        for script in sorted(cited):
            if not (ROOT / suite / script).exists():
                problems.append(f"{suite}: indexed script missing: {script}")
        for script in _experiment_scripts(suite):
            if suite == "paper_experiments" and script not in cited:
                problems.append(f"{suite}: script not in EXPERIMENT_INDEX.md: {script}")
            if script not in listed:
                problems.append(f"{suite}: script not in run_all_safe.py: {script}")
    return problems


def _experiment_ids(suite: str) -> set[str]:
    ids = set()
    for script in _experiment_scripts(suite):
        match = re.match(r"(\d+)_", Path(script).name)
        if match:
            ids.add(f"R{int(match.group(1))}")
    return ids


ARTIFACT = re.compile(r"`((?:tables|figures|results)/[\w./-]+)`")


def check_map() -> list[str]:
    problems = []
    for suite in SUITES:
        mapping = ROOT / suite / "MANUSCRIPT_MAP.md"
        text = mapping.read_text(encoding="utf-8")
        ids = _experiment_ids(suite)
        if suite == "paper_experiments":
            for rid in sorted(set(re.findall(r"\bR(\d+)\b", text)), key=int):
                if f"R{int(rid)}" not in ids:
                    problems.append(f"{suite}/MANUSCRIPT_MAP.md cites R{rid} with no script")
        for ref in ARTIFACT.findall(text):
            if any(c in ref for c in "*{<"):
                continue
            if not (ROOT / suite / ref).exists():
                problems.append(f"{suite}/MANUSCRIPT_MAP.md: missing {ref}")
    return problems


def check_metadata() -> list[str]:
    problems = []
    for suite in SUITES:
        json_dir = ROOT / suite / "results" / "json"
        with_meta = {
            p.name.split("_", 1)[0] for p in json_dir.glob("*.metadata.json")
        }  # experiment IDs that wrote a provenance companion
        for path in sorted(json_dir.glob("*.json")):
            name = path.name
            if not re.match(r"R\d", name) or name.endswith((".metadata.json", "_FAILURES.json")):
                continue
            if path.with_name(path.stem + ".metadata.json").exists():
                continue
            if name.split("_", 1)[0] in with_meta:
                continue  # auxiliary file of an experiment whose main result has provenance
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except ValueError as exc:
                problems.append(f"{_rel(path)}: unreadable JSON ({exc})")
                continue
            keys = set(data) if isinstance(data, dict) else set()
            if not keys & {"metadata", "environment", "provenance"}:
                problems.append(f"{_rel(path)}: no provenance")
    return problems


def check_evidence() -> list[str]:
    problems = []
    for suite in SUITES:
        status = (ROOT / suite / "STATUS.md").read_text(encoding="utf-8")
        results = ROOT / suite / "results"
        produced = {p.name for p in results.rglob("*") if p.is_file()}
        for line in status.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 4 or not re.fullmatch(r"R\d+", cells[0]):
                continue
            row = " ".join(cells).lower()
            if "passed" not in row and " y " not in f" {row} ":
                continue
            if not any(re.match(rf"{cells[0]}_", name) for name in produced):
                problems.append(f"{suite}/STATUS.md: {cells[0]} reported but no results file")
    return problems


def check_figures() -> list[str]:
    problems = []
    for suite in SUITES:
        ids = _experiment_ids(suite)
        for path in sorted((ROOT / suite / "figures").rglob("R*.*")):
            match = re.match(r"R(\d+)_", path.name)
            if match and f"R{int(match.group(1))}" not in ids and suite == "paper_experiments":
                problems.append(f"{_rel(path)}: no experiment script for R{match.group(1)}")
    return problems


def check_backends() -> list[str]:
    from photographiq.registry import BACKENDS

    problems = []
    text = (ROOT / "docs" / "user-guide" / "backends.md").read_text(encoding="utf-8")
    names = list(BACKENDS)
    for name in names:
        if f"`{name}`" not in text and name not in text:
            problems.append(f"docs/user-guide/backends.md does not document backend {name!r}")
    for doc in _docs():
        if _historical(doc):
            continue
        body = doc.read_text(encoding="utf-8")
        lines = body.splitlines()
        for number, line in enumerate(lines):
            mentions = 'backend="cudaq-dynamics"' in line or "backend='cudaq-dynamics'" in line
            context = " ".join(lines[number : number + 4])  # the line and the error it shows
            if mentions and not re.search(r"raise|Unknown|removed|ValueError", context):
                problems.append(f"{_rel(doc)}: advertises removed backend cudaq-dynamics")
    return problems


LABELS = (
    "implemented and validated",
    "implemented but experimental",
    "partially supported",
    "unsupported",
    "planned",
)


def check_labels() -> list[str]:
    """Every per-feature ``| Status | ... |`` row must start with one of the defined labels."""
    text = (ROOT / "docs" / "COMPLETE_FUNCTIONALITY.md").read_text(encoding="utf-8")
    problems = []
    rows = [
        line
        for line in text.splitlines()
        if line.startswith("| Status |") and not line.startswith("| Status | Meaning")
    ]
    for number, line in enumerate(rows, 1):
        value = line.split("|")[2].strip().strip("*").lower()
        if not value.startswith(LABELS):
            problems.append(
                f"COMPLETE_FUNCTIONALITY.md: Status row {number} unlabelled: {line[:80]}"
            )
    if not rows:
        problems.append("COMPLETE_FUNCTIONALITY.md: no Status rows found")
    return problems


LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def check_links() -> list[str]:
    problems = []
    for doc in _docs():
        if _historical(doc):
            continue
        for target in LINK.findall(doc.read_text(encoding="utf-8")):
            if re.match(r"[a-z]+:", target) or target.startswith("#"):
                continue
            path = target.split("#", 1)[0]
            if not path:
                continue
            if not (doc.parent / path).exists():
                problems.append(f"{_rel(doc)}: broken link {target}")
    return problems


def check_suites() -> list[str]:
    problems = []
    for suite in SUITES:
        runner = ROOT / suite / "run_all_safe.py"
        if not runner.exists():
            problems.append(f"{suite}: run_all_safe.py missing")
            continue
        for script in _run_all_safe_scripts(suite):
            if not (ROOT / suite / script).exists():
                problems.append(f"{suite}/run_all_safe.py lists missing {script}")
    return problems


HBAR = re.compile(r"(?:\\hbar|hbar|ħ)\s*(?:=|:=)\s*([0-9.]+)")


# A line naming another package's or paper's convention, or a conversion, is not a claim about
# PhotoGraphiQ's own convention.
FOREIGN = re.compile(
    r"source_hbar|Config\(|[Pp]iquasso|MFvL|Strawberry|literature|convert|units|uses|formulas"
)


def check_conventions() -> list[str]:
    problems = []
    for doc in _docs():
        if _historical(doc):
            continue
        for number, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            if FOREIGN.search(line):
                continue
            for value in HBAR.findall(line):
                if value.rstrip(".") not in {"2", "2.0"}:
                    problems.append(f"{_rel(doc)}:{number}: states hbar = {value}")
    return problems


CHECKS = {
    "api": check_api,
    "examples": check_examples,
    "index": check_index,
    "map": check_map,
    "metadata": check_metadata,
    "evidence": check_evidence,
    "figures": check_figures,
    "backends": check_backends,
    "labels": check_labels,
    "links": check_links,
    "suites": check_suites,
    "conventions": check_conventions,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", choices=sorted(CHECKS))
    parser.add_argument("--json", type=Path, help="write the findings as JSON")
    args = parser.parse_args()
    report = {}
    for name in args.only or CHECKS:
        try:
            report[name] = CHECKS[name]()
        except Exception as exc:  # a crashing check is a failure, not a pass
            report[name] = [f"check crashed: {type(exc).__name__}: {exc}"]
        status = "PASS" if not report[name] else f"FAIL ({len(report[name])})"
        print(f"{status:10s} {name}")
        for problem in report[name][:50]:
            print(f"           - {problem}")
        if len(report[name]) > 50:
            print(f"           ... {len(report[name]) - 50} more (see --json)")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 1 if any(report.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
