"""Run all release gates with separate logs, exit codes and environment provenance.

Run with each interpreter: python experiments/validate_release.py --output DIR.
No failed step skips later checks. Artifacts identify the exact source snapshot.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256()
    for path in sorted(
        [
            *root.glob("src/**/*.py"),
            *root.glob("tests/**/*.py"),
            root / "pyproject.toml",
            Path(__file__).resolve(),
            root / ".github/workflows/ci.yml",
        ]
    ):
        source_hash.update(path.relative_to(root).as_posix().encode())
        source_hash.update(path.read_bytes().replace(b"\r\n", b"\n"))
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "source_sha256": source_hash.hexdigest(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "checks": {},
    }
    env = os.environ.copy()
    env["COVERAGE_FILE"] = str(output / ".coverage")
    env["NUMBA_CACHE_DIR"] = str(output / "numba-cache")
    steps = {
        "pytest": [
            "pytest",
            "--cov=photographiq",
            f"--cov-report=json:{output / 'coverage.json'}",
            f"--cov-report=xml:{output / 'coverage.xml'}",
            f"--junitxml={output / 'tests.xml'}",
            "-q",
        ],
        "ruff-check": ["ruff", "check", "src", "tests", "examples", "experiments", "scripts"],
        "ruff-format": [
            "ruff",
            "format",
            "--check",
            "src",
            "tests",
            "examples",
            "experiments",
            "scripts",
        ],
        "mypy": ["mypy", "src/photographiq"],
        "build": ["build", "--outdir", str(output / "dist")],
        "physics-evidence": [
            "experiments.hardening_evidence",
            "--output",
            str(output / "physics.json"),
        ],
    }
    for name, arguments in steps.items():
        print(f"Python {platform.python_version()}: {name}", flush=True)
        with (output / f"{name}.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                [sys.executable, "-m", *arguments],
                cwd=root,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        report["checks"][name] = {"exit_code": completed.returncode, "log": f"{name}.log"}
        (output / "environment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    failed = [name for name, check in report["checks"].items() if check["exit_code"]]
    print("Failed: " + ", ".join(failed) if failed else "All release gates passed", flush=True)
    return bool(failed)


if __name__ == "__main__":
    sys.exit(main())
