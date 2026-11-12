"""Run local CI-equivalent checks and retain exit statuses without hiding failures."""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# PhotoGraphiQML-owned paths in the unified repository; the core has its own validator.
ML_PATHS = (
    "src/photographiqml",
    "tests/ml",
    "scripts/ml",
    "experiments/ml",
    "examples/ml",
    "paper_experiments_ml",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".validation" / "ml")
    output = parser.parse_args().output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, COVERAGE_FILE=str((output / ".coverage").resolve()))
    commands = [
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/ml",
            "--cov=photographiqml",
            "--cov-fail-under=90",
            f"--cov-report=json:{output / 'coverage.json'}",
            "--cov-report=term-missing",
        ],
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--exclude",
            "paper_experiments_ml/PhotoGraphiQML_Manuscript_Experiments.ipynb",
            *ML_PATHS,
        ],
        [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--check",
            "--exclude",
            "paper_experiments_ml/PhotoGraphiQML_Manuscript_Experiments.ipynb",
            *ML_PATHS,
        ],
        [sys.executable, "-m", "mypy", "src/photographiqml"],
        [sys.executable, "-m", "build", "--outdir", str(output / "dist")],
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--strict",
            "--site-dir",
            str((output / "site").resolve()),
        ],
        [sys.executable, "scripts/ml/execute_tutorials.py"],
    ]
    results = []
    for command in commands:
        process = subprocess.run(command, check=False, env=environment, cwd=ROOT)
        results.append({"command": command[1:], "returncode": process.returncode})
    (output / "validation-results.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "checks": results,
            },
            indent=2,
        )
    )
    return int(any(result["returncode"] for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
