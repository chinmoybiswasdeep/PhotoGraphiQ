"""Environment/provenance metadata recorded alongside every experiment's output.

Every script calls ``metadata.collect()`` once and writes the result as a
companion ``<name>.metadata.json`` file (or embeds it under a ``"metadata"``
key) next to its CSV/JSON results. This is descriptive provenance only: it
does not certify reproducibility on a different machine.
"""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_PACKAGES = (
    "numpy",
    "scipy",
    "networkx",
    "matplotlib",
    "piquasso",
    "photographiq",
    "graphix",
    "jax",
)


def _package_versions():
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _cpu_model():
    try:
        if platform.system() == "Windows":
            return platform.processor() or None
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        return platform.processor() or None
    except Exception:
        return None


def collect() -> dict:
    """Return a JSON-serializable snapshot of the environment used for a run."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "packages": _package_versions(),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(collect(), indent=2))
