"""Environment/provenance metadata recorded alongside every experiment's output.

Every script calls ``metadata.collect()`` once and writes the result as a
companion ``<name>.metadata.json`` file (or embeds it under a ``"metadata"``
key) next to its CSV/JSON results. This is descriptive provenance only: it
does not certify reproducibility on a different machine.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
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
    "mpmath",
    "numba",
    "cupy",
    "mentpy",
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


_FINGERPRINT_DIRS = ("src", "cpp", "paper_experiments")
_FINGERPRINT_SUFFIXES = {".py", ".cpp", ".hpp", ".h", ".txt", ".toml"}


def source_fingerprint() -> str:
    """SHA-256 over the source files that determine numerical results.

    ``git_commit`` alone does not identify the code when the working tree is
    dirty; this digest covers the actual bytes of ``src/``, ``cpp/`` and the
    experiment scripts (results/figures excluded), so a result can be traced
    to the exact code that produced it.
    """
    digest = hashlib.sha256()
    for top in _FINGERPRINT_DIRS:
        base = REPO_ROOT / top
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if (
                path.is_file()
                and path.suffix in _FINGERPRINT_SUFFIXES
                and "__pycache__" not in path.parts
                and "results" not in path.relative_to(base).parts
            ):
                digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
                digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _gpu_model():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return out.strip() or None
    except Exception:
        return None


def _native_info():
    try:
        from photographiq import native

        return {"available": bool(native.AVAILABLE), "components": list(native.components())}
    except Exception as exc:  # report, do not fail provenance collection
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def collect() -> dict:
    """Return a JSON-serializable snapshot of the environment used for a run.

    ``git_commit`` is the checked-out HEAD. When ``git_dirty`` is true the
    results were produced by uncommitted changes on top of that commit, and
    ``source_sha256`` (see :func:`source_fingerprint`) identifies the code.
    """
    dirty = bool(_git("status", "--porcelain"))
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": dirty,
        "git_note": (
            "uncommitted working-tree changes on top of git_commit; see source_sha256"
            if dirty
            else "clean checkout of git_commit"
        ),
        "source_sha256": source_fingerprint(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "gpu": _gpu_model(),
        "native": _native_info(),
        "packages": _package_versions(),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(collect(), indent=2))
