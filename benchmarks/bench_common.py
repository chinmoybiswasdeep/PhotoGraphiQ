"""Shared helpers for the PhotoGraphiQ benchmarks: environment metadata, timing, memory, output.

Timing protocol (``time_call``): the first call made by the timer is reported separately as
``first_call_s`` and is never a sample; a second call estimates the cost; then ``warmup``
untimed calls, then ``repeats`` samples, each the mean of ``inner`` back-to-back calls
(``inner`` chosen so a sample lasts at least ``min_sample_s``). Reported: median, IQR, min,
mean of the samples. Callers compute the correctness value with an earlier untimed call, so
for kernel/probability rows one-off JIT compilation (Piquasso's numba code) happens before
the timer and is in no reported number; ``first_call_s`` is a cold call only for ``*_setup``
rows.

Memory (``peak_python_bytes``): ``tracemalloc`` peak of one call. It sees allocations made
through Python's allocators, which include NumPy array buffers, but NOT memory allocated by
C/C++ code directly (PhotoGraphiQ's native Eigen/OpenMP kernels, Piquasso's C++ permanent and
torontonian, numba-compiled code using its own allocator), so it is a lower bound there.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import platform
import subprocess
import time
import tracemalloc
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def cpu_model() -> str:
    if platform.system() == "Windows":
        import winreg

        key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
            return str(winreg.QueryValueEx(handle, "ProcessorNameString")[0]).strip()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def git_state() -> dict:
    """Read-only: HEAD revision and whether the working tree differs from it."""

    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "--no-optional-locks", *args],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            ).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return ""

    commit = run("rev-parse", "HEAD")
    return {
        "commit": commit or None,
        "dirty": bool(run("status", "--porcelain")) if commit else None,
    }


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment(seed: int | None = None, **extra) -> dict:
    from photographiq import native

    threads = None
    if "sampling" in native.components():
        threads = native.require("sampling").max_threads()
    numba_threads = None
    try:
        import numba

        numba_threads = numba.config.NUMBA_NUM_THREADS
    except ImportError:
        pass
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            k: _version(k)
            for k in ("photographiq", "piquasso", "graphix", "numpy", "scipy", "numba")
        },
        "native_components": list(native.components()),
        "native_build": native.build_info(),
        "native_sampling_max_threads": threads,
        "numba_num_threads": numba_threads,
        "env_threads": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS",
            )
        },
        "precision": "complex128 / float64 throughout",
        "seed": seed,
        "git": git_state(),
        **extra,
    }


def time_call(
    fn: Callable[[], object],
    *,
    repeats: int = 7,
    warmup: int = 2,
    min_sample_s: float = 2e-3,
    budget_s: float = 5.0,
) -> dict:
    """Time ``fn`` (see module doc). ``repeats`` shrinks (to >= 3) if it would exceed ``budget_s``."""
    start = time.perf_counter()
    fn()
    first = time.perf_counter() - start
    # The cost estimate comes from a second call: a cold first call (JIT, caches) must
    # neither be a sample nor push a cheap case into the expensive branch.
    start = time.perf_counter()
    fn()
    single = max(time.perf_counter() - start, 1e-9)
    if single * (warmup + 4) > budget_s:
        # ponytail: expensive case - no warm-up, the second call is the first sample.
        samples = [single]
        while len(samples) < repeats and single * (len(samples) + 1) <= budget_s:
            start = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - start)
        return _stats(samples, first, 0, 1)
    for _ in range(warmup):
        fn()
    inner = max(1, min(10_000, math.ceil(min_sample_s / single)))
    repeats = max(3, min(repeats, int(budget_s / (single * inner)) or 3))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - start) / inner)
    return _stats(samples, first, warmup, inner)


def _stats(samples: list[float], first: float, warmup: int, inner: int) -> dict:
    q1, med, q3 = np.percentile(samples, [25, 50, 75])
    return {
        "median_s": float(med),
        "iqr_s": float(q3 - q1),
        "min_s": float(min(samples)),
        "mean_s": float(np.mean(samples)),
        "first_call_s": first,
        "warmup": warmup,
        "repeats": len(samples),
        "inner": inner,
    }


def peak_python_bytes(fn: Callable[[], object]) -> int:
    """``tracemalloc`` peak of one call of ``fn`` (Python/NumPy allocations only)."""
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def rel_err(value: complex, reference: complex) -> float:
    return float(abs(value - reference) / max(abs(reference), 1e-300))


def write_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def _json_default(obj):
    if isinstance(obj, np.integer | np.floating):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, complex):
        return [obj.real, obj.imag]
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


# Validated categorical palette (dataviz default, light surface), fixed order; colour follows
# the implementation everywhere, markers give a second, colour-independent encoding.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def series_style(index: int) -> dict:
    return {"color": PALETTE[index % 8], "marker": MARKERS[index % 8], "lw": 2, "ms": 6}


def new_axes(ncols: int = 1, width: float = 4.6, height: float = 3.4):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, ncols, figsize=(width * ncols, height), squeeze=False)
    for ax in axes[0]:
        ax.grid(True, which="major", color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return fig, list(axes[0])


def save_figure(fig, path_stem: Path) -> list[str]:
    """Save ``fig`` as ``<stem>.svg`` and ``<stem>.pdf``; return the paths."""
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("svg", "pdf"):
        p = path_stem.with_suffix(f".{ext}")
        fig.savefig(p, bbox_inches="tight")
        out.append(str(p))
    return out


def stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")
