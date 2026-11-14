"""Shared utilities for the PhotoGraphiQ manuscript experiment suite.

Every experiment script imports this module (via ``sys.path`` insertion of
this directory) rather than reimplementing CSV/JSON/plot/timing boilerplate.
Nothing here calls into ``photographiq`` internals beyond its public API.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np

import metadata

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CSV_DIR = RESULTS / "csv"
JSON_DIR = RESULTS / "json"
RAW_DIR = RESULTS / "raw"
LOG_DIR = RESULTS / "logs"
FIG_PDF = ROOT / "figures" / "pdf"
FIG_PNG = ROOT / "figures" / "png"
TABLES = ROOT / "tables"

for _d in (CSV_DIR, JSON_DIR, RAW_DIR, LOG_DIR, FIG_PDF, FIG_PNG, TABLES):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Deterministic randomness
# --------------------------------------------------------------------------- #


def rng(seed: int) -> np.random.Generator:
    """Return a NumPy Generator seeded deterministically for one experiment."""
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# Saving results
# --------------------------------------------------------------------------- #


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, complex):
        return {"re": value.real, "im": value.imag}
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def save_csv(rows: list[dict], name: str) -> Path:
    """Write a list of flat dict rows as CSV under results/csv/<name>.csv."""
    path = CSV_DIR / f"{name}.csv"
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_json(obj, name: str) -> Path:
    """Write an object as pretty JSON under results/json/<name>.json."""
    path = JSON_DIR / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    return path


def save_raw(text: str, name: str) -> Path:
    """Write raw text (e.g. serialized patterns) under results/raw/<name>."""
    path = RAW_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_metadata(name: str) -> Path:
    """Write a companion environment/provenance JSON file for one experiment."""
    return save_json(metadata.collect(), f"{name}.metadata")


def save_result(rows: list[dict], name: str, *, extra: dict | None = None) -> None:
    """Save CSV rows, a JSON mirror and a metadata companion for one experiment."""
    save_csv(rows, name)
    save_json({"rows": rows, **(extra or {})}, name)
    write_metadata(name)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

_STYLE_APPLIED = False


def setup_style():
    """Apply a consistent, manuscript-friendly matplotlib style once."""
    global _STYLE_APPLIED
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not _STYLE_APPLIED:
        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.titlesize": 11,
                "axes.labelsize": 10,
                "legend.fontsize": 8.5,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "figure.dpi": 120,
                "savefig.dpi": 300,
                "lines.linewidth": 1.6,
                "lines.markersize": 5,
            }
        )
        _STYLE_APPLIED = True
    return plt


def save_figure(fig, name: str, *, tight: bool = True) -> tuple[Path, Path]:
    """Save a figure as both vector PDF and high-resolution PNG; return paths."""
    if tight:
        fig.tight_layout()
    pdf_path = FIG_PDF / f"{name}.pdf"
    png_path = FIG_PNG / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    return pdf_path, png_path


# --------------------------------------------------------------------------- #
# Timing / benchmarking
# --------------------------------------------------------------------------- #


def benchmark(fn, *, warmup: int = 2, repeats: int = 11) -> dict:
    """Time ``fn()`` with warm-up runs and repeated measurements.

    Returns median, IQR and standard deviation in seconds, plus the raw
    per-repeat timings and the last return value for downstream inspection.
    Uses ``time.perf_counter``, never wall-clock import time.
    """
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be >=0 and repeats must be >=1")
    result = None
    for _ in range(warmup):
        result = fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    times = np.asarray(times)
    q1, median, q3 = np.percentile(times, [25, 50, 75])
    return {
        "median_seconds": float(median),
        "std_seconds": float(times.std(ddof=1)) if repeats > 1 else 0.0,
        "iqr_seconds": float(q3 - q1),
        "min_seconds": float(times.min()),
        "max_seconds": float(times.max()),
        "repeats": repeats,
        "warmup": warmup,
        "raw_seconds": times.tolist(),
        "result": result,
    }


# --------------------------------------------------------------------------- #
# Numerical helpers
# --------------------------------------------------------------------------- #


def frobenius_error(actual, expected) -> float:
    """Absolute Frobenius-norm error between two same-shaped arrays."""
    return float(np.linalg.norm(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float)))


def relative_error(actual, expected, *, floor: float = 1e-300) -> float:
    """Relative error ||actual-expected|| / max(||expected||, floor)."""
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    denom = max(float(np.linalg.norm(expected)), floor)
    return float(np.linalg.norm(actual - expected) / denom)


def safe_log10(value: float, *, floor: float = 1e-300) -> float:
    """log10(max(value, floor)); avoids -inf/NaN for exact-zero errors."""
    return float(np.log10(max(float(value), floor)))


def is_finite_real(value) -> bool:
    return isinstance(value, (int, float, np.floating, np.integer)) and math.isfinite(float(value))


# --------------------------------------------------------------------------- #
# Fock dimension / memory estimation
# --------------------------------------------------------------------------- #


def fock_dimension(modes: int, cutoff: int) -> int:
    """Total-photon Fock dimension binom(modes+cutoff-1, modes)."""
    from math import comb

    return comb(modes + cutoff - 1, modes)


def vector_bytes(dimension: int) -> int:
    """Bytes for one dense complex128 pure state vector."""
    return 16 * dimension


def density_bytes(dimension: int) -> int:
    """Bytes for one dense complex128 density matrix (rough dense upper estimate)."""
    return 16 * dimension * dimension


# --------------------------------------------------------------------------- #
# Optional dependencies / safe execution
# --------------------------------------------------------------------------- #


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


HAS_GRAPHIX = has_module("graphix")
HAS_JAX = has_module("jax")


class SkippedExperiment(Exception):
    """Raised (and caught by the runner) when an experiment cannot proceed here."""


def require_module(name: str, reason: str = ""):
    if not has_module(name):
        raise SkippedExperiment(f"optional dependency {name!r} is not installed. {reason}".strip())


def safe_cutoff_run(fn, *, label: str = "run"):
    """Run fn() and translate MemoryError/cutoff ValueErrors into a structured dict.

    Returns {"ok": True, "value": ...} on success or
    {"ok": False, "error": str, "kind": type name} on an expected failure mode.
    Unexpected exception types are re-raised, since silently swallowing an
    unrelated bug would hide a real defect rather than a resource guard.
    """
    try:
        return {"ok": True, "value": fn()}
    except MemoryError as exc:
        return {"ok": False, "error": str(exc), "kind": "MemoryError", "label": label}
    except ValueError as exc:
        message = str(exc)
        if "cutoff" in message.lower() or "dimension" in message.lower() or "norm" in message.lower():
            return {"ok": False, "error": message, "kind": "ValueError", "label": label}
        raise


# --------------------------------------------------------------------------- #
# Canonical ordering (for Graphix / graph comparisons)
# --------------------------------------------------------------------------- #


def canonical_edges(graph) -> set[tuple]:
    """Undirected edge set with a stable, label-repr-based orientation."""
    return {tuple(sorted((u, v), key=repr)) for u, v in graph.edges()}


def canonical_node_order(graph) -> tuple:
    """Deterministic node ordering (by repr) usable across two graph objects."""
    return tuple(sorted(graph.nodes(), key=repr))


# --------------------------------------------------------------------------- #
# Console summaries
# --------------------------------------------------------------------------- #


def print_summary(title: str, **fields):
    """Print a short, greppable one-block summary for a finished experiment."""
    print(f"\n=== {title} ===")
    for key, value in fields.items():
        print(f"  {key}: {value}")
