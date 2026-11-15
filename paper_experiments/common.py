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
import re
import textwrap
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
FIG_SVG = ROOT / "figures" / "svg"
TABLES = ROOT / "tables"

for _d in (CSV_DIR, JSON_DIR, RAW_DIR, LOG_DIR, FIG_PDF, FIG_PNG, FIG_SVG, TABLES):
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

# Manuscript style, identical in paper_experiments_ml/common.py: Times New Roman text (with
# metric-compatible fallbacks for machines without it), STIX math, 8 pt body text at the final
# printed size, TrueType fonts embedded in PDF/PS (journal requirement), text kept as text in SVG.
MANUSCRIPT_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.titleweight": "normal",
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.6,
    "ytick.minor.size": 1.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "patch.linewidth": 0.6,
    "errorbar.capsize": 2,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}
SINGLE_COLUMN_IN, DOUBLE_COLUMN_IN, MAX_HEIGHT_IN, MIN_HEIGHT_IN, MIN_FONT_PT = 3.4, 7.0, 9.0, 2.4, 6.0
_RUN_ID_PREFIX = re.compile(r"^\s*R(?:\d+|_PERF)[a-z]?\s*[:\u2013\u2014-]\s*")
_STATUS_FLAG = re.compile(r"status\s*=\s*[\w-]+(?:\s*,\s*)?")
_EMPTY_PARENS = re.compile(r"\s*\(\s*\)")
_PANEL_TAG = re.compile(r"^\(?[a-z]\)")


def setup_style():
    """Apply the manuscript matplotlib style (``MANUSCRIPT_RC``) once."""
    global _STYLE_APPLIED
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not _STYLE_APPLIED:
        plt.rcParams.update(MANUSCRIPT_RC)
        _STYLE_APPLIED = True
    return plt


def _words(text: str) -> str:
    """Print code identifiers as words (``n_qubits`` -> ``n qubits``, ``->`` -> arrow); math
    text is untouched."""
    return text if "$" in text else re.sub(r"(?<=[a-z0-9])_(?=[a-z0-9])", " ", text).replace("->", "→")


def _sentence_case(text: str) -> str:
    """Capitalize a leading plain lowercase word of 3+ letters (variables like ``q`` stay)."""
    word = re.split(r"[ -]", text, maxsplit=1)[0]
    return text[0].upper() + text[1:] if len(word) >= 3 and word.isalpha() and word.islower() else text


def _clean_title(text: str) -> str:
    """Drop run-ID prefixes and pass/fail flags: captions, not titles, carry those."""
    text = _EMPTY_PARENS.sub("", _STATUS_FLAG.sub("", _RUN_ID_PREFIX.sub("", text))).strip()
    return text if _PANEL_TAG.match(text) else _sentence_case(_words(text))


def _crowded(ax) -> bool:
    """Long tick labels, many annotations or a long legend: a single plot needing two columns."""
    legend = ax.get_legend()
    labels = [t.get_text() for t in ax.get_xticklabels() + ax.get_yticklabels()]
    return (max(map(len, labels), default=0) > 14 or len(ax.texts) > 8
            or (legend is not None and len(legend.get_texts()) > 6))


def _reflow(fig, panels):
    """Re-grid a crowded single row of panels; returns the new printed height (in) or None.

    Four or more panels become two columns; two or three panels whose legends do not fit
    (more than 6 entries) become one column with those legends beside their panel. Figures
    with twin axes or colour bars are left as drawn.
    """
    from math import ceil

    from matplotlib.gridspec import GridSpec

    specs = [ax.get_subplotspec() for ax in panels]
    if len(panels) < 2 or len(fig.axes) != len(panels) or None in specs:
        return None
    if any(spec.get_gridspec().nrows != 1 for spec in specs):
        return None
    crowded = [ax for ax in panels if ax.get_legend() is not None and len(ax.get_legend().get_texts()) > 6]
    if len(panels) >= 4:
        rows, cols = ceil(len(panels) / 2), 2
    elif crowded:
        rows, cols = len(panels), 1
    else:
        return None
    grid = GridSpec(rows, cols, figure=fig)
    for index, ax in enumerate(panels):
        ax.set_subplotspec(grid[index // cols, index % cols])
    if cols == 1:
        for ax in crowded:
            legend = ax.get_legend()
            handles = getattr(legend, "legend_handles", None) or legend.legendHandles
            labels = [text.get_text() for text in legend.get_texts()]
            ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6,
                      ncol=1 if len(labels) <= 12 else 2, frameon=False)
    return rows * (2.4 if cols == 2 else 2.0)


def finish_figure(fig) -> None:
    """Bring a figure to final printed size and manuscript conventions (idempotent).

    Single plots (and originals <= 4.5 in wide) become one journal column wide, multi-panel
    figures and diagrams two, height scaling with width; the suptitle is dropped (the caption
    carries it); panel titles lose run IDs and status flags; multi-panel figures without
    (a)/(b) tags get them; code identifiers print as words, titles and axis labels in
    sentence case, titles wrapped to the panel width; every label is set in the serif face (graph-drawing helpers such
    as networkx hard-code sans-serif) and no text falls below 6 pt.
    """
    from matplotlib.text import Text
    from matplotlib.ticker import FixedFormatter, FuncFormatter

    panels, seen = [], set()
    for ax in fig.axes:  # twinx/twiny axes share their host's position: one panel
        bounds = tuple(round(b, 6) for b in ax.get_position().bounds)
        if ax.get_visible() and ax.get_label() != "<colorbar>" and bounds not in seen:
            panels.append(ax)
            seen.add(bounds)
    reflowed_height = _reflow(fig, panels)
    width, height = fig.get_size_inches()
    single_plot = len(panels) == 1 and panels[0].axison and width <= 8 and not _crowded(panels[0])
    target = SINGLE_COLUMN_IN if width <= 4.5 or single_plot else DOUBLE_COLUMN_IN
    new_height = reflowed_height or height * target / width
    fig.set_size_inches(target, min(max(new_height, MIN_HEIGHT_IN), MAX_HEIGHT_IN))
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")
    for ax in panels:
        for title in (ax.title, ax._left_title, ax._right_title):
            title.set_text(_clean_title(title.get_text()))
        for axis in (ax.xaxis, ax.yaxis):
            axis.label.set_text(_sentence_case(_words(axis.label.get_text())))
            formatter = axis.get_major_formatter()  # label lists: set_ticklabels or categorical axes
            labelled = isinstance(formatter, (FixedFormatter, FuncFormatter))
            if (labelled or type(formatter).__name__ == "StrCategoryFormatter") and not getattr(
                formatter, "_as_words", False
            ):
                words = FuncFormatter(lambda x, pos, f=formatter: _words(str(f(x, pos))))
                words._as_words = True
                axis.set_major_formatter(words)
    texts = [t for t in fig.findobj(Text) if t.get_text()]
    if len(panels) > 1 and not any(_PANEL_TAG.match(t.get_text().strip()) for t in texts):
        for index, ax in enumerate(panels):
            tag = f"({chr(ord('a') + index)})"
            if ax.title.get_text():
                ax.title.set_text(f"{tag} {ax.title.get_text()}")
            else:
                ax.set_title(tag, loc="left", fontweight="bold")
    for ax in panels:  # wrap titles to the panel width (~0.5 em per character)
        title = ax.title.get_text()
        chars = int(ax.get_position().width * fig.get_figwidth() * 144 / ax.title.get_fontsize())
        if len(title) > chars > 10 and "$" not in title:
            ax.title.set_text(textwrap.fill(title, chars))
    for text in texts:
        text.set_text(_words(text.get_text()))
        text.set_fontfamily("serif")
        if text.get_fontsize() < MIN_FONT_PT:
            text.set_fontsize(MIN_FONT_PT)


def save_figure(fig, name: str, *, tight: bool = True, svg: bool = True) -> tuple[Path, Path]:
    """Save a figure as vector PDF (and SVG) plus high-resolution PNG.

    Returns ``(pdf_path, png_path)``; the SVG, when written, sits in
    ``figures/svg/``. PDF/SVG are the primary manuscript formats.
    """
    finish_figure(fig)
    if tight:
        fig.tight_layout(pad=0.4)
    pdf_path = FIG_PDF / f"{name}.pdf"
    png_path = FIG_PNG / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=600)
    if svg:
        fig.savefig(FIG_SVG / f"{name}.svg", bbox_inches="tight")
    return pdf_path, png_path


def save_caption(name: str, text: str) -> Path:
    """Write ``figures/captions/<name>.caption.md`` next to the figure files."""
    path = ROOT / "figures" / "captions" / f"{name}.caption.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


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


# --------------------------------------------------------------------------- #
# Timing status
# --------------------------------------------------------------------------- #


def timing_final() -> bool:
    """True when the run was declared to be on an otherwise idle machine (PGQ_IDLE_MACHINE=1)."""
    import os

    return os.environ.get("PGQ_IDLE_MACHINE") == "1"


def timing_label() -> str:
    """Status string for timing results: final only on a declared idle machine."""
    if timing_final():
        return "final (idle machine declared via PGQ_IDLE_MACHINE=1)"
    return "PROVISIONAL (machine not declared idle; rerun with PGQ_IDLE_MACHINE=1)"
