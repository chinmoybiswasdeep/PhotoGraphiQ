"""Shared utilities for the PhotoGraphiQML manuscript experiment suite.

Every experiment script imports this module (via sys.path insertion of this
directory) rather than reimplementing CSV/JSON/plot/timing boilerplate.
Nothing here calls into photographiqml internals beyond its public API, and
nothing here decides a scientific result for an experiment -- it only
standardizes bookkeeping, tolerances and presentation so results are
comparable and reproducible across the suite.

Visual conventions (rcParams, hex color roles, PDF+PNG output policy, the
CSV/JSON/metadata "save triple") mirror the sibling PhotoGraphiQ
``manuscript-experiments`` suite so figures and result files read as one
system across both papers, while oracle classification (A-E, N/A), bootstrap
confidence intervals and declared-in-advance tolerances are added because the
PhotoGraphiQML suite spans a physical/statistical layer PhotoGraphiQ's core
suite does not.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import re
import sys
import textwrap
import time
from math import comb
from pathlib import Path

import numpy as np

_EXPERIMENT_ROOT = Path(__file__).resolve().parent
# Execute against this checkout even if an unrelated ``photographiqml``
# distribution is installed in the active interpreter (a common research
# workstation configuration).  This is intentionally before any package
# import, and is recorded by each result's provenance metadata.
sys.path.insert(0, str(_EXPERIMENT_ROOT.parent / "src"))
sys.path.insert(0, str(_EXPERIMENT_ROOT))
import json as _json  # noqa: E402

import metadata  # noqa: E402
import publication  # noqa: E402

ROOT = _EXPERIMENT_ROOT
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

ORACLE_CLASSES = metadata.ORACLE_CLASSES
CANONICAL_R2_TOLERANCE = float(10 * np.finfo(float).eps)

# ---------------------------------------------------------------------------
# Reference colors: consistent role -> color across every figure in the suite.
# The first three reuse PhotoGraphiQ's own analytic/measured/independent hues
# so the two papers' figures read as one visual system; the rest extend that
# palette for roles PhotoGraphiQ's core suite did not need.
# ---------------------------------------------------------------------------
COLORS = {
    "analytic": "#24677b",  # analytic/closed-form oracle (class A)
    "photographiqml": "#c66d27",  # system under test (logical/physical MuTA)
    "mentpy": "#627a36",  # independent external logical reference (class B)
    "photographiq": "#5b4b8a",  # independently assembled public Pattern (class B/C)
    "piquasso": "#b23a48",  # raw physical simulation reference (class B/C)
    "classical_baseline": "#8c8c8c",  # logistic/RBF-SVM baselines
    "unsupported": "#bdbdbd",  # unsupported / rejected / not-applicable
    "acceptance": "#1a1a1a",  # acceptance-threshold reference lines
}

_STYLE_APPLIED = False


# Manuscript style, identical in paper_experiments/common.py: Times New Roman text (with
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


def panel_label(ax, label: str) -> None:
    """Place a consistent (a)/(b)/... panel label at the top-left of an axes."""
    ax.text(
        -0.12,
        1.05,
        f"({label})",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def save_figure(fig, name: str, *, tight: bool = True, svg: bool = True) -> dict:
    """Save a figure as vector PDF, 600-dpi PNG, and (if compact) SVG."""
    finish_figure(fig)
    if tight:
        fig.tight_layout(pad=0.4)
    pdf_path, png_path = FIG_PDF / f"{name}.pdf", FIG_PNG / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    paths = {"pdf": pdf_path, "png": png_path}
    if svg:
        svg_path = FIG_SVG / f"{name}.svg"
        fig.savefig(svg_path, bbox_inches="tight")
        # Matplotlib opens SVG output in platform text mode.  Normalize it
        # before hashing so Git's ``eol=lf`` checkout policy cannot change the
        # bytes recorded by the evidence manifest on Windows.
        publication.atomic_write_text(svg_path, svg_path.read_text(encoding="utf-8"))
        paths["svg"] = svg_path
    return paths


# ---------------------------------------------------------------------------
# Deterministic randomness
# ---------------------------------------------------------------------------
def rng(seed: int) -> np.random.Generator:
    """Return a NumPy Generator seeded deterministically for one experiment."""
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------
def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def save_csv(rows: list[dict], name: str) -> Path:
    """Write a list of flat dict rows as CSV under results/csv/<name>.csv."""
    path = CSV_DIR / f"{name}.csv"
    if not rows:
        publication.atomic_write_text(path, "")
        return path
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    publication.atomic_write_text(path, buffer.getvalue())
    return path


def save_json(obj, name: str) -> Path:
    """Write an object as pretty JSON under results/json/<name>.json."""
    path = JSON_DIR / f"{name}.json"
    publication.atomic_write_text(path, _json.dumps(obj, indent=2, default=_json_default) + "\n")
    return path


def save_raw(text: str, name: str) -> Path:
    """Write raw text (e.g. serialized patterns/logs) under results/raw/<name>."""
    path = RAW_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    publication.atomic_write_text(path, text)
    return path


def write_metadata(name: str, **extra) -> Path:
    """Write a companion environment/provenance JSON file for one experiment."""
    payload = metadata.collect()
    payload.update(extra)
    return save_json(payload, f"{name}.metadata")


def save_result(
    rows: list[dict], name: str, *, extra: dict | None = None, meta_extra: dict | None = None
) -> None:
    """Save evidence with separate execution, structural, and scientific status.

    ``status`` remains a compatibility alias for the structural contract only;
    it must never be read as an ML-performance claim for a descriptive study.
    """
    extra = dict(extra or {})
    structural = extra.get("structural_status", extra.get("status", "unknown"))
    descriptive = extra.get("oracle_class") == "N/A"
    outcome = extra.get("scientific_outcome")
    if outcome is None:
        outcome = (
            "negative"
            if name.startswith("R48_")
            else "descriptive"
            if descriptive
            else "positive"
            if structural == "pass"
            else "inconclusive"
        )
    extra.setdefault("execution_status", "completed")
    extra.setdefault("structural_status", structural)
    extra.setdefault("scientific_outcome", outcome)
    extra.setdefault(
        "claim_supported",
        "structural contract only" if descriptive else "declared experiment contract",
    )
    extra.setdefault(
        "claim_not_supported",
        "no scientific performance conclusion"
        if descriptive
        else "no claim beyond declared protocol",
    )
    meta_extra = dict(meta_extra or {})
    meta_extra.update(
        {
            "execution_status": extra["execution_status"],
            "structural_status": extra["structural_status"],
            "scientific_outcome": extra["scientific_outcome"],
        }
    )
    if "seeds" in extra:
        meta_extra.setdefault("seed_set", extra["seeds"])
        meta_extra.setdefault("n_repetitions", len(extra["seeds"]))
    if "n_repetitions" in extra:
        meta_extra.setdefault("n_repetitions", extra["n_repetitions"])
    if "uncertainty_method" in extra:
        meta_extra.setdefault("uncertainty_method", extra["uncertainty_method"])
    save_csv(rows, name)
    save_json({"rows": rows, **extra}, name)
    write_metadata(name, **meta_extra)


# ---------------------------------------------------------------------------
# Timing/benchmarking
# ---------------------------------------------------------------------------
def benchmark(fn, *, warmup: int = 2, repeats: int = 11) -> dict:
    """Time fn() with warm-up runs and repeated measurements.

    Returns median, IQR and standard deviation in seconds, plus raw per-repeat
    timings and the last return value. Uses time.perf_counter; never times
    module imports (those must have already happened before calling this).
    """
    result = None
    for _ in range(warmup):
        result = fn()
    raw = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        raw.append(time.perf_counter() - start)
    raw = np.asarray(raw)
    return {
        "median_seconds": float(np.median(raw)),
        "std_seconds": float(np.std(raw, ddof=1)) if len(raw) > 1 else 0.0,
        "iqr_seconds": float(np.percentile(raw, 75) - np.percentile(raw, 25)),
        "min_seconds": float(np.min(raw)),
        "max_seconds": float(np.max(raw)),
        "repeats": repeats,
        "warmup": warmup,
        "raw_seconds": raw.tolist(),
        "result": result,
    }


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------
def frobenius_error(actual, expected) -> float:
    return float(np.linalg.norm(np.asarray(actual) - np.asarray(expected)))


def relative_error(actual, expected, *, floor: float = 1e-300) -> float:
    denom = max(float(np.linalg.norm(np.asarray(expected))), floor)
    return frobenius_error(actual, expected) / denom


def safe_log10(value: float, *, floor: float = 1e-300) -> float:
    return float(np.log10(max(float(value), floor)))


def is_finite_real(value) -> bool:
    return bool(np.isfinite(value)) and np.isreal(value)


def declare_tolerance(
    scale: float, *, conditioning: float = 1.0, truncation: float = 0.0, safety_factor: float = 10.0
) -> float:
    """Derive an exact-comparison tolerance before running a sweep.

    tol = safety_factor * max(machine_eps * conditioning * scale, truncation).
    Call this and record the return value in the experiment's JSON/metadata
    *before* evaluating the acceptance condition, never after inspecting
    whether a looser tolerance would make a borderline case pass.
    """
    eps = np.finfo(float).eps
    return float(safety_factor * max(eps * conditioning * scale, truncation))


# ---------------------------------------------------------------------------
# Statistics: bootstrap CIs for multi-seed / multi-split studies
# ---------------------------------------------------------------------------
def bootstrap_ci(
    values, *, statistic, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict:
    """Percentile bootstrap CI for an explicitly selected 1-D statistic.

    ``statistic`` is deliberately required: a mean CI must not be presented
    as a median CI (or conversely).
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {
            "point": None,
            "low": None,
            "high": None,
            "n_boot": n_boot,
            "n": 0,
            "statistic": getattr(statistic, "__name__", str(statistic)),
        }
    generator = np.random.default_rng(seed)
    idx = generator.integers(0, len(values), size=(n_boot, len(values)))
    boots = statistic(values[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(statistic(values)),
        "low": float(lo),
        "high": float(hi),
        "n_boot": n_boot,
        "alpha": alpha,
        "n": int(len(values)),
        "statistic": getattr(statistic, "__name__", str(statistic)),
    }


def paired_difference_ci(a, b, **kwargs) -> dict:
    """Bootstrap CI on the paired difference a - b (identical seeds/splits)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Paired comparison requires matching shapes")
    return bootstrap_ci(a - b, **kwargs)


# ---------------------------------------------------------------------------
# Fock dimension / memory estimation (performance panels: R13, R36, R42, R47)
# ---------------------------------------------------------------------------
def fock_dimension(modes: int, cutoff: int) -> int:
    return comb(modes + cutoff - 1, modes)


def vector_bytes(dimension: int) -> int:
    return 16 * dimension  # complex128


def density_bytes(dimension: int) -> int:
    return 16 * dimension**2  # dense complex128 density matrix


# ---------------------------------------------------------------------------
# Optional dependencies / safe execution
# ---------------------------------------------------------------------------
def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


HAS_MENTPY = has_module("mentpy")
HAS_PIQUASSO = has_module("piquasso")
HAS_SKLEARN = has_module("sklearn")


class SkippedExperiment(Exception):
    """Raised (and caught by run_all_safe.py / the script's __main__ guard) when
    an experiment cannot proceed in this environment."""


def require_module(name: str, reason: str = "") -> None:
    if not has_module(name):
        raise SkippedExperiment(f"optional dependency {name!r} is not installed. {reason}".strip())


def safe_cutoff_run(fn, *, label: str = "run"):
    """Run fn() and translate MemoryError/expected resource ValueErrors into a
    structured dict. Returns {"ok": True, "value": ...} on success or
    {"ok": False, "error": str, "kind": type name} on an anticipated failure
    mode (resource/allocation guards). Unexpected exception types re-raise.
    """
    try:
        return {"ok": True, "value": fn()}
    except (MemoryError, ValueError, ArithmeticError, NotImplementedError) as exc:
        return {"ok": False, "error": str(exc), "kind": type(exc).__name__, "label": label}


# ---------------------------------------------------------------------------
# Console summaries
# ---------------------------------------------------------------------------
def print_summary(title: str, **fields) -> None:
    """Print a short, greppable one-block summary for a finished experiment.
    Full evidence belongs in the saved CSV/JSON/figures, not stdout."""
    print(f"\n=== {title} ===")
    for key, value in fields.items():
        print(f"  {key}: {value}")
