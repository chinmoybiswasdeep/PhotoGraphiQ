"""Helper namespaces inlined into the experiment cells of the Colab manuscript notebook.

``scripts/build_colab_notebook.py`` copies into every experiment cell only the members of these
classes that the experiment uses (transitively), naming the class ``common`` (or ``metadata``) so
the experiment code runs unchanged and each cell stays self-contained. They reproduce the public
helpers of ``paper_experiments/common.py`` and ``paper_experiments_ml/common.py`` with three
notebook changes: outputs go to ``manuscript_outputs/<suite>/``, figures are also shown inline,
and provenance is a compact environment record instead of the suites' Git fingerprinting.

Rules for members (the builder relies on them): a method refers to another member only as
``common.<name>`` (``metadata`` members may use ``common.<name>`` too); a class-body constant
may use earlier constants by bare name; imports needed by a method happen inside the method,
except the fixed cell header (``importlib.util``, ``json``, ``re``, ``textwrap``, ``time``,
``Path``, ``numpy as np``). ``Shared`` members are used by both suites, a suite class overrides
a ``Shared`` member of the same name.
"""

# ruff: noqa: F821  (``common`` is the name these classes get inside a notebook cell)
import importlib.util
import json
import re
import textwrap
import time
from pathlib import Path

import numpy as np


class Shared:
    # Validated colour-blind-safe categorical order (blue, orange, aqua, yellow, magenta, green,
    # violet, red) for any series a script does not colour explicitly; white print background.
    PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    MANUSCRIPT_RC = {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "normal",
        "axes.labelsize": 8,
        "axes.edgecolor": "#1a1a1a",
        "axes.labelcolor": "#0b0b0b",
        "text.color": "#0b0b0b",
        "xtick.color": "#1a1a1a",
        "ytick.color": "#1a1a1a",
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
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "patch.linewidth": 0.6,
        "errorbar.capsize": 2,
        "image.cmap": "viridis",
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    SINGLE_COLUMN_IN, DOUBLE_COLUMN_IN, MAX_HEIGHT_IN, MIN_HEIGHT_IN, MIN_FONT_PT = 3.4, 7.0, 9.0, 2.4, 6.0
    _RUN_ID_PREFIX = re.compile(r"^\s*R(?:\d+|_PERF)[a-z]?\s*[:–—-]\s*")
    _STATUS_FLAG = re.compile(r"status\s*=\s*[\w-]+(?:\s*,\s*)?")
    _EMPTY_PARENS = re.compile(r"\s*\(\s*\)")
    _PANEL_TAG = re.compile(r"^\(?[a-z]\)")
    _style_applied = False

    @staticmethod
    def setup_style():
        """Manuscript style: Times New Roman, 8 pt at print size, white background."""
        import glob

        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        if not common._style_applied:
            # Colab installs Times New Roman with ttf-mscorefonts-installer (first notebook cell).
            for font in glob.glob("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman*.ttf"):
                font_manager.fontManager.addfont(font)
            plt.rcParams.update(common.MANUSCRIPT_RC)
            plt.rcParams["axes.prop_cycle"] = plt.cycler(color=common.PALETTE)
            common._style_applied = True
        return plt

    @staticmethod
    def _words(text):
        """Print code identifiers as words (``n_qubits`` -> ``n qubits``, ``->`` -> arrow)."""
        return text if "$" in text else re.sub(r"(?<=[a-z0-9])_(?=[a-z0-9])", " ", text).replace("->", "→")

    @staticmethod
    def _sentence_case(text):
        """Capitalize a leading lowercase word of 3+ letters (variables like ``q`` stay)."""
        word = re.split(r"[ -]", text, maxsplit=1)[0]
        if len(word) >= 3 and word.isalpha() and word.islower():
            return text[0].upper() + text[1:]
        return text

    @staticmethod
    def _clean_title(text):
        """Drop run-ID prefixes and pass/fail flags: captions, not titles, carry those."""
        text = common._RUN_ID_PREFIX.sub("", text)
        text = common._EMPTY_PARENS.sub("", common._STATUS_FLAG.sub("", text)).strip()
        return text if common._PANEL_TAG.match(text) else common._sentence_case(common._words(text))

    @staticmethod
    def _crowded(ax):
        """Long tick labels, many annotations or a long legend: a single plot needing two columns."""
        legend = ax.get_legend()
        labels = [t.get_text() for t in ax.get_xticklabels() + ax.get_yticklabels()]
        return (max(map(len, labels), default=0) > 14 or len(ax.texts) > 8
                or (legend is not None and len(legend.get_texts()) > 6))

    @staticmethod
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

    @staticmethod
    def finish_figure(fig):
        """Final printed size and manuscript conventions (idempotent).

        Single plots become one journal column wide (3.4 in), multi-panel figures and diagrams two
        (7.0 in); the suptitle is dropped (the caption carries it); panel titles lose run IDs and
        status flags, get (a), (b), ... tags and wrap to the panel width; identifiers print as
        words; every label is serif and at least 6 pt.
        """
        from matplotlib.text import Text
        from matplotlib.ticker import FixedFormatter, FuncFormatter

        panels, seen = [], set()
        for ax in fig.axes:  # twinx/twiny axes share their host's position: one panel
            bounds = tuple(round(b, 6) for b in ax.get_position().bounds)
            if ax.get_visible() and ax.get_label() != "<colorbar>" and bounds not in seen:
                panels.append(ax)
                seen.add(bounds)
        reflowed_height = common._reflow(fig, panels)
        width, height = fig.get_size_inches()
        single_plot = len(panels) == 1 and panels[0].axison and width <= 8 and not common._crowded(panels[0])
        target = common.SINGLE_COLUMN_IN if width <= 4.5 or single_plot else common.DOUBLE_COLUMN_IN
        new_height = reflowed_height or height * target / width
        fig.set_size_inches(target, min(max(new_height, common.MIN_HEIGHT_IN), common.MAX_HEIGHT_IN))
        fig.patch.set_facecolor("white")
        if getattr(fig, "_suptitle", None) is not None:
            fig._suptitle.set_text("")
        for ax in panels:
            ax.set_facecolor("white")
            for title in (ax.title, ax._left_title, ax._right_title):
                title.set_text(common._clean_title(title.get_text()))
            for axis in (ax.xaxis, ax.yaxis):
                axis.label.set_text(common._sentence_case(common._words(axis.label.get_text())))
                formatter = axis.get_major_formatter()  # label lists: set_ticklabels / categories
                labelled = isinstance(formatter, (FixedFormatter, FuncFormatter))
                categorical = type(formatter).__name__ == "StrCategoryFormatter"
                if (labelled or categorical) and not getattr(formatter, "_as_words", False):
                    words = FuncFormatter(lambda x, pos, f=formatter: common._words(str(f(x, pos))))
                    words._as_words = True
                    axis.set_major_formatter(words)
        texts = [t for t in fig.findobj(Text) if t.get_text()]
        if len(panels) > 1 and not any(common._PANEL_TAG.match(t.get_text().strip()) for t in texts):
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
            text.set_text(common._words(text.get_text()))
            text.set_fontfamily("serif")
            if text.get_fontsize() < common.MIN_FONT_PT:
                text.set_fontsize(common.MIN_FONT_PT)

    @staticmethod
    def save_figure(fig, name, *, tight=True, svg=True):
        """Show the figure inline and save vector PDF/SVG plus a 600-dpi PNG."""
        import sys

        common.finish_figure(fig)
        if tight:
            fig.tight_layout(pad=0.4)
        paths = {"pdf": common.FIG_PDF / f"{name}.pdf", "png": common.FIG_PNG / f"{name}.png"}
        if svg:
            paths["svg"] = common.FIG_SVG / f"{name}.svg"
        for kind, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=600 if kind == "png" else None)
        if "ipykernel" in sys.modules:
            from IPython.display import display

            display(fig)
        return (paths["pdf"], paths["png"])

    @staticmethod
    def save_caption(name, text):
        path = common.ROOT / "figures" / "captions" / f"{name}.caption.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.strip() + "\n", encoding="utf-8")
        return path

    @staticmethod
    def _json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, complex):
            return {"real": value.real, "imag": value.imag}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, set):
            return sorted(value)
        raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")

    @staticmethod
    def save_csv(rows, name):
        """Write a list of flat dict rows as results/csv/<name>.csv."""
        import csv

        path = common.CSV_DIR / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return path
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        return path

    @staticmethod
    def save_json(obj, name):
        """Write an object as pretty JSON under results/json/<name>.json."""
        path = common.JSON_DIR / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, default=common._json_default) + "\n", encoding="utf-8")
        return path

    @staticmethod
    def save_raw(text, name):
        """Write raw text (e.g. serialized patterns) under results/raw/<name>."""
        path = common.RAW_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    @staticmethod
    def provenance():
        """Compact environment record written next to every result."""
        import os
        import platform
        import subprocess
        from datetime import datetime, timezone
        from importlib.metadata import PackageNotFoundError, version

        def run(*command):
            try:
                done = subprocess.run(command, capture_output=True, text=True, timeout=10)
                return (done.stdout.strip() or None) if done.returncode == 0 else None
            except (OSError, subprocess.TimeoutExpired):
                return None

        packages = {}
        for package in ("photographiq", "numpy", "scipy", "networkx", "matplotlib", "piquasso",
                        "graphix", "jax", "mpmath", "numba", "cupy", "mentpy", "scikit-learn"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = None
        try:
            from photographiq import native

            native_info = {"available": bool(native.AVAILABLE), "components": list(native.components())}
        except Exception as exc:  # report, never fail provenance collection
            native_info = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": run("git", "rev-parse", "HEAD"),
            "git_branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "gpu": run("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"),
            "native": native_info,
            "packages": packages,
        }

    @staticmethod
    def rng(seed):
        """NumPy Generator seeded deterministically for one experiment."""
        return np.random.default_rng(seed)

    @staticmethod
    def benchmark(fn, *, warmup=2, repeats=11):
        """Median/IQR/std (seconds) of repeated ``fn()`` calls after warm-up runs."""
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

    @staticmethod
    def safe_log10(value, *, floor=1e-300):
        return float(np.log10(max(float(value), floor)))

    @staticmethod
    def fock_dimension(modes, cutoff):
        """Total-photon Fock dimension binom(modes + cutoff - 1, modes)."""
        from math import comb

        return comb(modes + cutoff - 1, modes)

    @staticmethod
    def vector_bytes(dimension):
        return 16 * dimension  # complex128 state vector

    @staticmethod
    def density_bytes(dimension):
        return 16 * dimension * dimension  # dense complex128 density matrix

    @staticmethod
    def has_module(name):
        return importlib.util.find_spec(name) is not None

    class SkippedExperiment(Exception):
        """Raised when an experiment cannot run in this environment (missing dependency)."""

    @staticmethod
    def require_module(name, reason=""):
        if not common.has_module(name):
            message = f"optional dependency {name!r} is not installed. {reason}".strip()
            raise common.SkippedExperiment(message)

    @staticmethod
    def print_summary(title, **fields):
        print(f"\n=== {title} ===")
        for key, value in fields.items():
            print(f"  {key}: {value}")


class Core:
    def _made(path):  # plain function used in the class body: output folders exist up front
        path.mkdir(parents=True, exist_ok=True)
        return path

    ROOT = Path("manuscript_outputs") / "photographiq"
    RESULTS = ROOT / "results"
    CSV_DIR = _made(RESULTS / "csv")
    JSON_DIR = _made(RESULTS / "json")
    RAW_DIR = _made(RESULTS / "raw")
    FIG_PDF = _made(ROOT / "figures" / "pdf")
    FIG_PNG = _made(ROOT / "figures" / "png")
    FIG_SVG = _made(ROOT / "figures" / "svg")
    TABLES = _made(ROOT / "tables")
    HAS_GRAPHIX = importlib.util.find_spec("graphix") is not None
    HAS_JAX = importlib.util.find_spec("jax") is not None

    @staticmethod
    def write_metadata(name):
        return common.save_json(common.provenance(), f"{name}.metadata")

    @staticmethod
    def save_result(rows, name, *, extra=None):
        """CSV rows, a JSON mirror and a provenance companion for one experiment."""
        common.save_csv(rows, name)
        common.save_json({"rows": rows, **(extra or {})}, name)
        common.write_metadata(name)

    @staticmethod
    def frobenius_error(actual, expected):
        return float(np.linalg.norm(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float)))

    @staticmethod
    def relative_error(actual, expected, *, floor=1e-300):
        actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        return float(np.linalg.norm(actual - expected) / max(float(np.linalg.norm(expected)), floor))

    @staticmethod
    def is_finite_real(value):
        import math

        return isinstance(value, (int, float, np.floating, np.integer)) and math.isfinite(float(value))

    @staticmethod
    def safe_cutoff_run(fn, *, label="run"):
        """fn() with MemoryError / cutoff ValueErrors turned into a structured failure record."""
        try:
            return {"ok": True, "value": fn()}
        except MemoryError as exc:
            return {"ok": False, "error": str(exc), "kind": "MemoryError", "label": label}
        except ValueError as exc:
            message = str(exc)
            if any(word in message.lower() for word in ("cutoff", "dimension", "norm")):
                return {"ok": False, "error": message, "kind": "ValueError", "label": label}
            raise

    @staticmethod
    def canonical_edges(graph):
        return {tuple(sorted((u, v), key=repr)) for u, v in graph.edges()}

    @staticmethod
    def canonical_node_order(graph):
        return tuple(sorted(graph.nodes(), key=repr))

    @staticmethod
    def timing_final():
        """True when the run is declared to be on an otherwise idle machine (PGQ_IDLE_MACHINE=1)."""
        import os

        return os.environ.get("PGQ_IDLE_MACHINE") == "1"

    @staticmethod
    def timing_label():
        if common.timing_final():
            return "final (idle machine declared via PGQ_IDLE_MACHINE=1)"
        return "PROVISIONAL (machine not declared idle; rerun with PGQ_IDLE_MACHINE=1)"


class ML:
    def _made(path):  # plain function used in the class body: output folders exist up front
        path.mkdir(parents=True, exist_ok=True)
        return path

    ROOT = Path("manuscript_outputs") / "photographiqml"
    RESULTS = ROOT / "results"
    CSV_DIR = _made(RESULTS / "csv")
    JSON_DIR = _made(RESULTS / "json")
    RAW_DIR = _made(RESULTS / "raw")
    FIG_PDF = _made(ROOT / "figures" / "pdf")
    FIG_PNG = _made(ROOT / "figures" / "png")
    FIG_SVG = _made(ROOT / "figures" / "svg")
    TABLES = _made(ROOT / "tables")
    CANONICAL_R2_TOLERANCE = float(10 * np.finfo(float).eps)
    # Role -> colour, identical in every figure of the PhotoGraphiQML suite.
    COLORS = {
        "analytic": "#24677b",  # analytic / closed-form oracle (class A)
        "photographiqml": "#c66d27",  # system under test (logical / physical MuTA)
        "mentpy": "#627a36",  # independent external logical reference (class B)
        "photographiq": "#5b4b8a",  # independently assembled public Pattern (class B/C)
        "piquasso": "#b23a48",  # raw physical simulation reference (class B/C)
        "classical_baseline": "#8c8c8c",  # logistic / RBF-SVM baselines
        "unsupported": "#bdbdbd",  # unsupported / rejected / not applicable
        "acceptance": "#1a1a1a",  # acceptance-threshold reference lines
    }
    ORACLE_CLASSES = {
        "A": "independent analytic oracle",
        "B": "independent external implementation",
        "C": "external implementation with explicitly identified shared inputs/resources",
        "D": "independent code path or cross-backend comparison",
        "E": "structural/self-consistency test",
        "N/A": "descriptive performance measurement with no correctness oracle",
    }
    HAS_MENTPY = importlib.util.find_spec("mentpy") is not None
    HAS_PIQUASSO = importlib.util.find_spec("piquasso") is not None
    HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None

    @staticmethod
    def panel_label(ax, label):
        """Bold (a)/(b)/... panel label at the top-left of an axes."""
        ax.text(-0.12, 1.05, f"({label})", transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom", ha="right")

    @staticmethod
    def write_metadata(name, **extra):
        return common.save_json({**common.provenance(), **extra}, f"{name}.metadata")

    @staticmethod
    def save_result(rows, name, *, extra=None, meta_extra=None):
        """Evidence with separate execution, structural and scientific status.

        ``status`` is the structural contract only; it is never an ML-performance claim for a
        descriptive study.
        """
        extra = dict(extra or {})
        structural = extra.get("structural_status", extra.get("status", "unknown"))
        descriptive = extra.get("oracle_class") == "N/A"
        outcome = extra.get("scientific_outcome")
        if outcome is None:
            if name.startswith("R48_"):
                outcome = "negative"
            elif descriptive:
                outcome = "descriptive"
            else:
                outcome = "positive" if structural == "pass" else "inconclusive"
        extra.setdefault("execution_status", "completed")
        extra.setdefault("structural_status", structural)
        extra.setdefault("scientific_outcome", outcome)
        extra.setdefault("claim_supported",
                         "structural contract only" if descriptive else "declared experiment contract")
        extra.setdefault("claim_not_supported", "no scientific performance conclusion"
                         if descriptive else "no claim beyond declared protocol")
        meta_extra = dict(meta_extra or {})
        meta_extra.update({key: extra[key] for key in
                           ("execution_status", "structural_status", "scientific_outcome")})
        if "seeds" in extra:
            meta_extra.setdefault("seed_set", extra["seeds"])
            meta_extra.setdefault("n_repetitions", len(extra["seeds"]))
        if "n_repetitions" in extra:
            meta_extra.setdefault("n_repetitions", extra["n_repetitions"])
        if "uncertainty_method" in extra:
            meta_extra.setdefault("uncertainty_method", extra["uncertainty_method"])
        common.save_csv(rows, name)
        common.save_json({"rows": rows, **extra}, name)
        common.write_metadata(name, **meta_extra)

    @staticmethod
    def frobenius_error(actual, expected):
        return float(np.linalg.norm(np.asarray(actual) - np.asarray(expected)))

    @staticmethod
    def relative_error(actual, expected, *, floor=1e-300):
        denom = max(float(np.linalg.norm(np.asarray(expected))), floor)
        return common.frobenius_error(actual, expected) / denom

    @staticmethod
    def is_finite_real(value):
        return bool(np.isfinite(value)) and np.isreal(value)

    @staticmethod
    def declare_tolerance(scale, *, conditioning=1.0, truncation=0.0, safety_factor=10.0):
        """tol = safety_factor * max(eps * conditioning * scale, truncation), fixed before a sweep."""
        eps = np.finfo(float).eps
        return float(safety_factor * max(eps * conditioning * scale, truncation))

    @staticmethod
    def bootstrap_ci(values, *, statistic, n_boot=2000, alpha=0.05, seed=0):
        """Percentile bootstrap CI for an explicitly chosen 1-D statistic."""
        values = np.asarray(values, dtype=float)
        name = getattr(statistic, "__name__", str(statistic))
        if len(values) == 0:
            return {"point": None, "low": None, "high": None, "n_boot": n_boot, "n": 0, "statistic": name}
        index = np.random.default_rng(seed).integers(0, len(values), size=(n_boot, len(values)))
        low, high = np.percentile(statistic(values[index], axis=1), [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return {"point": float(statistic(values)), "low": float(low), "high": float(high),
                "n_boot": n_boot, "alpha": alpha, "n": int(len(values)), "statistic": name}

    @staticmethod
    def paired_difference_ci(a, b, **kwargs):
        """Bootstrap CI on the paired difference a - b (identical seeds/splits)."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a.shape != b.shape:
            raise ValueError("Paired comparison requires matching shapes")
        return common.bootstrap_ci(a - b, **kwargs)

    @staticmethod
    def safe_cutoff_run(fn, *, label="run"):
        """fn() with anticipated resource/allocation failures turned into a structured record."""
        try:
            return {"ok": True, "value": fn()}
        except (MemoryError, ValueError, ArithmeticError, NotImplementedError) as exc:
            return {"ok": False, "error": str(exc), "kind": type(exc).__name__, "label": label}


class Metadata:
    @staticmethod
    def collect():
        return common.provenance()
