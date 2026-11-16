"""Shared helpers for the CV diagram / CV-MBQC optimisation studies R61-R68 (core catalogue).

Each study saves, with the suite's conventions (``common.save_result``/``save_figure``):
CSV + JSON rows, a metadata companion (environment), figures (PDF/PNG/SVG) and a caption; the
JSON additionally carries ``configuration``, ``validation_report`` and ``source_revision``.
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "tests" / "validation"))

import common  # noqa: E402

import photographiq as pg  # noqa: E402

COLORS = {"original": "#8c8c8c", "optimised": "#c66d27", "reference": "#24677b",
          "logical": "#5b4b8a", "physical": "#b23a48", "accent": "#627a36"}


def plt():
    return common.setup_style()


def revision():
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--short"], cwd=REPO, capture_output=True,
                                    text=True, check=True).stdout.strip())
        return {"revision": rev, "working_tree_modified": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": "unknown", "working_tree_modified": None}


def save(rows, name, *, configuration, validation, figure=None, caption=None):
    common.save_result(rows, name, extra={"configuration": configuration, "validation_report": validation,
                                          "source_revision": revision(), "status": validation["status"]})
    if figure is not None:
        common.save_figure(figure, name)
    if caption:
        common.save_caption(name, caption)


def random_circuit(rng, modes, depth, *, allow=("rotate", "squeeze", "displace", "cz", "beamsplitter")):
    """Random Gaussian circuit with a deterministic generator (documented distribution).

    Each of ``depth`` gates is uniform over ``allow`` (two-mode gates only if modes > 1);
    angles U(-pi, pi), squeezing N(0, 0.3^2), displacements N(0, 0.5^2), CZ weights U(-1, 1).
    """
    c = pg.Circuit(modes)
    for _ in range(depth):
        kinds = [k for k in allow if modes > 1 or k not in ("cz", "beamsplitter")]
        kind = kinds[rng.integers(len(kinds))]
        m = int(rng.integers(modes))
        o = int((m + 1 + rng.integers(max(modes - 1, 1))) % modes) if modes > 1 else m
        if kind == "rotate":
            c.rotate(m, rng.uniform(-np.pi, np.pi))
        elif kind == "squeeze":
            c.squeeze(m, 0.3 * rng.normal())
        elif kind == "displace":
            c.displace(m, 0.5 * rng.normal(), 0.5 * rng.normal())
        elif kind == "cz":
            c.cz(m, o, rng.uniform(-1, 1))
        else:
            c.beamsplitter(m, o, rng.uniform(-np.pi, np.pi))
    return c


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t
