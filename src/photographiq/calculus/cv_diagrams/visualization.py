"""Static drawings of CV diagrams (matplotlib) and JSON rewrite traces.

One horizontal line per bosonic wire (solid), boxes as labelled rectangles placed in the
earliest column free on all their wires, classical records as dashed lines from the
producing measurement to every box that reads them. Highlighted boxes (e.g. a rewritten
region) are filled. No interactive view is provided.
"""

from __future__ import annotations

import json

import numpy as np

from .ir import (
    CZ,
    BeamSplitter,
    ClassicalSource,
    Displacement,
    Feedforward,
    Homodyne,
    PhasePolynomial,
    PhotonCount,
    Prepare,
    Rotation,
    Shear,
    Squeezer,
    Symplectic,
)


def _label(b):
    if isinstance(b, Prepare):
        return f"S{b.squeezing:.2g}" if b.squeezing is not None else "G"
    if isinstance(b, Rotation):
        return f"R{b.theta:.2f}"
    if isinstance(b, Squeezer):
        return f"Sq{b.r:.2f}"
    if isinstance(b, Shear):
        return f"P{b.s:.2f}"
    if isinstance(b, Displacement):
        return "D"
    if isinstance(b, Homodyne):
        return f"H{b.theta:.2f}"
    if isinstance(b, Feedforward):
        return "FF"
    if isinstance(b, Symplectic):
        return "Sp"
    if isinstance(b, PhasePolynomial):
        return f"e^if{len(b.coefficients) - 1}"
    if isinstance(b, PhotonCount):
        return "n"
    if isinstance(b, ClassicalSource):
        return "N(m,v)"
    return type(b).__name__[:4]


def draw(diagram, ax=None, *, highlight=(), title=None):
    """Draw ``diagram``; returns the matplotlib Axes."""
    import matplotlib.pyplot as plt

    wires = list(
        dict.fromkeys(list(diagram.inputs) + [m for b in diagram.boxes for m in b.creates])
    )
    row = {w: i for i, w in enumerate(wires)}
    free = {w: 0 for w in wires}
    columns: list = []
    positions: dict = {}
    for i, b in enumerate(diagram.boxes):
        ws = list(b.modes) + list(b.creates) + list(b.consumes)
        col = max([free[w] for w in ws] + [0]) if ws else max(free.values(), default=0)
        for k in b.reads:
            col = max(col, positions.get(("rec", k), (0, 0))[0] + 1)
        for w in ws:
            free[w] = col + 1
        positions[i] = (col, [row[w] for w in ws] or [len(wires)])
        for k in b.writes:
            positions[("rec", k)] = (col, [row[w] for w in ws] or [len(wires)])
        columns.append(col)
    if ax is None:
        width = min(16, 1.2 + 0.32 * (max(columns, default=0) + 2))
        _, ax = plt.subplots(figsize=(width, 0.5 + 0.45 * len(wires)))
    start = {w: 0 for w in diagram.inputs}
    for i, b in enumerate(diagram.boxes):
        for w in b.creates:
            start[w] = positions[i][0]
    end = {w: positions[i][0] for i, b in enumerate(diagram.boxes) for w in b.consumes}
    last = max(columns, default=0) + 1
    for w in wires:
        ax.plot([start.get(w, 0), end.get(w, last)], [-row[w]] * 2, color="0.35", lw=0.8, zorder=0)
    for i, b in enumerate(diagram.boxes):
        col, rows = positions[i]
        lo, hi = min(rows), max(rows)
        fc = "#f4c95d" if i in highlight else ("#dbe9f6" if b.gaussian else "#f3c4c4")
        ax.add_patch(
            plt.Rectangle(
                (col - 0.38, -hi - 0.3), 0.76, hi - lo + 0.6, fc=fc, ec="0.2", lw=0.6, zorder=2
            )
        )
        ax.text(col, -(lo + hi) / 2, _label(b), ha="center", va="center", fontsize=5.5, zorder=3)
        if isinstance(b, (CZ, BeamSplitter)) and hi > lo:
            ax.plot([col, col], [-lo, -hi], color="0.2", lw=0.6, zorder=1)
        for k in b.reads:
            src = positions.get(("rec", k))
            if src:
                ax.plot(
                    [src[0], col],
                    [-min(src[1]) - 0.3, -lo - 0.3],
                    color="#b23a48",
                    ls="--",
                    lw=0.5,
                    zorder=1,
                )
    classical = any(not (set(b.modes) | set(b.creates) | set(b.consumes)) for b in diagram.boxes)
    ticks = [-row[w] for w in wires] + ([-len(wires)] if classical else [])
    names = [str(w) for w in wires] + (["classical"] if classical else [])
    ax.set_yticks(ticks, names, fontsize=6)
    ax.set_xticks([])
    ax.set_xlim(-0.8, last + 0.3)
    ax.set_ylim(-len(wires) - (0.7 if classical else -0.3), 0.7)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title)
    return ax


def trace_json(diagram, **extra) -> str:
    """Rewrite trace of ``diagram`` as JSON (rule, boxes before/after, equivalence, ...)."""

    def enc(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return repr(o)

    return json.dumps(
        {"trace": [s.to_dict() for s in diagram.trace], **extra}, default=enc, indent=1
    )
