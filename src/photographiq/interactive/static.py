"""Matplotlib export of a :class:`GraphView` stage (SVG, PDF or PNG; no browser).

Styling matches the HTML view: CZ edges solid grey with weight labels, classical
dependency arrows dashed blue, correction arrows dotted orange with coefficient
labels, inputs as squares, outputs as diamonds, measured nodes as circles.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.typing import RcKeyType

    from .graph_view import GraphView

COLORS = {
    "input": "#7fd1d6",
    "output": "#f3c766",
    "measured": "#a8c7e0",
    "edge": "#64748b",
    "flow": "#2563eb",
    "correction": "#c2410c",
}
RC: dict[RcKeyType, Any] = {
    "font.size": 10,
    "font.family": "sans-serif",  # SVG text then lists a generic fallback after DejaVu Sans
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
}


def _marker(roles: list[str]) -> tuple[str, str]:
    if "output" in roles:
        return "D", COLORS["output"]
    if "input" in roles:
        return "s", COLORS["input"]
    return "o", COLORS["measured"]


def draw_static(
    view: GraphView,
    path: str | Path | None = None,
    stage: str | None = None,
    *,
    ax=None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 300,
    formulas: bool = True,
    title: str | None = None,
):
    """Draw ``view`` at ``stage``; save to ``path`` (suffix picks the format) if given.

    Returns the saved path, or the Axes when ``path`` is None. When saving without an
    ``ax`` a standalone ``Figure`` is used (no pyplot state, no GUI backend).
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    from .graph_view import STAGES

    level = view._stage(stage)
    assert view.layout is not None  # GraphView.__post_init__ always sets a layout
    g, pos = view.graph, view.layout.positions
    xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    span = (
        max(xs, default=0) - min(xs, default=0) + 1.2,
        max(ys, default=0) - min(ys, default=0) + 1.4,
    )
    with plt.rc_context(RC):
        if ax is None:
            size = figsize or (min(1.3 * span[0] + 3.0, 16), min(1.3 * span[1] + 0.9, 12))
            ax = (plt.figure(figsize=size) if path is None else Figure(figsize=size)).subplots()
        for u, v, data in g.graph.edges(data=True):
            (x1, y1), (x2, y2) = pos[u], pos[v]
            ax.plot([x1, x2], [y1, y2], color=COLORS["edge"], lw=1.6, zorder=1)
            if g.model.continuous:
                ax.text(
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    f"{float(data['weight']):.3g}",
                    fontsize=8,
                    ha="center",
                    va="center",
                    zorder=4,
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none"},
                )
        kind, arrows = view._arrows(level)
        for s, t, label in arrows:
            if s not in pos or s == t:
                continue
            color, style, rad = (
                (COLORS["flow"], "--", 0.25)
                if kind == "flow"
                else (COLORS["correction"], ":", -0.25)
            )
            ax.add_patch(
                FancyArrowPatch(
                    pos[s],
                    pos[t],
                    arrowstyle="-|>",
                    mutation_scale=12,
                    lw=1.4,
                    linestyle=style,
                    color=color,
                    shrinkA=13,
                    shrinkB=13,
                    connectionstyle=f"arc3,rad={rad}",
                    zorder=2,
                )
            )
            if label:
                (x1, y1), (x2, y2) = pos[s], pos[t]
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                dx, dy = x2 - x1, y2 - y1
                ax.text(
                    mx + rad * dy * 0.5,
                    my - rad * dx * 0.5,
                    label,
                    fontsize=7,
                    color=color,
                    ha="center",
                    va="center",
                    zorder=4,
                    bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none"},
                )
        order, _ = view._ordering(level)
        outcomes = {}
        if level >= 3:
            assert view.trace is not None  # _stage rejects level 3 without a trace
            outcomes = {s.nodes[0]: s.outcome for s in view.trace.steps if s.kind == "measure"}
        for node in g.nodes:
            x, y = pos[node]
            marker, color = _marker(view.roles(node))
            ax.scatter(
                [x],
                [y],
                s=620,
                marker=marker,
                c=color,
                edgecolors="#334155",
                linewidths=1.2,
                zorder=3,
                clip_on=False,
            )
            ax.text(x, y, str(node), ha="center", va="center", fontsize=9, zorder=5)
            if node in order:
                ax.text(
                    x + 0.17, y + 0.2, f"#{order[node]}", fontsize=7, color=COLORS["flow"], zorder=5
                )
            if node in g.squeezing and level == 0:
                ax.text(
                    x,
                    y - 0.2,
                    f"r={g.squeezing[node]:.3g}",
                    fontsize=7,
                    ha="center",
                    va="top",
                    color="#475569",
                    zorder=5,
                )
            if node in outcomes:
                value = outcomes[node]
                text = f"{value:.3g}" if isinstance(value, float) else str(value)
                ax.text(
                    x,
                    y - 0.2,
                    f"m={text}",
                    fontsize=7,
                    ha="center",
                    va="top",
                    color="#b91c1c",
                    zorder=5,
                )
        handles = [
            Line2D([], [], marker="s", ls="", mfc=COLORS["input"], mec="#334155", label="input"),
            Line2D([], [], marker="D", ls="", mfc=COLORS["output"], mec="#334155", label="output"),
            Line2D(
                [], [], marker="o", ls="", mfc=COLORS["measured"], mec="#334155", label="measured"
            ),
            Line2D([], [], color=COLORS["edge"], lw=1.6, label="CZ edge"),
        ]
        if kind == "flow":
            handles.append(Line2D([], [], color=COLORS["flow"], ls="--", label="dependency"))
        if kind == "correction":
            handles.append(Line2D([], [], color=COLORS["correction"], ls=":", label="correction"))
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            frameon=False,
            ncol=len(handles),
            fontsize=8,
        )
        if formulas and level >= 2 and view.corrections:
            lines = [f"{t}: {c.formula()}" for t, c in view.corrections.items()]
            ax.text(
                1.01,
                1.0,
                "corrections:\n" + "\n".join(lines),
                transform=ax.transAxes,
                fontsize=7,
                va="top",
                family="monospace",
            )
        ax.set_title(title if title is not None else f"{view.title} ({STAGES[level]})", fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlim(min(xs, default=0) - 0.6, max(xs, default=0) + 0.6)
        ax.set_ylim(min(ys, default=0) - 0.5, max(ys, default=0) + 0.6)
        ax.set_axis_off()
        if path is None:
            return ax
        path = Path(path)
        ax.figure.savefig(path, dpi=dpi)
        return path
