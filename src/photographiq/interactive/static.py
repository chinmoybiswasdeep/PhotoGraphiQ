"""Matplotlib export of a :class:`GraphView` stage (SVG, PDF or PNG; no browser).

Styling matches the HTML view (colour-blind-safe Okabe-Ito hues, white background):

* CZ (quantum) edges: neutral solid grey with weight labels;
* flow dependencies: solid blue arrows (grey dashed when the flow is not certified);
* classical corrections: dashed vermillion-orange arrows with coefficient labels;
* inputs: blue squares; outputs: green diamonds; other (ancilla) nodes: neutral circles;
  measured nodes (every non-output) carry a thick dark outline, so roles never rely on
  colour alone.

Edges and arrows are quadratic curves routed around nodes that are not their
endpoints (:func:`~photographiq.interactive.layouts.route`). Every drawn element has a
stable SVG ``id`` (``node:<id>``, ``label:<id>``, ``edge:<u>--<v>``,
``flow:<s>-><t>``, ``correction:<s>-><t>``) so exported SVG can be checked against
the model.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.typing import RcKeyType

    from .graph_view import GraphView

COLORS = {
    "input": "#9ccbee",  # light fill of Okabe-Ito blue #0072B2
    "input_edge": "#0072b2",
    "output": "#a6dcc6",  # light fill of Okabe-Ito bluish green #009E73
    "output_edge": "#00795a",
    "ancilla": "#eef0f3",
    "ancilla_edge": "#475569",
    "measured_edge": "#1f2937",
    "edge": "#64748b",
    "flow": "#0072b2",
    "uncertified": "#6b7280",
    "correction": "#d55e00",
    "outcome": "#9f1d6b",
}
NODE_SIZE = 560  # marker area, points^2
MEASURED_LW, UNMEASURED_LW = 2.4, 1.0
ARROW_LW, EDGE_LW = 1.4, 1.6
RC: dict[RcKeyType, Any] = {
    "font.size": 10,
    "font.family": "sans-serif",  # SVG text then lists a generic fallback after DejaVu Sans
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
}


def node_style(roles: list[str]) -> dict:
    """Marker, face colour, edge colour and line width of a node with ``roles``."""
    if "output" in roles:
        marker, face, edge = "D", COLORS["output"], COLORS["output_edge"]
    elif "input" in roles:
        marker, face, edge = "s", COLORS["input"], COLORS["input_edge"]
    else:
        marker, face, edge = "o", COLORS["ancilla"], COLORS["ancilla_edge"]
    measured = "measured" in roles
    return {
        "marker": marker,
        "facecolor": face,
        "edgecolor": COLORS["measured_edge"] if measured else edge,
        "linewidth": MEASURED_LW if measured else UNMEASURED_LW,
    }


def legend_handles(kind: str = "", uncertified: bool = False) -> list:
    """Legend entries shared by every static graph figure."""
    from matplotlib.lines import Line2D

    def node(roles, label):
        st = node_style(roles)
        return Line2D(
            [],
            [],
            marker=st["marker"],
            ls="",
            ms=9,
            mfc=st["facecolor"],
            mec=st["edgecolor"],
            mew=st["linewidth"],
            label=label,
        )

    handles = [
        node(["input", "measured"], "input (measured)"),
        node(["output"], "output (unmeasured)"),
        node(["measured", "ancilla"], "ancilla (measured)"),
        Line2D([], [], color=COLORS["edge"], lw=EDGE_LW, label="CZ edge (weight)"),
    ]
    if kind == "flow":
        color, ls = (COLORS["uncertified"], "--") if uncertified else (COLORS["flow"], "-")
        text = "dependency (NOT certified)" if uncertified else "flow dependency"
        handles.append(
            Line2D([], [], color=color, ls=ls, lw=ARROW_LW, marker=">", ms=5, label=text)
        )
    if kind == "correction":
        handles.append(
            Line2D(
                [],
                [],
                color=COLORS["correction"],
                ls="--",
                lw=ARROW_LW,
                marker=">",
                ms=5,
                label="classical correction",
            )
        )
    return handles


def _curve(
    ax, a, b, offset, *, gid, color, lw, ls="-", arrow=False, shrink=0.0, zorder=1, head=11.0
):
    from matplotlib.patches import FancyArrowPatch

    length = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
    patch = FancyArrowPatch(
        a,
        b,
        arrowstyle="-|>" if arrow else "-",
        mutation_scale=head,
        lw=lw,
        linestyle=ls,
        color=color,
        shrinkA=shrink,
        shrinkB=shrink,
        connectionstyle=f"arc3,rad={offset / length}",
        zorder=zorder,
    )
    patch.set_gid(gid)
    ax.add_patch(patch)


def _apex(a, b, offset):
    """Point of the quadratic curve at t = 1/2 (where labels sit)."""
    (x1, y1), (x2, y2) = a, b
    length = math.hypot(x2 - x1, y2 - y1) or 1.0
    return (x1 + x2) / 2 + offset / 2 * (y2 - y1) / length, (y1 + y2) / 2 - offset / 2 * (
        x2 - x1
    ) / length


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
    legend: bool = True,
    node_size: float = NODE_SIZE,
    fontsize: float = 9.0,
    weights: bool = True,
    arrow_labels: bool = True,
    order_label: str = "order",
):
    """Draw ``view`` at ``stage``; save to ``path`` (suffix picks the format) if given.

    Returns the saved path, or the Axes when ``path`` is None. When saving without an
    ``ax`` a standalone ``Figure`` is used (no pyplot state, no GUI backend).
    ``node_size`` (points^2) and ``fontsize`` scale the drawing for narrow columns (line
    widths and arrow heads follow ``node_size``); ``weights`` / ``arrow_labels`` hide CZ
    weights and correction coefficients on dense figures (formulas stay available).
    ``order_label`` annotates flow stages with the measurement order ``#k`` ("order"),
    the flow layer ``Lk`` ("layer") or both ("both").
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    from .graph_view import ARC, STAGES
    from .layouts import route

    level = view._stage(stage)
    assert view.layout is not None  # GraphView.__post_init__ always sets a layout
    g, layout = view.graph, view.layout
    pos = layout.positions
    xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    span = (
        max(xs, default=0) - min(xs, default=0) + 1.2,
        max(ys, default=0) - min(ys, default=0) + 1.4,
    )
    small = fontsize * 0.8
    apexes: list[tuple[float, float]] = []
    radius_pt = math.sqrt(node_size) / 2
    k = max(0.5, math.sqrt(node_size / NODE_SIZE))  # line widths scale with the nodes
    with plt.rc_context(RC):
        if ax is None:
            size = figsize or (min(1.3 * span[0] + 3.0, 16), min(1.3 * span[1] + 0.9, 12))
            ax = (plt.figure(figsize=size) if path is None else Figure(figsize=size)).subplots()
        for u, v, data in g.graph.edges(data=True):
            offset = route(layout, u, v, 0.0)
            apexes.append(_apex(pos[u], pos[v], offset))
            _curve(
                ax,
                pos[u],
                pos[v],
                offset,
                gid=f"edge:{u}--{v}",
                color=COLORS["edge"],
                lw=EDGE_LW * k,
            )
            if g.model.continuous and weights:
                ax.text(
                    *_apex(pos[u], pos[v], offset),
                    f"{float(data['weight']):.3g}",
                    fontsize=small,
                    ha="center",
                    va="center",
                    zorder=4,
                    color="#334155",
                    bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none"},
                )
        kind, arrows = view._arrows(level)
        uncertified = view._uncertified(kind)
        for s, t, label in arrows:
            if s not in pos or s == t:
                continue
            length = math.hypot(pos[t][0] - pos[s][0], pos[t][1] - pos[s][1])
            offset = route(layout, s, t, (ARC if kind == "flow" else -ARC) * length)
            apexes.append(_apex(pos[s], pos[t], offset))
            if kind == "flow":
                color, ls = (COLORS["uncertified"], "--") if uncertified else (COLORS["flow"], "-")
            else:
                color, ls = COLORS["correction"], "--"
            _curve(
                ax,
                pos[s],
                pos[t],
                offset,
                gid=f"{kind}:{s}->{t}",
                color=color,
                lw=ARROW_LW * k,
                ls=ls,
                arrow=True,
                shrink=radius_pt + 2 * k,
                head=11 * max(k, 0.55),
                zorder=2,
            )
            if label and arrow_labels:
                ax.text(
                    *_apex(pos[s], pos[t], offset),
                    label,
                    fontsize=small,
                    color=color,
                    ha="center",
                    va="center",
                    zorder=4,
                    bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none"},
                )
        order, layers = view._ordering(level)
        if order_label not in ("order", "layer", "both"):
            raise ValueError("order_label must be order, layer or both")
        outcomes = {}
        if level >= 3:
            assert view.trace is not None  # _stage rejects level 3 without a trace
            outcomes = {s.nodes[0]: s.outcome for s in view.trace.steps if s.kind == "measure"}
        for node in g.nodes:
            x, y = pos[node]
            st = node_style(view.roles(node))
            artist = ax.scatter(
                [x],
                [y],
                s=node_size,
                marker=st["marker"],
                c=st["facecolor"],
                edgecolors=st["edgecolor"],
                linewidths=st["linewidth"] * k,
                zorder=3,
                clip_on=False,
            )
            artist.set_gid(f"node:{node}")
            text = ax.text(x, y, str(node), ha="center", va="center", fontsize=fontsize, zorder=5)
            text.set_gid(f"label:{node}")
            below = []
            if node in order and order_label in ("order", "both"):
                below.append((f"#{order[node]}", COLORS["flow"]))
            if node in layers and order_label in ("layer", "both"):
                below.append((f"L{layers[node]}", COLORS["flow"]))
            if node in g.squeezing and level == 0:
                below.append((f"r={g.squeezing[node]:.3g}", "#475569"))
            if node in outcomes:
                value = outcomes[node]
                below.append(
                    (
                        f"m={value:.3g}" if isinstance(value, float) else f"m={value}",
                        COLORS["outcome"],
                    )
                )
            if below:  # annotations sit below the marker, never inside it
                ax.annotate(
                    "\n".join(t for t, _ in below),
                    (x, y),
                    xytext=(0, -radius_pt - 1.5 * k),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=small,
                    color=below[-1][1] if len(below) == 1 else "#1f2937",
                    zorder=5,
                    bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.85},
                )
        if legend:
            handles = legend_handles(kind, uncertified)
            ax.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                frameon=False,
                ncol=min(len(handles), 3),
                fontsize=small,
            )
        if formulas and level >= 2 and view.corrections:
            lines = [f"{t}: {c.formula()}" for t, c in view.corrections.items()]
            ax.text(
                1.01,
                1.0,
                "corrections:\n" + "\n".join(lines),
                transform=ax.transAxes,
                fontsize=small,
                va="top",
                family="monospace",
            )
        status = ""
        if level >= 1 and view.flow is not None and view.flow.status != "certified":
            status = f", flow {view.flow.status}"
        ax.set_title(
            title if title is not None else f"{view.title} ({STAGES[level]}{status})",
            fontsize=fontsize + 1,
        )
        ax.set_aspect("equal")
        # curves peak at their apex: keep every routed arc inside the axes
        xs += [x for x, _ in apexes]
        ys += [y for _, y in apexes]
        ax.set_xlim(min(xs, default=0) - 0.6, max(xs, default=0) + 0.6)
        ax.set_ylim(min(ys, default=0) - 0.65, max(ys, default=0) + 0.6)
        ax.set_axis_off()
        if path is None:
            return ax
        path = Path(path)
        ax.figure.savefig(path, dpi=dpi)
        return path
