"""Render the manuscript architecture figures from docs/architecture/figures.json.

figures.json is the editable source. Every box cites the code it depicts (``"refs"``):

* ``"module:attr.sub"``: the module is imported and every attribute is looked up for real;
* ``"path:relative/path"``: the repository file or directory must exist;
* ``"path:relative/file#symbol"``: the file must also define ``symbol`` (a top-level
  ``class``/``def``/assignment for ``.py`` files, a whole-word occurrence otherwise, e.g. a
  C++ function declared in a header).

Edges may carry ``"refs"`` too (e.g. the conversion function an arrow is labelled with);
they are validated the same way.

Validation fails on any missing reference, so a figure cannot show functionality that is
not implemented. Rendering additionally fails if a label overflows its box, an edge passes
through a box other than its endpoints, an edge label overlaps a box or another label, or
the legend overlaps the diagram, and (print-readability check) if any text would be smaller
than ``MIN_PT`` once the figure is placed at ``PRINT_WIDTH_PT`` (7.0 in, full text width;
narrower figures are placed at their native width). Setting the environment variable
``PGQ_FIGURES_SKIP_PRINT_CHECK=1`` downgrades only the print-readability check to a warning,
for use while figures are being redesigned. Outputs go to docs/figures/architecture/ as SVG
and PDF; captions go to captions.md and the figure index in docs/architecture/README.md
(between the ``figure-index`` markers) is regenerated. Nothing is written to docs/ when a
check fails; use ``--preview`` (PNG) or ``--report-sizes`` to inspect.

Figures may name their layers with ``"lanes": [{"label": "User API", "rows": [first, last]}]``
(inclusive grid rows): each lane is drawn as a subtle band across the figure with its label
rotated in a left gutter of ``"lane_width"`` inches (default ``LANE_GUTTER``).

Usage: python scripts/generate_architecture_figures.py [--check] [--preview DIR] [--report-sizes]
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "architecture" / "figures.json"
README = ROOT / "docs" / "architecture" / "README.md"
OUT = ROOT / "docs" / "figures" / "architecture"

# Consistent graphical conventions across all figures: kind -> (face, edge, legend text).
KIND_STYLE = {
    "api": ("#e8f1fb", "#2f6db5", "user API"),
    "ir": ("#fdf2e3", "#c27c1a", "intermediate representation"),
    "compiler": ("#efe9f8", "#6d4bb0", "compilation / analysis"),
    "engine": ("#e6f4ea", "#2e7d4f", "execution engine"),
    "optional": ("#f4f4f4", "#7a7a7a", "optional / unavailable here"),
    "result": ("#fbeaea", "#b03a3a", "results / output"),
    "ml": ("#e9f6f7", "#1f7f86", "PhotoGraphiQML"),
    "classical": ("#fff8d6", "#9a7d00", "classical processing"),
    "reference": ("#fbeef6", "#a0457a", "validation reference / oracle"),
    "test": ("#eef1f5", "#44546a", "test suite / release gate"),
}
EDGE_STYLE = {
    "quantum": {"color": "#1f3b73", "linestyle": "-", "linewidth": 1.5},
    "classical": {"color": "#a87800", "linestyle": (0, (4, 2)), "linewidth": 1.3},
    "data": {"color": "#444444", "linestyle": "-", "linewidth": 1.0},
    "optional": {"color": "#8a8a8a", "linestyle": (0, (1.5, 2)), "linewidth": 1.1},
}
EDGE_LEGEND = {
    "quantum": "quantum information",
    "classical": "classical outcomes / feed-forward",
    "data": "data / control flow",
    "optional": "optional dependency",
}
SIDES = {"n", "s", "e", "w"}
FONT = "DejaVu Sans"
LABEL_PT, DETAIL_PT, EDGE_PT, LEGEND_PT, TITLE_PT = 8.4, 6.6, 6.2, 6.6, 10.5
MIN_SCALE = 0.72  # smallest font scale tried before a box is declared too small
PAD = 0.07  # inner box padding, inches
TITLE_BAND = 0.42  # inches above the grid
LEGEND_ROW = 0.2  # inches per legend row below the grid
LANE_PT = 7.6  # lane (layer) label size, bold
LANE_GUTTER = 0.36  # inches left of the grid for rotated lane labels (figure key "lane_width")
LANE_STYLE = {"facecolor": "#f4f5f7", "edgecolor": "#d6dae0", "text": "#44546a"}
# Print target: full text width. Effective size = native size * min(1, PRINT_WIDTH_PT / width).
PRINT_WIDTH_PT = 504.0
MIN_PT = {"title": 7.0, "detail": 6.0, "edge": 6.0, "legend": 6.0, "lane": 6.0}
SKIP_PRINT_ENV = "PGQ_FIGURES_SKIP_PRINT_CHECK"


# --------------------------------------------------------------------------- validation
def _defines(path: Path, symbol: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if path.suffix != ".py":
        return re.search(rf"\b{re.escape(symbol)}\b", text) is not None
    for node in ast.parse(text).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == symbol:
                return True
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == symbol for t in targets):
                return True
    return False


def check_ref(ref: str) -> str | None:
    """Return an error message if a cited module, attribute, path or symbol does not exist."""
    if ref.startswith("path:"):
        relative, _, symbol = ref[5:].partition("#")
        path = ROOT / relative
        if not path.exists():
            return f"missing path {relative}"
        if symbol and not (path.is_file() and _defines(path, symbol)):
            return f"{relative} does not define {symbol}"
        return None
    module, _, attribute = ref.partition(":")
    try:
        target = importlib.import_module(module)
    except Exception as exc:  # any import failure is a missing reference, never a crash
        return f"cannot import {module}: {exc}"
    for part in filter(None, attribute.split(".")):
        if not hasattr(target, part):
            return f"{module} has no attribute {attribute}"
        target = getattr(target, part)
    return None


def _port(spec: str | None) -> str | None:
    if spec is None:
        return None
    if spec[:1] not in SIDES:
        return f"bad port {spec!r} (side n/s/e/w, optional offset like 's+0.25')"
    try:
        offset = float(spec[1:] or 0)
    except ValueError:
        return f"bad port offset {spec!r}"
    return None if -0.5 <= offset <= 0.5 else f"port offset outside [-0.5, 0.5]: {spec!r}"


def validate(spec: dict) -> list[str]:
    """All reference and schema errors of a figure specification (empty list: valid)."""
    errors = []
    seen = set()
    for figure in spec["figures"]:
        fid = figure.get("id", "?")
        for key in ("id", "slug", "title", "caption", "boxes", "edges"):
            if key not in figure:
                errors.append(f"{fid}: missing figure key {key!r}")
        if fid in seen:
            errors.append(f"{fid}: duplicate figure id")
        seen.add(fid)
        ids, cells = set(), {}
        for box in figure.get("boxes", []):
            name = f"{fid}/{box.get('id')}"
            if box.get("id") in ids:
                errors.append(f"{name}: duplicate box id")
            ids.add(box.get("id"))
            if not box.get("label"):
                errors.append(f"{name}: missing label")
            if box.get("kind", "api") not in KIND_STYLE:
                errors.append(f"{name}: unknown kind {box.get('kind')}")
            if not box.get("refs"):
                errors.append(f"{name}: no refs (every box must cite implemented code)")
            col, row, span = box.get("col"), box.get("row"), box.get("span", 1)
            if not all(isinstance(v, int) and v >= 0 for v in (col, row)) or span < 1:
                errors.append(f"{name}: col/row must be integers >= 0 and span >= 1")
                continue
            for c in range(col, col + span):
                if (c, row) in cells:
                    errors.append(f"{name}: overlaps box {cells[c, row]} at col {c}, row {row}")
                cells[c, row] = box["id"]
            for ref in box.get("refs", []):
                message = check_ref(ref)
                if message:
                    errors.append(f"{name}: {message}")
        for edge in figure.get("edges", []):
            if edge.get("from") not in ids or edge.get("to") not in ids:
                errors.append(f"{fid}: edge {edge.get('from')}->{edge.get('to')} unknown box")
            if edge.get("kind", "data") not in EDGE_STYLE:
                errors.append(f"{fid}: unknown edge kind {edge.get('kind')}")
            for ref in edge.get("refs", []):
                message = check_ref(ref)
                if message:
                    errors.append(f"{fid}: edge {edge.get('from')}->{edge.get('to')}: {message}")
            for key in ("from_port", "to_port"):
                message = _port(edge.get(key))
                if message:
                    errors.append(f"{fid}: edge {edge.get('from')}->{edge.get('to')}: {message}")
        last_row = max((r for _, r in cells), default=0)
        lane_rows: set[int] = set()
        for lane in figure.get("lanes", []):
            rows = lane.get("rows")
            if not (
                lane.get("label")
                and isinstance(rows, list)
                and len(rows) == 2
                and all(isinstance(r, int) for r in rows)
                and 0 <= rows[0] <= rows[1] <= last_row
            ):
                errors.append(
                    f"{fid}: lane {lane.get('label')!r} needs a label and rows [first, last] "
                    f"with 0 <= first <= last <= {last_row}"
                )
                continue
            covered = set(range(rows[0], rows[1] + 1))
            if covered & lane_rows:
                errors.append(f"{fid}: lane {lane['label']!r} overlaps another lane")
            lane_rows |= covered
    return errors


def print_sizes(figure_id: str, width_in: float, texts) -> tuple[float, dict, list[str]]:
    """Scale to the print width, smallest effective size per text category, and failures.

    ``texts`` holds ``(category, name, native_pt)`` for every text element, with any per-box
    fit scaling already applied; categories are the keys of ``MIN_PT``.
    """
    scale = min(1.0, PRINT_WIDTH_PT / (72 * width_in))
    smallest, problems = {}, []
    for category, minimum in MIN_PT.items():
        found = sorted((pt * scale, name) for c, name, pt in texts if c == category)
        if not found:
            continue
        smallest[category] = found[0][0]
        low = [f for f in found if f[0] < minimum - 1e-6]
        if low:
            problems.append(
                f"{figure_id}: {len(low)} {category} text(s) below {minimum:g} pt at "
                f"{min(width_in, PRINT_WIDTH_PT / 72):.2f} in placed width "
                f"(smallest {low[0][0]:.2f} pt: {low[0][1]})"
            )
    return scale, smallest, problems


# --------------------------------------------------------------------------- layout helpers
def _wrap(text: str, width_pt: float, prop, measure) -> list[str]:
    """Greedy word wrap with measured glyph widths; explicit newlines are kept."""
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            trial = f"{line} {word}".strip()
            if line and measure(trial, prop) > width_pt:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return lines


def _fit(box: dict, w: float, h: float, measure, font_properties):
    """Largest font scale at which label and detail fit the box; None if none does."""
    scale = 1.0
    while scale >= MIN_SCALE - 1e-9:
        label_prop = font_properties(LABEL_PT * scale, "bold")
        detail_prop = font_properties(DETAIL_PT * scale, "normal")
        width_pt = (w - 2 * PAD) * 72
        # Wrap at the same 96% margin the check below uses, so a full line never fails it.
        label = _wrap(box["label"], 0.96 * width_pt, label_prop, measure)
        detail = _wrap(box.get("detail", ""), 0.96 * width_pt, detail_prop, measure)
        detail = [] if detail == [""] else detail
        widest = max(
            measure(s, p)
            for s, p in [(t, label_prop) for t in label] + [(t, detail_prop) for t in detail]
        )
        height = 1.22 * (len(label) * LABEL_PT + len(detail) * DETAIL_PT) * scale
        height += 3 * scale if detail else 0
        # 4% margin: glyph metrics from TextToPath and the Agg renderer differ slightly.
        if widest <= 0.96 * width_pt and height <= (h - 2 * PAD) * 72:
            return scale, label, detail
        scale -= 0.04
    return None


def _anchor(rect, side: str, offset: float):
    x, y, w, h = rect  # centre and size
    if side in "ns":
        return x + offset * w, y + (h / 2 if side == "n" else -h / 2)
    return x + (w / 2 if side == "e" else -w / 2), y + offset * h


def _default_sides(a, b):
    (x0, y0), (x1, y1) = a[:2], b[:2]
    if abs(y1 - y0) < 1e-9:
        return ("e", "w") if x1 > x0 else ("w", "e")
    return ("s", "n") if y1 < y0 else ("n", "s")


def _arc_point(start, end, rad: float, s: float) -> tuple[float, float]:
    """Point at parameter s of matplotlib's arc3 quadratic Bezier from start to end."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    control = ((start[0] + end[0]) / 2 + rad * dy, (start[1] + end[1]) / 2 - rad * dx)
    return tuple(
        (1 - s) ** 2 * start[i] + 2 * s * (1 - s) * control[i] + s**2 * end[i] for i in (0, 1)
    )


def _crossed_boxes(start, end, rad: float, rects: dict, endpoints) -> list[str]:
    """Boxes other than the edge's own endpoints whose interior the edge passes through."""
    crossed = []
    for name, (x, y, w, h) in rects.items():
        if name in endpoints:
            continue
        for i in range(1, 200):
            px, py = _arc_point(start, end, rad, i / 200)
            if abs(px - x) < w / 2 and abs(py - y) < h / 2:
                crossed.append(name)
                break
    return crossed


def _overlap(a, b, margin: float = 0.0) -> bool:
    return not (
        a.x1 <= b.x0 + margin
        or b.x1 <= a.x0 + margin
        or a.y1 <= b.y0 + margin
        or b.y1 <= a.y0 + margin
    )


# --------------------------------------------------------------------------- rendering
def render(figure: dict) -> tuple[object, list[str], dict]:
    """Draw one figure; return (matplotlib Figure, layout problems, print-size report).

    The report holds the native ``width`` (inches), the print ``scale``, the ``smallest``
    effective size per text category and the print-readability ``problems``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
    from matplotlib.textpath import TextToPath
    from matplotlib.transforms import Bbox

    text_to_path = TextToPath()

    def measure(text, prop):
        return text_to_path.get_text_width_height_descent(text, prop, ismath=False)[0]

    def font_properties(size, weight):
        return FontProperties(family=FONT, size=size, weight=weight)

    boxes = figure["boxes"]
    cell_w, cell_h = figure.get("cell", [3.0, 1.3])
    columns = max(b["col"] + b.get("span", 1) for b in boxes)
    rows = max(b["row"] for b in boxes) + 1
    fill_w, fill_h = figure.get("box_fill", [0.8, 0.64])
    box_w, box_h = cell_w * fill_w, cell_h * fill_h
    lanes = figure.get("lanes", [])
    gutter = figure.get("lane_width", LANE_GUTTER) if lanes else 0.0
    legend_items = len({e.get("kind", "data") for e in figure["edges"]}) + len(
        {b.get("kind", "api") for b in boxes}
    )
    width = gutter + columns * cell_w
    legend_cols = max(1, min(legend_items, int(width // 2.1)))
    legend_rows = -(-legend_items // legend_cols)
    legend_band = 0.12 + LEGEND_ROW * legend_rows
    height = TITLE_BAND + rows * cell_h + legend_band
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.text(
        0.08,
        height - 0.2,
        f"{figure['id']}  {figure['title']}",
        fontsize=TITLE_PT,
        fontweight="bold",
        ha="left",
        va="center",
    )
    problems: list[str] = []
    rects, text_items = {}, []
    texts = [("title", "figure title", TITLE_PT)]  # (category, name, native pt) for print check
    for lane in lanes:
        first, last = lane["rows"]
        top = height - TITLE_BAND - first * cell_h - 0.03
        bottom = height - TITLE_BAND - (last + 1) * cell_h + 0.03
        ax.add_patch(
            FancyBboxPatch(
                (0.04, bottom),
                width - 0.08,
                top - bottom,
                boxstyle="round,pad=0,rounding_size=0.05",
                facecolor=LANE_STYLE["facecolor"],
                edgecolor=LANE_STYLE["edgecolor"],
                linewidth=0.6,
                zorder=0,
            )
        )
        # The gutter part of the band is an obstacle like a box: edges, edge labels and the
        # legend must keep clear of it, and the rotated label must fit inside it.
        name = f"lane:{lane['label']}"
        rects[name] = ((0.04 + gutter) / 2, (top + bottom) / 2, gutter - 0.04, top - bottom)
        lane_prop = font_properties(LANE_PT, "bold")
        label = _wrap(lane["label"], 0.96 * (top - bottom - 2 * PAD) * 72, lane_prop, measure)
        t = ax.text(
            (0.04 + gutter) / 2,
            (top + bottom) / 2,
            "\n".join(label),
            rotation=90,
            ha="center",
            va="center",
            fontsize=LANE_PT,
            fontweight="bold",
            color=LANE_STYLE["text"],
            linespacing=1.22,
            zorder=3,
        )
        text_items.append((name, t))
        texts.append(("lane", name, LANE_PT))
    for box in boxes:
        span = box.get("span", 1)
        x = gutter + box["col"] * cell_w + span * cell_w / 2
        y = height - TITLE_BAND - (box["row"] * cell_h + cell_h / 2)
        w = box_w + (span - 1) * cell_w
        rects[box["id"]] = (x, y, w, box_h)
        face, edge, _ = KIND_STYLE[box.get("kind", "api")]
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - box_h / 2),
                w,
                box_h,
                boxstyle="round,pad=0,rounding_size=0.07",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.2,
                linestyle=(0, (4, 2)) if box.get("kind") == "optional" else "-",
                zorder=2,
            )
        )
        fitted = _fit(box, w, box_h, measure, font_properties)
        if fitted is None:
            problems.append(f"{figure['id']}/{box['id']}: text does not fit the box")
            continue
        scale, label, detail = fitted
        label_h = 1.22 * len(label) * LABEL_PT * scale / 72
        detail_h = (1.22 * len(detail) * DETAIL_PT * scale + (3 * scale if detail else 0)) / 72
        top = y + (label_h + detail_h) / 2
        t = ax.text(
            x,
            top,
            "\n".join(label),
            ha="center",
            va="top",
            fontsize=LABEL_PT * scale,
            fontweight="bold",
            linespacing=1.22,
            zorder=3,
        )
        text_items.append((box["id"], t))
        texts.append(("title", box["id"], LABEL_PT * scale))
        if detail:
            t = ax.text(
                x,
                top - label_h - 3 * scale / 72,
                "\n".join(detail),
                ha="center",
                va="top",
                fontsize=DETAIL_PT * scale,
                color="#2b2b2b",
                linespacing=1.22,
                zorder=3,
            )
            text_items.append((box["id"], t))
            texts.append(("detail", box["id"], DETAIL_PT * scale))
    edge_labels = []
    for edge in figure["edges"]:
        a, b = rects[edge["from"]], rects[edge["to"]]
        sides = _default_sides(a, b)
        ports = [edge.get("from_port") or sides[0], edge.get("to_port") or sides[1]]
        start = _anchor(a, ports[0][0], float(ports[0][1:] or 0))
        end = _anchor(b, ports[1][0], float(ports[1][1:] or 0))
        style = EDGE_STYLE[edge.get("kind", "data")]
        rad = edge.get("bend", 0.0)
        for name in _crossed_boxes(start, end, rad, rects, (edge["from"], edge["to"])):
            problems.append(f"{figure['id']}: edge {edge['from']}->{edge['to']} crosses box {name}")
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="<|-|>" if edge.get("both") else "-|>",
                mutation_scale=9,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=1,
                shrinkB=1,
                zorder=4,
                **style,
            )
        )
        if edge.get("label"):
            px, py = _arc_point(start, end, rad, edge.get("label_t", 0.5))
            ox, oy = edge.get("label_offset", [0.0, 0.0])
            t = ax.text(
                px + ox,
                py + oy,
                edge["label"],
                fontsize=EDGE_PT,
                ha="center",
                va="center",
                color=style["color"],
                zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.92),
            )
            edge_labels.append((f"{edge['from']}->{edge['to']}", t))
            texts.append(("edge", f"{edge['from']}->{edge['to']}", EDGE_PT))
    handles = [
        Line2D([0], [0], label=EDGE_LEGEND[k], **{**EDGE_STYLE[k], "linewidth": 1.5})
        for k in sorted({e.get("kind", "data") for e in figure["edges"]})
    ]
    kinds = [k for k in KIND_STYLE if k in {b.get("kind", "api") for b in boxes}]
    handles += [
        Patch(
            facecolor=KIND_STYLE[k][0],
            edgecolor=KIND_STYLE[k][1],
            label=KIND_STYLE[k][2],
            linestyle="--" if k == "optional" else "-",
        )
        for k in kinds
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02 / height),
        fontsize=LEGEND_PT,
        frameon=False,
        ncol=legend_cols,
        handlelength=2.4,
        columnspacing=1.4,
        borderaxespad=0.1,
    )
    texts.append(("legend", "legend", LEGEND_PT))

    # Layout verification in display coordinates.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_display = ax.transData.transform
    box_extents = {}
    for name, (x, y, w, h) in rects.items():
        (x0, y0), (x1, y1) = to_display((x - w / 2, y - h / 2)), to_display((x + w / 2, y + h / 2))
        box_extents[name] = Bbox([[x0, y0], [x1, y1]])
    for name, text in text_items:
        if not box_extents[name].contains(*text.get_window_extent(renderer).p0) or not box_extents[
            name
        ].contains(*text.get_window_extent(renderer).p1):
            problems.append(f"{figure['id']}/{name}: text overflows the box")
    label_boxes = [(n, t.get_bbox_patch().get_window_extent(renderer)) for n, t in edge_labels]
    for i, (name, extent) in enumerate(label_boxes):
        for box_name, box_extent in box_extents.items():
            if _overlap(extent, box_extent):
                problems.append(f"{figure['id']}: edge label {name} overlaps box {box_name}")
        for other, other_extent in label_boxes[i + 1 :]:
            if _overlap(extent, other_extent):
                problems.append(f"{figure['id']}: edge labels {name} and {other} overlap")
    legend_extent = legend.get_window_extent(renderer)
    for box_name, box_extent in box_extents.items():
        if _overlap(legend_extent, box_extent):
            problems.append(f"{figure['id']}: legend overlaps box {box_name}")
    figure_extent = fig.bbox
    if legend_extent.x0 < figure_extent.x0 or legend_extent.x1 > figure_extent.x1:
        problems.append(f"{figure['id']}: legend wider than the figure")
    scale, smallest, print_problems = print_sizes(figure["id"], width, texts)
    report = {"width": width, "scale": scale, "smallest": smallest, "problems": print_problems}
    return fig, problems, report


def _size_table(reports) -> str:
    """Per-figure native width and smallest effective text sizes at the print width."""
    header = f"{'fig':<4}{'native_in':>10}{'scale':>7}" + "".join(f"{c:>8}" for c in MIN_PT)
    lines = [header + "  status", "min".ljust(21) + "".join(f"{m:>8g}" for m in MIN_PT.values())]
    for fid, report in reports:
        cells = "".join(
            f"{report['smallest'][c]:>8.2f}" if c in report["smallest"] else f"{'-':>8}"
            for c in MIN_PT
        )
        status = "FAIL" if report["problems"] else "ok"
        lines.append(f"{fid:<4}{report['width']:>10.2f}{report['scale']:>7.3f}{cells}  {status}")
    return "\n".join(lines)


def _index(spec: dict) -> str:
    lines = []
    for figure in spec["figures"]:
        stem = f"figure_{figure['id']}_{figure['slug']}"
        lines += [
            f"### Figure {figure['id']}: {figure['title']}",
            "",
            f"![Figure {figure['id']}](../figures/architecture/{stem}.svg)",
            "",
            f"Files: [`{stem}.svg`](../figures/architecture/{stem}.svg), "
            f"[`{stem}.pdf`](../figures/architecture/{stem}.pdf)",
            "",
            figure["caption"],
            "",
        ]
    return "\n".join(lines)


def _update_readme(spec: dict) -> None:
    start, end = "<!-- figure-index:start -->", "<!-- figure-index:end -->"
    text = README.read_text(encoding="utf-8")
    if start not in text or end not in text:
        raise SystemExit(f"{README} lacks the {start} / {end} markers")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    README.write_text(
        f"{head}{start}\n\n{_index(spec)}\n{end}{tail}", encoding="utf-8", newline="\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate references only")
    parser.add_argument("--preview", type=Path, help="also write PNG previews to this directory")
    parser.add_argument(
        "--report-sizes",
        action="store_true",
        help="print effective text sizes at the print width; writes nothing to docs/",
    )
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    errors = validate(spec)
    if errors:
        print("Architecture specification refers to missing functionality:", file=sys.stderr)
        print("\n".join(f"  {e}" for e in errors), file=sys.stderr)
        return 1
    if args.check:
        items = [i for f in spec["figures"] for i in f["boxes"] + f["edges"]]
        refs = sum(len(i.get("refs", [])) for i in items)
        print(f"{len(spec['figures'])} figures, {refs} references to existing code")
        return 0
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["svg.hashsalt"] = "photographiq"
    matplotlib.rcParams["svg.fonttype"] = "path"
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["font.family"] = FONT
    skip_print = os.environ.get(SKIP_PRINT_ENV) == "1"
    problems, print_problems, reports, rendered = [], [], [], []
    for figure in spec["figures"]:
        fig, issues, report = render(figure)
        problems += issues
        print_problems += report["problems"]
        reports.append((figure["id"], report))
        if args.preview:
            args.preview.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.preview / f"figure_{figure['id']}_{figure['slug']}.png", dpi=110)
        rendered.append((figure, fig))
    if args.report_sizes:
        plt.close("all")
        print(_size_table(reports), flush=True)
        if problems or print_problems:
            print("Problems:", file=sys.stderr)
            print("\n".join(f"  {p}" for p in problems + print_problems), file=sys.stderr)
            return 1
        return 0
    if print_problems and skip_print:
        print(f"{SKIP_PRINT_ENV}=1, print-readability check skipped:", file=sys.stderr)
        print("\n".join(f"  {p}" for p in print_problems), file=sys.stderr)
        print_problems = []
    if problems or print_problems:
        plt.close("all")
        print("Layout problems (nothing written to docs/):", file=sys.stderr)
        print("\n".join(f"  {p}" for p in problems + print_problems), file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    for figure, fig in rendered:
        stem = OUT / f"figure_{figure['id']}_{figure['slug']}"
        # Fixed metadata keeps SVG/PDF output reproducible across runs.
        fig.savefig(stem.with_suffix(".svg"), metadata={"Date": None})
        fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)
    lines = ["# Architecture figure captions", ""]
    for figure in spec["figures"]:
        lines += [f"## Figure {figure['id']}: {figure['title']}", "", figure["caption"], ""]
    (OUT / "captions.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    _update_readme(spec)
    print(f"Wrote {len(spec['figures'])} figures to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
