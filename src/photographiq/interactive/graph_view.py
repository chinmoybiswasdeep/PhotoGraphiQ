"""Staged interactive view of an :class:`~photographiq.ir.OpenGraph`.

Stages (each a new immutable view):

A. ``construct()``: vertices, weighted CZ edges, input/output/measured/ancilla roles,
   squeezing and measurement metadata; no flow arrows.
B. ``with_flow(result)``: classical dependency arrows (dashed, from the reported
   ``correction_map``), measurement order and layers, certification status and
   assumptions of a ``FlowAnalysisResult``-like object.
C. ``with_corrections(source)``: per-node correction formulas (qubit X/Z domains or CV
   displacements as linear combinations of outcomes m_k) and labelled correction arrows.
D. ``with_execution(trace)``: animated execution steps from
   :func:`~photographiq.interactive.execution_trace`.

Python is authoritative: HTML only displays and edits, and edits are re-validated by
:func:`~photographiq.interactive.apply_edits`. Static figures use Matplotlib only.
"""

from __future__ import annotations

import html
import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import networkx as nx

from ..commands import (
    BeamSplitter,
    Displace,
    Entangle,
    Loss,
    Measure,
    Output,
    Prepare,
    PrepareResource,
    Rotate,
    Signal,
    Squeeze,
    command_dependencies,
    quantum_nodes,
)
from ..expressions import resolve
from ..graph import CVGraph
from ..ir import HomodyneSpec, Model, OpenGraph, QubitMeasurement
from ..measurements import Generaldyne, Heterodyne, Homodyne
from ..pattern import Pattern
from ..states import GaussianInput, GaussianState
from . import _html
from ._template import GRAPH_TEMPLATE
from .layouts import _BEND_STEP, _BEND_TRIES, CLEARANCE, Layout, compute_layout, keyed, route
from .overlays import Correction, FlowOverlay, normalize_corrections, pattern_corrections
from .trace import ExecutionTrace

STAGES = ("construct", "flow", "corrections", "execution")
SCALE = 110.0  # pixels per layout unit in the browser
_GAUSSIAN = (Prepare, Entangle, Displace, Rotate, Squeeze, BeamSplitter, Loss, Signal, Output)


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def measurement_text(spec: Any) -> str:
    """Display text of an OpenGraph measurement specification."""
    if isinstance(spec, HomodyneSpec):
        return f"homodyne angle {_fmt(spec.angle)} rad (shear k={_fmt(spec.shear)})"
    if isinstance(spec, QubitMeasurement):
        pauli = f"Pauli {spec.pauli}, " if spec.pauli else ""
        return f"{pauli}{spec.plane.value} plane, angle {_fmt(spec.angle)}"
    return "unspecified"


def _measurement_json(spec: Any) -> dict | None:
    if isinstance(spec, HomodyneSpec):
        return {"shear": spec.shear}
    if isinstance(spec, QubitMeasurement):
        return {"plane": spec.plane.value, "angle": spec.angle, "pauli": spec.pauli}
    return None


def _wire_step(angle: float) -> HomodyneSpec | None:
    """Shear k with atan2(1, k) == angle mod 2pi (to 1e-12 rad); prefers k rounded to 12 digits."""
    angle %= 2 * math.pi
    if not 0 < angle < math.pi:
        return None
    shear = math.cos(angle) / math.sin(angle)
    for k in (round(shear, 12), shear):
        if abs(HomodyneSpec(k).angle - angle) < 1e-12:
            return HomodyneSpec(k)
    return None


def _pattern_graph(pattern: Pattern, parameters: Mapping) -> tuple[OpenGraph, dict, list]:
    """OpenGraph of a pattern plus per-node annotations and global notes (never silent)."""
    pattern.validate()
    notes: list[str] = []
    annotations: dict[Hashable, list[str]] = {}

    def note(node, text):
        annotations.setdefault(node, []).append(text)

    gaussian = True
    squeezing, measurements = {}, {}
    seen_edges: set = set()
    interleaved: set = set()  # CZ edges with a later non-CZ operation on an endpoint
    network = nx.Graph()
    network.add_nodes_from(pattern.inputs)
    for node in pattern.inputs:
        note(node, "external input")
    for command in pattern.commands:
        if not isinstance(command, (*_GAUSSIAN, Measure)):
            gaussian = False
        if not isinstance(command, Entangle):
            acted = set(quantum_nodes(command))
            interleaved |= {e for e in seen_edges if e & acted}
        if isinstance(command, Prepare):
            network.add_node(command.node)
            if command.state is None:
                squeezing[command.node] = resolve(command.squeezing, {}, parameters)
            else:
                gaussian &= isinstance(command.state, (GaussianInput, GaussianState))
                note(command.node, f"prepared state {type(command.state).__name__}")
        elif isinstance(command, PrepareResource):
            network.add_nodes_from(command.nodes)
            for node in command.nodes:
                note(node, f"correlated resource {type(command.state).__name__}")
        elif isinstance(command, Entangle):
            edge = frozenset((command.u, command.v))
            weight = resolve(command.weight, {}, parameters)
            if edge in seen_edges:
                text = f"Repeated CZ on {tuple(edge)} aggregated into one edge weight"
                if edge in interleaved:
                    text += (
                        " (other operations act on its endpoints in between, so the "
                        "aggregated weight need not equal the executed operation)"
                    )
                notes.append(text)
                if network.has_edge(command.u, command.v):
                    weight += network.edges[command.u, command.v]["weight"]
            seen_edges.add(edge)
            if weight == 0:  # CZ(g) CZ(-g) = identity
                if network.has_edge(command.u, command.v):
                    network.remove_edge(command.u, command.v)
                notes.append(f"CZs on {tuple(edge)} cancel: no edge drawn")
            else:
                network.add_edge(command.u, command.v, weight=weight)
        elif isinstance(command, Measure):
            m = command.measurement
            gaussian &= isinstance(m, (Homodyne, Heterodyne, Generaldyne))
            if command.key is not None:
                note(command.node, f"outcome key {command.key!r}")
            if isinstance(m, Homodyne) and not getattr(m.angle, "dependencies", None):
                angle = resolve(m.angle, {}, parameters)
                spec = _wire_step(angle)
                if spec is not None:
                    measurements[command.node] = spec
                    if m.efficiency != 1 or m.noise:
                        note(command.node, f"efficiency {m.efficiency}, noise {m.noise}")
                    continue
                note(command.node, f"homodyne angle {_fmt(angle)} rad (not a wire-step shear)")
            elif isinstance(m, Homodyne):
                keys = sorted(map(str, m.angle.dependencies))
                note(command.node, f"adaptive homodyne angle depending on {keys}")
            else:
                note(command.node, f"{type(m).__name__} measurement")
        elif isinstance(command, BeamSplitter):
            notes.append(f"BeamSplitter({command.u}, {command.v}) is not a CZ edge; not drawn")
            note(command.u, "beamsplitter (not drawn)")
            note(command.v, "beamsplitter (not drawn)")
        elif isinstance(command, Displace):
            if not command_dependencies(command):  # feed-forward is drawn as corrections
                q, p = (resolve(v, {}, parameters) for v in (command.q, command.p))
                note(command.node, f"constant displacement q={_fmt(q)}, p={_fmt(p)}")
        elif not isinstance(command, (Signal, Output)):
            keys = sorted(map(str, command_dependencies(command)))
            adaptive = f" (adaptive: depends on outcomes {keys})" if keys else ""
            for node in quantum_nodes(command):
                note(node, f"local {type(command).__name__}{adaptive}")
    model = Model.CV_GAUSSIAN if gaussian else Model.CV_NONGAUSSIAN
    graph = OpenGraph(model, network, pattern.inputs, pattern.outputs, measurements, squeezing)
    return graph, {n: "; ".join(t) for n, t in annotations.items()}, notes


@dataclass(frozen=True)
class GraphView:
    """Immutable staged view; every ``with_*``/``construct`` returns a new view."""

    graph: OpenGraph
    layout: Layout | None = None
    stage: str = "construct"
    flow: FlowOverlay | None = None
    corrections: dict | None = None
    trace: ExecutionTrace | None = None
    annotations: dict = field(default_factory=dict)
    notes: tuple = ()
    title: str = "PhotoGraphiQ graph"

    def __post_init__(self):
        if not isinstance(self.graph, OpenGraph):
            raise TypeError(
                "GraphView needs a photographiq.ir.OpenGraph; use from_cvgraph/from_pattern"
            )
        object.__setattr__(self, "graph", self.graph.copy())
        layout = self.layout or compute_layout(self.graph, "publication")
        object.__setattr__(self, "layout", layout.require(self.graph.nodes))
        keyed(self.graph.nodes)
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")

    # ---- constructors -----------------------------------------------------------
    @classmethod
    def from_cvgraph(cls, graph: CVGraph, shears: Mapping | None = None, **options) -> GraphView:
        """View of a :class:`CVGraph`; ``shears`` gives wire-step homodyne shears."""
        return cls(OpenGraph.from_cvgraph(graph, shears), **options)

    @classmethod
    def from_pattern(
        cls, pattern: Pattern, *, parameters: Mapping | None = None, **options
    ) -> GraphView:
        """View of the resource of a pattern.

        Repeated CZs are aggregated, beam splitters and non-wire-step measurements are
        listed as notes/annotations rather than drawn or approximated.
        """
        graph, annotations, notes = _pattern_graph(pattern, dict(parameters or {}))
        return cls(graph, annotations=annotations, notes=tuple(notes), **options)

    # ---- stages -----------------------------------------------------------------
    def construct(self) -> GraphView:
        return replace(self, stage="construct")

    def with_layout(self, layout: Layout | str = "publication", **options) -> GraphView:
        """New view with a :class:`Layout` or a named layout from ``layouts.LAYOUTS``."""
        if isinstance(layout, str):
            if layout == "layered" and not options and self.flow is not None:
                options = (
                    {"layers": self.flow.layers} if self.flow.layers else {"order": self.flow.order}
                )
            layout = compute_layout(self.graph, layout, **options)
        return replace(self, layout=layout)

    def with_flow(self, result: Any) -> GraphView:
        """Overlay a ``FlowAnalysisResult``-like object or dict (stage B)."""
        flow = FlowOverlay.from_result(result, self.graph)
        return replace(self, stage="flow", flow=flow)

    def with_corrections(
        self, source: Any = None, *, parameters: Mapping | None = None
    ) -> GraphView:
        """Overlay corrections (stage C).

        ``source`` may be None (use the flow overlay's correction map), a flow result,
        a :class:`Pattern` (its actual feed-forward displacements) or a correction map.
        """
        flow = self.flow
        if source is None:
            if flow is None:
                raise ValueError("No flow overlay: pass a correction source")
            corrections = flow.corrections
        elif isinstance(source, Pattern):
            corrections = pattern_corrections(source, parameters)
        elif hasattr(source, "correction_map") or (
            isinstance(source, Mapping) and "correction_map" in source
        ):
            flow = FlowOverlay.from_result(source, self.graph)
            corrections = flow.corrections
        else:
            corrections = normalize_corrections(source, self.graph)
        if set(corrections) - set(self.graph.nodes):
            raise ValueError("Correction targets must be graph nodes")
        return replace(self, stage="corrections", flow=flow, corrections=dict(corrections))

    def with_execution(self, trace: ExecutionTrace) -> GraphView:
        """Animate an :class:`ExecutionTrace` (stage D)."""
        unknown = {n for s in trace.steps for n in s.nodes} - set(self.graph.nodes)
        if unknown:
            raise ValueError(f"Trace acts on nodes absent from the graph: {unknown!r}")
        return replace(self, stage="execution", trace=trace)

    # ---- data -------------------------------------------------------------------
    def _stage(self, stage: str | None) -> int:
        stage = stage or self.stage
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        level = STAGES.index(stage)
        needs = {1: self.flow, 2: self.corrections, 3: self.trace}
        if level and needs[level] is None:
            raise ValueError(f"Stage {stage!r} needs its overlay; call with_{stage}() first")
        return level

    def roles(self, node: Hashable) -> list[str]:
        g = self.graph
        return [
            role
            for role, present in (
                ("input", node in g.inputs),
                ("output", node in g.outputs),
                ("measured", node not in g.outputs),
                ("ancilla", node not in g.inputs),
            )
            if present
        ]

    def _arrows(self, level: int) -> tuple[str, list[tuple[Hashable, Hashable, str]]]:
        """Arrow class and (source, target, label) list for a stage level."""
        if level >= 2 and self.corrections is not None:
            arrows = [(s, t, lab) for t, c in self.corrections.items() for s, lab in c.terms()]
            return "correction", arrows
        if level >= 1 and self.flow is not None:
            return "flow", [(s, t, "") for s, t in sorted(self.flow.dependencies(), key=str)]
        return "", []

    def flow_arrows(self) -> set[tuple[Hashable, Hashable]]:
        """Dependency arrows drawn in stage B: ``(outcome key, corrected node)``."""
        return {(s, t) for s, t, _ in self._arrows(1)[1]} if self.flow else set()

    def correction_arrows(self) -> dict[tuple[Hashable, Hashable], str]:
        """Correction arrows drawn in stage C with their labels."""
        return {(s, t): lab for s, t, lab in self._arrows(2)[1]} if self.corrections else {}

    def _ordering(self, level: int) -> tuple[dict, dict]:
        if level < 1 or self.flow is None:
            return {}, {}
        order = {n: i + 1 for i, n in enumerate(self.flow.order)}
        layer = {n: i for i, layer in enumerate(self.flow.layers) for n in layer}
        return order, layer

    def elements(self, stage: str | None = None) -> list[dict]:
        """Cytoscape.js element JSON for ``stage`` (default: the view's stage)."""
        level = self._stage(stage)
        g, ids = self.graph, {n: str(n) for n in self.graph.nodes}
        order, layer = self._ordering(level)
        corrections = self.corrections if level >= 2 and self.corrections else {}
        out: list[dict] = []
        assert self.layout is not None  # __post_init__ always sets a layout
        for node in g.nodes:
            x, y = self.layout[node]
            spec = g.measurements.get(node)
            out.append(
                {
                    "group": "nodes",
                    "data": {
                        "id": ids[node],
                        "raw": _html.jsonable(node),
                        "label": ids[node],
                        "squeezing": g.squeezing.get(node),
                        "meas": _measurement_json(spec),
                        "measText": None if spec is None else measurement_text(spec),
                        "order": order.get(node),
                        "layer": layer.get(node),
                        "formula": corrections[node].formula() if node in corrections else None,
                        "annotation": self.annotations.get(node),
                    },
                    "position": {"x": x * SCALE, "y": -y * SCALE},
                    "classes": " ".join(self.roles(node)),
                }
            )
        for i, (u, v, data) in enumerate(g.graph.edges(data=True)):
            weight = float(data["weight"])
            out.append(
                {
                    "group": "edges",
                    "data": {
                        "id": f"p:{i}",
                        "source": ids[u],
                        "target": ids[v],
                        "weight": weight,
                        "label": _fmt(weight) if g.model.continuous else "",
                        "bend": _bend(self.layout, u, v, "physical"),
                    },
                    "classes": "physical",
                }
            )
        kind, arrows = self._arrows(level)
        for i, (s, t, label) in enumerate(arrows):
            if s not in ids:
                continue  # outcome keys that are not nodes are listed in notes
            description = f"m[{s}] -> {t}" + (f" ({label})" if label else "")
            bend = _bend(self.layout, s, t, kind)
            out.append(
                {
                    "group": "edges",
                    "data": {
                        "id": f"{kind[0]}:{i}",
                        "source": ids[s],
                        "target": ids[t],
                        "label": label,
                        "description": description,
                        "bend": bend,
                        "z": 1000 - min(round(abs(bend)), 999),  # short arcs (labels) on top
                    },
                    "classes": kind + (" uncertified" if self._uncertified(kind) else ""),
                }
            )
        return out

    def _uncertified(self, kind: str) -> bool:
        """Flow arrows of a result that is not ``certified`` are styled as uncertified."""
        return kind == "flow" and self.flow is not None and self.flow.status != "certified"

    def layouts(self) -> dict[str, Layout]:
        """Layouts offered by the page: the view's own, then every applicable named one."""
        options: dict[str, Layout] = {"view": self.layout}  # type: ignore[dict-item]
        for name in ("publication", "layered", "hierarchical", "grid", "circular", "force"):
            if name == "layered":
                if self.flow is None or not (self.flow.layers or self.flow.order):
                    continue
                kwargs = (
                    {"layers": self.flow.layers} if self.flow.layers else {"order": self.flow.order}
                )
                options[name] = compute_layout(self.graph, name, **kwargs)
            else:
                options[name] = compute_layout(self.graph, name)
        return options

    def _notes(self, level: int) -> list[str]:
        notes = list(self.notes)
        unplaced = sorted(
            {str(s) for s, _, _ in self._arrows(level)[1] if s not in self.graph.graph}
        )
        if unplaced:
            notes.append(f"Outcome keys without a graph node (no arrow drawn): {unplaced}")
        if level >= 1 and self.flow and not self.flow.corrections:
            notes.append("The flow result reports no correction map: no dependency arrows")
        if level >= 2 and self.corrections:
            notes += [f"{t}: {c.formula()}" for t, c in self.corrections.items()]
        if level >= 3:
            assert self.trace is not None  # _stage rejects level 3 without a trace
            if self.trace.postselected:
                notes.append(
                    f"Postselected outcomes {list(self.trace.postselected)}: this is one "
                    "conditional branch, not a deterministic run"
                )
        return notes

    def _stage_payload(self, level: int) -> dict:
        """Stage-dependent part of the page data (elements, flow panel, trace, notes)."""
        ids = {n: str(n) for n in self.graph.nodes}
        flow = None
        if level >= 1 and self.flow is not None:
            f = self.flow
            flow = {
                "status": f.status,
                "algorithm": f.algorithm,
                "order": [str(n) for n in f.order],
                "layers": [[str(n) for n in layer] for layer in f.layers],
                "assumptions": [str(a) for a in f.assumptions],
                "certificate": None if f.certificate is None else _short(f.certificate),
                "diagnostics": None if f.diagnostics is None else _short(f.diagnostics),
            }
        trace = None
        if level >= 3:
            assert self.trace is not None  # _stage rejects level 3 without a trace
            trace = {
                "backend": self.trace.backend,
                "steps": [
                    {
                        "label": s.label,
                        "kind": s.kind,
                        "nodes": [ids[n] for n in s.nodes],
                        "sources": [ids[k] for k in s.depends_on if k in ids],
                        "outcome": s.outcome,
                    }
                    for s in self.trace.steps
                ],
            }
        return {
            "stage": STAGES[level],
            "elements": self.elements(STAGES[level]),
            "flow": flow,
            "trace": trace,
            "notes": self._notes(level),
        }

    def available_stages(self) -> tuple[str, ...]:
        """Stages whose overlay exists (always ``construct``)."""
        present = (
            True,
            self.flow is not None,
            self.corrections is not None,
            self.trace is not None,
        )
        return tuple(s for s, ok in zip(STAGES, present) if ok)

    def payload(self, stage: str | None = None, *, editable: bool = True) -> dict:
        """Everything the HTML page needs, as JSON-compatible data.

        The top level describes ``stage``; ``stages`` holds the same stage-dependent
        data for every available stage (the page's stage selector) and ``layouts``
        the positions of :meth:`layouts` in pixels (the layout selector).
        """
        level = self._stage(stage)
        ids = {n: str(n) for n in self.graph.nodes}
        layouts = {
            name: {ids[n]: [x * SCALE, -y * SCALE] for n, (x, y) in layout.positions.items()}
            for name, layout in self.layouts().items()
        }
        return {
            "title": self.title,
            "model": self.graph.model.value,
            "continuous": self.graph.model.continuous,
            "editable": bool(editable),
            "scale": SCALE,
            "inputs": [ids[n] for n in self.graph.inputs],
            "outputs": [ids[n] for n in self.graph.outputs],
            **self._stage_payload(level),
            "stages": {s: self._stage_payload(STAGES.index(s)) for s in self.available_stages()},
            "layouts": layouts,
            "routing": {
                "clearance": CLEARANCE,
                "arc": ARC,
                "step": _BEND_STEP,
                "tries": _BEND_TRIES,
            },
        }

    # ---- HTML -------------------------------------------------------------------
    def to_html(
        self,
        stage: str | None = None,
        *,
        editable: bool = True,
        cytoscape_js: str | Path | None = None,
    ) -> str:
        """Self-contained HTML page; ``cytoscape_js`` inlines a local copy (offline)."""
        return (
            GRAPH_TEMPLATE.replace("__TITLE__", html.escape(self.title))
            .replace("__CYTOSCAPE__", _html.script_tag(_html.CYTOSCAPE_URL, cytoscape_js))
            .replace("__DATA__", _html.script_json(self.payload(stage, editable=editable)))
        )

    def save_html(self, path: str | Path, stage: str | None = None, **options) -> Path:
        return _html.write(self.to_html(stage, **options), path)

    def _repr_html_(self) -> str:
        return _html.iframe(self.to_html(), 620)

    def show(self, stage: str | None = None, **options) -> Path:
        """Open the page in the default web browser; returns the temporary file path."""
        return _html.open_in_browser(self.to_html(stage, **options), "photographiq-graph-")

    # ---- static export ----------------------------------------------------------
    def save(self, path: str | Path, stage: str | None = None, **options) -> Path:
        """Static SVG/PDF/PNG (by suffix) via Matplotlib; see :func:`draw_static`."""
        from .static import draw_static

        return draw_static(self, path, stage, **options)


def _short(value: Any, limit: int = 400) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


__all__ = ["GraphView", "STAGES", "Correction", "measurement_text"]


ARC = 0.3  # default overlay arc: control offset = ARC * chord length (flow +, corrections -)


def _bend(layout: Layout, source: Hashable, target: Hashable, kind: str) -> float:
    """Control-point offset in pixels: the default arc (or straight CZ edge), routed by
    :func:`~photographiq.interactive.layouts.route` around nodes that are not endpoints."""
    (x1, y1), (x2, y2) = layout[source], layout[target]
    base = {"flow": ARC, "correction": -ARC}.get(kind, 0.0) * math.hypot(x2 - x1, y2 - y1)
    return route(layout, source, target, base) * SCALE
