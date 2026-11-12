"""Round trip of browser edits and layout files; Python re-validates every edit.

The HTML editor exports ``{"graph": OpenGraph.to_dict(), "layout": {str(node): [x, y]}}``.
:func:`apply_edits` rebuilds the graph through :meth:`OpenGraph.from_dict`, so any
invalid edit (zero weight, weighted qubit edge, measured output, unknown node, ...)
raises exactly as constructing the OpenGraph in Python would.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..ir import OpenGraph
from .layouts import Layout


def _label(value: Any) -> Any:
    """JSON has no tuples: list-valued labels (e.g. grid nodes ``(i, j)``) become tuples."""
    return tuple(_label(v) for v in value) if isinstance(value, list) else value


def _load(data: str | Path | Mapping) -> dict:
    if isinstance(data, Mapping):
        return dict(data)
    text = str(data)
    if not text.lstrip().startswith("{"):
        text = Path(text).read_text(encoding="utf-8")
    return json.loads(text)


def apply_edits(data: str | Path | Mapping) -> tuple[OpenGraph, Layout]:
    """Validate an editor export (JSON text, file path or dict) as ``(OpenGraph, Layout)``.

    Raises:
        KeyError / ValueError / TypeError: The edited graph is not a valid OpenGraph or
            the layout does not cover exactly its nodes.
    """
    payload = _load(data)
    graph = dict(payload["graph"])
    graph["nodes"] = [{**node, "id": _label(node["id"])} for node in graph["nodes"]]
    graph["edges"] = [
        {**e, "source": _label(e["source"]), "target": _label(e["target"])} for e in graph["edges"]
    ]
    graph["inputs"] = [_label(n) for n in graph.get("inputs", ())]
    graph["outputs"] = [_label(n) for n in graph.get("outputs", ())]
    open_graph = OpenGraph.from_dict(graph)
    layout = Layout.from_dict(payload.get("layout", {}), open_graph.nodes)
    return open_graph, layout.require(open_graph.nodes)


def export_layout(layout: Layout, path: str | Path | None = None) -> str:
    """Layout JSON ``{str(node): [x, y]}``; also written to ``path`` when given."""
    text = layout.to_json()
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def import_layout(source: str | Path | Mapping, graph: OpenGraph) -> Layout:
    """Read layout JSON (text, path or dict) for ``graph``; every node must be placed."""
    data = _load(source)
    return Layout.from_dict(data, graph.nodes).require(graph.nodes)
