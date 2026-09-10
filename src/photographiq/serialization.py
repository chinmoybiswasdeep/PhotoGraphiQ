"""Versioned JSON with an allowlisted schema; never eval or pickle."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass

import numpy as np

from . import commands, measurements, resources, states
from .expressions import _OPS, CallableExpression, Expr
from .graph import CVGraph
from .pattern import Pattern

_TYPES = {
    cls.__name__: cls
    for cls in (
        *commands.COMMANDS,
        measurements.Homodyne,
        measurements.Heterodyne,
        measurements.Generaldyne,
        measurements.PhotonNumber,
        states.GaussianInput,
        states.FockInput,
        states.FockSuperposition,
        resources.CatResource,
        resources.CubicPhaseResource,
        Expr,
    )
}


def _encode(value):
    if isinstance(value, CallableExpression) or callable(value):
        raise TypeError("Callables are runtime-only and cannot be serialized")
    if isinstance(value, np.ndarray):
        return _encode(value.tolist())
    if isinstance(value, np.generic):
        return _encode(value.item())
    if is_dataclass(value):
        if type(value).__name__ not in _TYPES:
            raise TypeError("Unsupported dataclass")
        return {
            "type": type(value).__name__,
            "fields": {f.name: _encode(getattr(value, f.name)) for f in fields(value)},
        }
    if isinstance(value, tuple):
        return {"tuple": [_encode(v) for v in value]}
    if isinstance(value, complex):
        return {"complex": [value.real, value.imag]}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        return {"mapping": [[_encode(k), _encode(v)] for k, v in value.items()]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported serialized value {type(value).__name__}")


def _decode(value):
    if isinstance(value, list):
        return [_decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"tuple"}:
        return tuple(_decode(v) for v in value["tuple"])
    if set(value) == {"complex"}:
        return complex(*value["complex"])
    if set(value) == {"mapping"}:
        return {_decode(k): _decode(v) for k, v in value["mapping"]}
    if set(value) == {"type", "fields"} and value["type"] in _TYPES:
        cls = _TYPES[value["type"]]
        result = cls(**{k: _decode(v) for k, v in value["fields"].items()})
        if isinstance(result, Expr):
            arities: dict[str, int | tuple[int, ...]] = {
                "constant": 1,
                "parameter": 1,
                "outcome": (1, 2),
                "neg": 1,
                "sin": 1,
                "cos": 1,
                "exp": 1,
            }
            allowed = arities.get(result.op, 2 if result.op in _OPS else -1)
            if len(result.args) not in ((allowed,) if isinstance(allowed, int) else allowed):
                raise ValueError("Invalid expression schema")
            if result.op in _OPS and not all(isinstance(a, Expr) for a in result.args):
                raise ValueError("Expression operands must be expression nodes")
        return result
    raise ValueError("Unknown JSON value schema")


def dumps(pattern):
    pattern.validate()
    resource = None
    if pattern.resource is not None:
        g = pattern.resource
        resource = {
            "nodes": list(g.network.nodes(data=True)),
            "edges": list(g.network.edges(data=True)),
            "inputs": g.inputs,
            "outputs": g.outputs,
            "ideal": g.ideal,
        }
    return json.dumps(
        {
            "schema": "photographiq",
            "version": 1,
            "inputs": _encode(pattern.inputs),
            "commands": _encode(pattern.commands),
            "resource": _encode(resource),
        },
        indent=2,
        allow_nan=False,
    )


def loads(text):
    def invalid_constant(value):
        raise ValueError(f"Invalid JSON numeric constant: {value}")

    data = json.loads(text, parse_constant=invalid_constant)
    if data.get("schema") != "photographiq" or data.get("version") != 1:
        raise ValueError("Unsupported PhotoGraphiQ schema/version")
    if set(data) != {"schema", "version", "inputs", "commands", "resource"}:
        raise ValueError("Invalid pattern schema fields")
    pattern = Pattern(inputs=_decode(data["inputs"]))
    pattern.extend(_decode(data["commands"]))
    resource = _decode(data["resource"])
    if resource is not None:
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(resource["nodes"])
        graph.add_edges_from(resource["edges"])
        pattern.resource = CVGraph(
            graph, inputs=resource["inputs"], outputs=resource["outputs"], ideal=resource["ideal"]
        )
    return pattern.validate()
