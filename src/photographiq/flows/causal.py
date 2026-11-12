"""Causal flow of qubit open graphs.

Definition (Danos and Kashefi, PRA 74, 052310, 2006): a causal flow is a map
f: O^c -> I^c and a strict partial order < such that for every measured i
(F1) f(i) ~ i, (F2) i < f(i), (F3) every neighbour k != i of f(i) has i < k.
Causal flow is defined for XY-plane measurements only.

Finder: Mhalla and Perdrix, ICALP 2008 (LNCS 5125), Algorithm 1 (equivalently de
Beaudrap's characterisation). It is complete: when it stops early, no causal flow
exists. It returns the maximally delayed flow, layered backwards from the outputs.
Complexity O(n * m) for n nodes and m edges in this implementation.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..ir import Model, OpenGraph, Plane
from ._common import correction_map, flatten, layer_problems, qubit_specs
from .result import FlowAnalysisResult, unsupported

ALGORITHM = "causal-flow (Danos-Kashefi 2006; Mhalla-Perdrix 2008 Alg. 1)"


def verify_causal_flow(og: OpenGraph, flow: Mapping, layers) -> list[str]:
    """Check (F1)-(F3), domain/codomain and injectivity directly; [] means valid."""
    problems, rank = layer_problems(og, layers)
    specs = qubit_specs(og)
    for u, m in specs.items():
        if m.plane is not Plane.XY:
            problems.append(f"node {u!r} is measured in {m.plane.value}, not XY")
    if set(flow) != set(og.measured):
        problems.append("flow domain differs from the measured nodes")
    if len(set(flow.values())) != len(flow):
        problems.append("flow is not injective")
    for u, v in flow.items():
        if v in og.inputs or v not in og.graph:
            problems.append(f"f({u!r})={v!r} is not a non-input node")
            continue
        if not og.graph.has_edge(u, v):
            problems.append(f"F1: f({u!r})={v!r} is not a neighbour")
        if not rank.get(u, -1) < rank.get(v, -1):
            problems.append(f"F2: {u!r} is not before f({u!r})={v!r}")
        for k in og.graph.neighbors(v):
            if k != u and not rank.get(u, -1) < rank.get(k, -1):
                problems.append(f"F3: neighbour {k!r} of f({u!r}) is not after {u!r}")
    return problems


def find_causal_flow(og: OpenGraph) -> FlowAnalysisResult:
    """Maximally delayed causal flow of a qubit open graph, verified before returning."""
    if og.model is not Model.QUBIT:
        return unsupported(og.model, ALGORITHM, "causal flow is a qubit-model notion")
    specs = qubit_specs(og)
    non_xy = [n for n, m in specs.items() if m.plane is not Plane.XY]
    if non_xy:
        return FlowAnalysisResult(
            "not_certified",
            og.model,
            ALGORITHM,
            diagnostics={
                "reason": f"causal flow is defined for XY measurements only; non-XY: {non_xy}",
                "complete": True,
            },
        )
    processed = set(og.outputs)
    correctors = [n for n in og.outputs if n not in og.inputs]
    flow: dict = {}
    layers: list[tuple] = []
    while len(processed) < og.graph.number_of_nodes():
        layer: list = []
        used: set = set()
        for v in correctors:
            open_neighbours = [n for n in og.graph.neighbors(v) if n not in processed]
            if len(open_neighbours) == 1 and open_neighbours[0] not in flow:
                flow[open_neighbours[0]] = v
                layer.append(open_neighbours[0])
                used.add(v)
        if not layer:
            return FlowAnalysisResult(
                "not_certified",
                og.model,
                ALGORITHM,
                diagnostics={
                    "reason": "no causal flow exists (the Mhalla-Perdrix finder is complete)",
                    "complete": True,
                    "uncorrectable": tuple(n for n in og.nodes if n not in processed),
                },
            )
        processed.update(layer)
        correctors = [n for n in correctors if n not in used] + [
            n for n in layer if n not in og.inputs
        ]
        layers.insert(0, tuple(n for n in og.nodes if n in layer))
    layers_t = tuple(layers)
    problems = verify_causal_flow(og, flow, layers_t)
    if problems:  # pragma: no cover - would be an implementation bug
        raise AssertionError(f"causal flow failed verification: {problems}")
    order = flatten(layers_t)
    sets = {u: {v} for u, v in flow.items()}
    return FlowAnalysisResult(
        "certified",
        og.model,
        ALGORITHM,
        order,
        layers_t,
        correction_map(og, sets, order),
        {"flow": dict(flow), "checks": ("domain", "codomain", "injective", "F1", "F2", "F3")},
        ("ideal qubits and noiseless CZ graph state", "XY-plane measurements"),
        {"complete": True},
    )
