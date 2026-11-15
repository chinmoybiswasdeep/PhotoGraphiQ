"""Measurement-plane and Pauli aware Pauli flow of qubit open graphs.

Definition: Simmons, EPTCS 343 (2021) 50, "Relating measurement patterns to circuits
via Pauli flow", Def. 2.4 (the corrected form of Browne et al. 2007 Def. 5; identical
to Mitosek-Backens 2024, arXiv:2410.23439). p: O^c -> P(I^c) with strict partial order
< such that for every measured u, with labels lambda in {XY, XZ, YZ, X, Y, Z}:
(P1) v in p(u), v != u, lambda(v) not in {X, Y}  =>  u < v;
(P2) v in Odd(p(u)), v != u, lambda(v) not in {Y, Z}  =>  u < v;
(P3) not u < v, v != u, lambda(v) = Y  =>  (v in p(u)  <=>  v in Odd(p(u)));
(P4-P6) XY/XZ/YZ: as the gflow plane conditions;
(P7) X: u in Odd(p(u)); (P8) Z: u in p(u); (P9) Y: u in p(u) xor u in Odd(p(u)).

Finder: Simmons (2021) layered algorithm, i.e. the backward GF(2) layered search of
``gflow.layered_search`` with the P1-P3 relaxations (unsolved X/Y nodes may be
correctors; Y nodes contribute x_v + Odd_v = 0; Z nodes are unconstrained in Odd).
Solvability of a node is monotone in the solved set, so the greedy search is
complete for Pauli-flow existence and returns maximally delayed layers.

Labels: a node is a Pauli measurement only when ``QubitMeasurement.pauli`` is set;
its (plane, angle) must then describe that Pauli observable. Conventions: XY measures
cos(a)X + sin(a)Y, XZ measures cos(a)Z + sin(a)X, YZ measures cos(a)Z + sin(a)Y.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from ..ir import Model, OpenGraph, QubitMeasurement
from ._common import correction_map, flatten, layer_problems, odd_neighbourhood, qubit_specs
from .gflow import layered_search
from .result import FlowAnalysisResult, unsupported

ALGORITHM = "pauli-flow (Simmons 2021 Def. 2.4 and layered algorithm)"

# Plane -> (axis multiplying cos(a), axis multiplying sin(a)).
_AXES = {"XY": ("X", "Y"), "XZ": ("Z", "X"), "YZ": ("Z", "Y")}


def pauli_label(m: QubitMeasurement, *, atol: float = 1e-9) -> str:
    """Pauli label X/Y/Z when ``m.pauli`` is set (checked against plane/angle), else plane."""
    if m.pauli is None:
        return m.plane.value
    cos_axis, sin_axis = _AXES[m.plane.value]
    c, s = math.cos(m.angle), math.sin(m.angle)
    on_axis = (cos_axis == m.pauli and abs(s) < atol) or (sin_axis == m.pauli and abs(c) < atol)
    if not on_axis:
        raise ValueError(
            f"pauli={m.pauli!r} disagrees with plane {m.plane.value} at angle {m.angle}"
        )
    return m.pauli


def verify_pauli_flow(og: OpenGraph, sets: Mapping, layers) -> list[str]:
    """Check domain, codomain and (P1)-(P9) with plain set operations; [] means valid."""
    problems, rank = layer_problems(og, layers)
    labels = {n: pauli_label(m) for n, m in qubit_specs(og).items()}
    if set(sets) != set(og.measured):
        problems.append("correction-set domain differs from the measured nodes")

    def before(a, b) -> bool:
        return rank.get(a, -1) < rank.get(b, -1)

    for u, p in sets.items():
        p = set(p)
        if p & set(og.inputs) or not p <= set(og.nodes):
            problems.append(f"p({u!r}) is not a subset of the non-input nodes")
        odd = odd_neighbourhood(og, p)
        for v in p - {u}:
            if labels.get(v) not in ("X", "Y") and not before(u, v):
                problems.append(f"P1 fails for u={u!r}, v={v!r}")
        for v in odd - {u}:
            if labels.get(v) not in ("Y", "Z") and not before(u, v):
                problems.append(f"P2 fails for u={u!r}, v={v!r}")
        for v, lab in labels.items():
            if lab == "Y" and v != u and not before(u, v) and ((v in p) != (v in odd)):
                problems.append(f"P3 fails for u={u!r}, v={v!r}")
        lab, inp, ino = labels[u], u in p, u in odd
        ok = {
            "XY": not inp and ino,
            "XZ": inp and ino,
            "YZ": inp and not ino,
            "X": ino,
            "Z": inp,
            "Y": inp != ino,
        }[lab]
        if not ok:
            problems.append(f"P4-P9: self condition for label {lab} fails at {u!r}")
    return problems


def find_pauli_flow(og: OpenGraph) -> FlowAnalysisResult:
    """Maximally delayed Pauli flow, verified before returning."""
    if og.model is not Model.QUBIT:
        return unsupported(og.model, ALGORITHM, "Pauli flow is a qubit-model notion")
    labels = {n: pauli_label(m) for n, m in qubit_specs(og).items()}
    sets, layers, unsolved = layered_search(og, labels)
    if unsolved:
        return FlowAnalysisResult(
            "not_certified",
            og.model,
            ALGORITHM,
            diagnostics={
                "reason": "no Pauli flow exists for these labels (the layered search is "
                "complete for Pauli flow)",
                "complete": True,
                "uncorrectable": unsolved,
                "labels": labels,
            },
        )
    problems = verify_pauli_flow(og, sets, layers)
    if problems:  # pragma: no cover - would be an implementation bug
        raise AssertionError(f"Pauli flow failed verification: {problems}")
    order = flatten(layers)
    return FlowAnalysisResult(
        "certified",
        og.model,
        ALGORITHM,
        order,
        layers,
        correction_map(og, sets, order),
        {
            "correction_sets": sets,
            "labels": labels,
            "checks": ("domain", "codomain", "P1", "P2", "P3", "P4-P9"),
        },
        (
            "ideal qubits and noiseless CZ graph state",
            "Pauli labels taken only from QubitMeasurement.pauli",
            "corrections on earlier-measured Pauli nodes act as phases and are omitted",
        ),
        {"complete": True},
    )
