"""Flow and correction overlays, normalized from analysis results or patterns.

Nothing here derives corrections from flow theory: overlays display exactly what a
flow analysis reported (``correction_map``) or what a :class:`~photographiq.Pattern`
actually executes (its outcome-dependent ``Displace`` commands).

Accepted ``correction_map`` (the :class:`photographiq.flows.FlowAnalysisResult`
convention, keyed by the *measured* node whose outcome drives the correction):

* qubit models: ``{source: {"x": targets, "z": targets}}``: X (resp. Z) on every
  target conditioned on the binary outcome s_source; parity-reduced per target;
* CV models: ``{source: {target: {"q": cq, "p": cp}}}``: q_target += cq * m and
  p_target += cp * m with m = record / sin(theta_source), theta_source = atan2(1, k)
  from the graph's ``HomodyneSpec(k)``. Overlays convert to coefficients of the raw
  record m[source] (multiply by sqrt(1 + k^2)) so they match patterns and traces.

A mapping ``{target: Correction}`` is also accepted. Anything else raises ``TypeError``
(e.g. a bare number does not say whether it is a q or p displacement).
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import numpy as np

from ..commands import Displace
from ..expressions import CallableExpression, resolve
from ..ir import HomodyneSpec, Model, OpenGraph

_FLOW_FIELDS = (
    "status",
    "model",
    "algorithm",
    "measurement_order",
    "layers",
    "correction_map",
    "certificate",
    "assumptions",
    "diagnostics",
)


def _fields(value: Any) -> dict | None:
    """Mapping view of a dict, dataclass or attribute object; None for scalars/strings."""
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: getattr(value, f.name) for f in fields(value)}
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes, type)):
        return dict(vars(value))
    return None


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def _key(key: Hashable) -> str:
    return f"m[{key}]"


@dataclass(frozen=True)
class Correction:
    """Correction applied to ``target``, as a function of earlier outcomes.

    Qubit: ``x``/``z`` are the X and Z domains (outcome keys, parity-reduced).
    CV: ``q``/``p`` are ``((key, coefficient), ...)`` with constant parts ``q0``/``p0``.
    """

    target: Hashable
    kind: str  # "qubit" or "cv"
    x: tuple = ()
    z: tuple = ()
    q: tuple = ()
    p: tuple = ()
    q0: float = 0.0
    p0: float = 0.0

    @property
    def sources(self) -> tuple:
        """Outcome keys this correction depends on, in first-appearance order."""
        keys = [k for k in self.x + self.z] + [k for k, _ in self.q + self.p]
        return tuple(dict.fromkeys(keys))

    def terms(self) -> list[tuple[Hashable, str]]:
        """``(source, label)`` per dependency, the label used on correction arrows."""
        out = []
        for source in self.sources:
            if self.kind == "qubit":
                label = "".join(a for a, d in (("X", self.x), ("Z", self.z)) if source in d)
            else:
                parts = [
                    f"{a}:{_fmt(c)}"
                    for a, t in (("q", self.q), ("p", self.p))
                    for k, c in t
                    if k == source
                ]
                label = " ".join(parts)
            out.append((source, label))
        return out

    def formula(self) -> str:
        """Human-readable correction, e.g. ``X^(m[0]) Z^(m[1]+m[2])`` or ``q += 0.5 m[0]``."""
        if self.kind == "qubit":
            parts = [
                f"{axis}^({' xor '.join(_key(k) for k in domain)})"
                for axis, domain in (("X", self.x), ("Z", self.z))
                if domain
            ]
            return " ".join(parts) or "none"

        def linear(constant, terms):
            parts = ([(constant, "")] if constant else []) + [(c, f" {_key(k)}") for k, c in terms]
            text = " ".join(f"{'-' if c < 0 else '+'} {_fmt(abs(c))}{name}" for c, name in parts)
            return text[2:] if text.startswith("+") else "-" + text[2:]

        parts = [
            f"{axis} += {linear(c0, t)}"
            for axis, c0, t in (("q", self.q0, self.q), ("p", self.p0, self.p))
            if t or c0
        ]
        return "; ".join(parts) or "none"

    def evaluate(self, records: Mapping) -> tuple:
        """Qubit: (X exponent, Z exponent) mod 2. CV: (dq, dp) for recorded outcomes."""
        if self.kind == "qubit":
            return tuple(int(sum(int(records[k]) for k in d) % 2) for d in (self.x, self.z))
        return tuple(
            float(c0 + sum(c * float(records[k]) for k, c in t))
            for c0, t in ((self.q0, self.q), (self.p0, self.p))
        )


def _xor(domain: tuple, keys) -> tuple:
    out = list(domain)
    for key in keys:
        if key in out:
            out.remove(key)
        else:
            out.append(key)
    return tuple(out)


def _add(terms: tuple, key: Hashable, coefficient: float) -> tuple:
    table = dict(terms)
    table[key] = table.get(key, 0.0) + float(coefficient)
    return tuple((k, c) for k, c in table.items() if c != 0)


def normalize_corrections(correction_map: Any, graph: OpenGraph) -> dict[Hashable, Correction]:
    """Normalize a ``correction_map`` (see the module docstring) to ``{target: Correction}``."""
    if correction_map is None:
        return {}
    table = _fields(correction_map)
    if table is None:
        raise TypeError("correction_map must be a mapping")
    out: dict[Hashable, Correction] = {}
    for source, value in table.items():
        if isinstance(value, Correction):
            entries = [value]
        elif graph.model.continuous:
            entries = _cv_entries(source, value, graph)
        else:
            entries = _qubit_entries(source, value)
        for entry in entries:
            if entry.kind != ("cv" if graph.model.continuous else "qubit"):
                raise TypeError(f"{entry.kind} correction in a {graph.model.value} correction map")
            old = out.get(entry.target)
            out[entry.target] = entry if old is None else _merge(old, entry)
    return out


def _qubit_entries(source, value) -> list[Correction]:
    spec = _fields(value)
    axes = {str(k).lower(): v for k, v in (spec or {}).items()}
    if spec is None or not set(axes) <= {"x", "z"}:
        raise TypeError(f"Qubit correction of {source!r} must be {{'x': targets, 'z': targets}}")
    entries = [Correction(t, "qubit", x=(source,)) for t in axes.get("x") or ()]
    return entries + [Correction(t, "qubit", z=(source,)) for t in axes.get("z") or ()]


def _cv_entries(source, value, graph: OpenGraph) -> list[Correction]:
    spec = graph.measurements.get(source)
    if not isinstance(spec, HomodyneSpec):
        raise ValueError(f"CV correction source {source!r} needs a HomodyneSpec in the graph")
    targets = _fields(value)
    if targets is None:
        raise TypeError(f"CV correction of {source!r} must map targets to {{'q': cq, 'p': cp}}")
    scale = math.hypot(1.0, float(spec.shear))  # 1 / sin(atan2(1, k))
    entries = []
    for target, displacement in targets.items():
        terms = _fields(displacement)
        if terms is None or not {str(a).lower() for a in terms} <= {"q", "p"}:
            raise TypeError(
                f"CV correction {source!r}->{target!r} must be {{'q': cq, 'p': cp}}, "
                f"not {displacement!r}"
            )
        terms = {str(a).lower(): float(c) * scale for a, c in terms.items() if float(c) != 0}
        entries.append(
            Correction(
                target,
                "cv",
                q=((source, terms["q"]),) if "q" in terms else (),
                p=((source, terms["p"]),) if "p" in terms else (),
            )
        )
    return entries


def _merge(a: Correction, b: Correction) -> Correction:
    if a.kind == "qubit":
        return Correction(a.target, "qubit", x=_xor(a.x, b.x), z=_xor(a.z, b.z))
    q, p = a.q, a.p
    for k, c in b.q:
        q = _add(q, k, c)
    for k, c in b.p:
        p = _add(p, k, c)
    return Correction(a.target, "cv", q=q, p=p, q0=a.q0 + b.q0, p0=a.p0 + b.p0)


def _linear(value, keys: tuple, parameters: Mapping, rng: np.random.Generator):
    """Exact coefficients of an expression affine in scalar outcomes, verified numerically.

    The two-point check is only sound for analytic ``Expr`` trees; opaque callables
    (e.g. piecewise GKP decoders) could pass it while being non-affine, so they raise.
    """
    if isinstance(value, CallableExpression):
        raise ValueError(f"Feed-forward {value!r} is an opaque callable; affinity is unverifiable")
    zero = {k: 0.0 for k in keys}
    constant = resolve(value, zero, parameters)
    coefficients = {k: resolve(value, {**zero, k: 1.0}, parameters) - constant for k in keys}
    for _ in range(2):
        point = {k: float(v) for k, v in zip(keys, rng.normal(scale=3.0, size=len(keys)))}
        exact = resolve(value, point, parameters)
        guess = constant + sum(c * point[k] for k, c in coefficients.items())
        if abs(exact - guess) > 1e-9 * (
            1 + abs(exact) + sum(abs(c) for c in coefficients.values())
        ):
            raise ValueError(f"Feed-forward {value!r} is not affine in outcomes {keys}")
    return constant, {k: c for k, c in coefficients.items() if c != 0}


def pattern_corrections(pattern, parameters: Mapping | None = None) -> dict[Hashable, Correction]:
    """Outcome-dependent ``Displace`` commands of a CV pattern as corrections.

    Coefficients are obtained by evaluating each command's expressions at zero and at
    unit outcomes, then checked at two random points; non-affine feed-forward and
    opaque ``CallableExpression`` feed-forward raise ``ValueError``. Several displacements on one node are summed. Only scalar
    (homodyne/photon-number) outcomes are supported.
    """
    parameters = dict(parameters or {})
    rng = np.random.default_rng(0)
    out: dict[Hashable, Correction] = {}
    for command in pattern.commands:
        if not isinstance(command, Displace):
            continue
        keys = tuple(
            sorted(
                set().union(*(getattr(v, "dependencies", ()) for v in (command.q, command.p))),
                key=str,
            )
        )
        if not keys:
            continue
        try:
            (q0, q), (p0, p) = (_linear(v, keys, parameters, rng) for v in (command.q, command.p))
        except (TypeError, IndexError) as exc:
            raise NotImplementedError("Only scalar outcomes are supported in overlays") from exc
        entry = Correction(command.node, "cv", q=tuple(q.items()), p=tuple(p.items()), q0=q0, p0=p0)
        old = out.get(command.node)
        out[command.node] = entry if old is None else _merge(old, entry)
    return out


@dataclass(frozen=True)
class FlowOverlay:
    """Display data of a flow analysis (duck-typed ``FlowAnalysisResult``)."""

    status: str
    algorithm: str
    model: str
    order: tuple
    layers: tuple
    corrections: dict
    assumptions: tuple
    certificate: Any
    diagnostics: Any

    @classmethod
    def from_result(cls, result: Any, graph: OpenGraph) -> FlowOverlay:
        """Read a ``FlowAnalysisResult``-like object or dict and check it against ``graph``."""
        data = _fields(result)
        if data is None:
            raise TypeError("A flow result must be an object or mapping")
        data = {k: data.get(k) for k in _FLOW_FIELDS}
        model = getattr(data["model"], "value", data["model"]) or graph.model.value
        if Model(model) != graph.model:
            raise ValueError(
                f"Flow result model {model!r} differs from graph {graph.model.value!r}"
            )
        layers = data["layers"]
        if isinstance(layers, Mapping):  # {node: layer index}
            count = max(layers.values(), default=-1) + 1
            layers = [[n for n, i in layers.items() if i == j] for j in range(count)]
        layers = tuple(tuple(layer) for layer in (layers or ()))
        order = tuple(data["measurement_order"] or ())
        unknown = {n for n in order + sum(layers, ()) if n not in graph.graph}
        if unknown:
            raise ValueError(f"Flow result names nodes not in the graph: {unknown!r}")
        corrections = normalize_corrections(data["correction_map"], graph)
        if set(corrections) - set(graph.graph):
            raise ValueError("Correction targets must be graph nodes")
        assumptions = data["assumptions"] or ()
        return cls(
            status=str(getattr(data["status"], "value", data["status"])),
            algorithm=str(data["algorithm"] or "unspecified"),
            model=Model(model).value,
            order=order,
            layers=layers,
            corrections=corrections,
            assumptions=(assumptions,) if isinstance(assumptions, str) else tuple(assumptions),
            certificate=data["certificate"],
            diagnostics=data["diagnostics"],
        )

    def dependencies(self) -> set[tuple[Hashable, Hashable]]:
        """Classical dependency arrows ``(outcome key, corrected node)``."""
        return {(s, t) for t, c in self.corrections.items() for s in c.sources}
