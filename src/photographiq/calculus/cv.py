"""Command-level calculus view of CV patterns and ideal-limit identity checks.

This is deliberately *not* a CV graphical calculus: no CV-ZX or graphical symplectic algebra
(e.g. Booth-Carette-Comfort) identities are implemented. Two sound tools are provided:

(a) :func:`command_view` classifies the commands of a :class:`photographiq.Pattern` into
    quantum, measurement (quantum -> classical), classical and feed-forward roles, lists the
    continuous (or discrete) outcome variables and extracts displacement corrections as exact
    affine forms ``c0 + sum_k c_k m_k`` of outcomes when the expression is affine; anything else
    is reported as non-affine with the original expression (never approximated).
(b) :func:`ideal_limit_check` evaluates the unconditional Gaussian channel
    ``V -> S V S^T + N`` of a fixed pattern (:func:`photographiq.analysis.gaussian_channel`) at
    increasing resource squeezing and reports the deviation of ``S`` from the ideal symplectic
    map separately from the added noise ``N``. For one wire step ``S = T(k) = [[-k,-1],[1,0]]``
    exactly at every finite ``r`` and ``N = diag(0, e^{-2r})`` (hbar = 2, vacuum V = I).
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..analysis import gaussian_channel
from ..commands import (
    Displace,
    Entangle,
    Measure,
    Prepare,
    Signal,
    command_dependencies,
    quantum_nodes,
)
from ..expressions import CallableExpression, Expr, Outcome, atan2, resolve, sin
from ..measurements import Generaldyne, Heterodyne, Homodyne, PhotonNumber
from ..pattern import Pattern

__all__ = [
    "AffineForm",
    "CVCommandView",
    "CVPatternView",
    "CommandRole",
    "DisplacementCorrection",
    "IdealLimitPoint",
    "IdealLimitReport",
    "OutcomeVariable",
    "affine_form",
    "command_view",
    "ideal_limit_check",
    "wire_step_pattern",
]


class CommandRole(str, enum.Enum):
    QUANTUM = "quantum"  # outcome-independent quantum operation (incl. preparation, Output)
    FEEDFORWARD = "feedforward"  # quantum operation whose parameters read outcomes
    MEASUREMENT = "measurement"  # consumes a mode, produces a classical record
    CLASSICAL = "classical"  # classical post-processing (Signal)


@dataclass(frozen=True)
class CVCommandView:
    index: int
    command: Any
    role: CommandRole
    nodes: tuple
    reads: frozenset
    writes: Any = None


@dataclass(frozen=True)
class OutcomeVariable:
    """A classical record. ``angle`` is the homodyne angle when outcome-independent."""

    key: Any
    node: Any
    measurement: str
    continuous: bool
    components: int
    angle: float | None = None


@dataclass(frozen=True)
class AffineForm:
    """``constant + sum(c * m)`` over coefficients keyed by ``(record key, component)``.

    ``component`` is ``None`` for scalar outcomes and an index for vector outcomes.
    """

    coefficients: dict
    constant: float

    def evaluate(self, records: Mapping) -> float:
        total = self.constant
        for (key, comp), c in self.coefficients.items():
            total += c * (records[key] if comp is None else records[key][comp])
        return total


@dataclass(frozen=True)
class DisplacementCorrection:
    """A ``Displace`` reading outcomes; ``q``/``p`` are AffineForm or the raw expression."""

    index: int
    node: Any
    q: Any
    p: Any

    @property
    def affine(self) -> bool:
        return isinstance(self.q, AffineForm) and isinstance(self.p, AffineForm)


@dataclass(frozen=True)
class CVPatternView:
    commands: tuple[CVCommandView, ...]
    outcomes: tuple[OutcomeVariable, ...]
    corrections: tuple[DisplacementCorrection, ...]


def affine_form(value: Any, parameters: Mapping | None = None) -> AffineForm | None:
    """Exact affine form of ``value`` in outcome records, or ``None`` if not affine/resolvable.

    Outcome-free subexpressions are evaluated with ``parameters``; unbound parameters,
    callables and nonlinear dependence on outcomes give ``None``.
    """
    params = dict(parameters or {})
    if isinstance(value, CallableExpression):
        return None
    if not isinstance(value, Expr):
        return AffineForm({}, resolve(value, {}, {}))
    if not value.dependencies:
        if not value.parameters <= params.keys():
            return None
        return AffineForm({}, value.evaluate({}, params))
    if value.op == "outcome":
        comp = value.args[1] if len(value.args) == 2 else None
        return AffineForm({(value.args[0], comp): 1.0}, 0.0)
    maybe = [affine_form(a, params) for a in value.args]
    parts = [p for p in maybe if p is not None]
    if len(parts) != len(maybe):
        return None
    if value.op == "neg":
        return _scale(parts[0], -1.0)
    if value.op in ("add", "sub"):
        sign = 1.0 if value.op == "add" else -1.0
        a, b = parts
        keys = a.coefficients.keys() | b.coefficients.keys()
        coefficients = {
            k: a.coefficients.get(k, 0.0) + sign * b.coefficients.get(k, 0.0) for k in keys
        }
        return AffineForm(coefficients, a.constant + sign * b.constant)
    if value.op == "mul":
        a, b = parts
        if not a.coefficients:
            return _scale(b, a.constant)
        if not b.coefficients:
            return _scale(a, b.constant)
    if value.op == "div" and not parts[1].coefficients and parts[1].constant != 0:
        return _scale(parts[0], 1.0 / parts[1].constant)
    return None


def _scale(form: AffineForm, c: float) -> AffineForm:
    return AffineForm({k: c * v for k, v in form.coefficients.items()}, c * form.constant)


def _outcome(command: Measure, parameters: Mapping) -> OutcomeVariable:
    m = command.measurement
    angle = None
    if isinstance(m, Homodyne) and not getattr(m.angle, "dependencies", frozenset()):
        form = affine_form(m.angle, parameters)
        angle = None if form is None else form.constant
    continuous = isinstance(m, (Homodyne, Heterodyne, Generaldyne))
    components = 2 if isinstance(m, (Heterodyne, Generaldyne)) else 1
    if not continuous and not isinstance(m, PhotonNumber):
        raise NotImplementedError(f"Unknown measurement {type(m).__name__}")
    return OutcomeVariable(
        command.result_key, command.node, type(m).__name__, continuous, components, angle
    )


def command_view(pattern: Pattern, parameters: Mapping | None = None) -> CVPatternView:
    """Typed view of a CV pattern (see module docstring, part (a))."""
    pattern.validate()
    params = dict(parameters or {})
    views, outcomes, corrections = [], [], []
    for i, c in enumerate(pattern.commands):
        reads = command_dependencies(c)
        if isinstance(c, Measure):
            outcomes.append(_outcome(c, params))
            role, writes = CommandRole.MEASUREMENT, c.result_key
        elif isinstance(c, Signal):
            role, writes = CommandRole.CLASSICAL, c.key
        else:
            role = CommandRole.FEEDFORWARD if reads else CommandRole.QUANTUM
            writes = None
            if isinstance(c, Displace) and reads:
                q, p = affine_form(c.q, params), affine_form(c.p, params)
                corrections.append(
                    DisplacementCorrection(
                        i, c.node, c.q if q is None else q, c.p if p is None else p
                    )
                )
        nodes = () if isinstance(c, Signal) else tuple(quantum_nodes(c))
        views.append(CVCommandView(i, c, role, nodes, reads, writes))
    return CVPatternView(tuple(views), tuple(outcomes), tuple(corrections))


# --------------------------------------------------------------------------- ideal limit


def wire_step_pattern(k: float, squeezing: float) -> Pattern:
    """One CV wire step: input 0, resource 1 (p-squeezed, r), CZ, homodyne at atan2(1, k).

    The q correction ``-m / sin(theta)`` makes the ideal map ``T(k) = [[-k,-1],[1,0]]``.
    """
    angle = atan2(1.0, float(k))
    pattern = Pattern(inputs=(0,))
    pattern.extend([Prepare(1, float(squeezing)), Entangle(0, 1)])
    pattern.measure(0, Homodyne(angle))
    pattern.displace(1, q=-Outcome(0) / sin(angle))
    return pattern.validate()


@dataclass(frozen=True)
class IdealLimitPoint:
    squeezing: float
    matrix: np.ndarray
    noise: np.ndarray
    displacement: np.ndarray
    matrix_error: float  # max |S(r) - S_ideal|
    noise_norm: float  # max |N(r)|
    noise_error: float | None  # max |N(r) - expected(r)| when an expected noise is given


@dataclass(frozen=True)
class IdealLimitReport:
    """Ideal symplectic map versus finite-squeezing added noise, reported separately."""

    target: np.ndarray
    points: tuple[IdealLimitPoint, ...]

    @property
    def max_matrix_error(self) -> float:
        return max(p.matrix_error for p in self.points)

    @property
    def noise_decreasing(self) -> bool:
        norms = [p.noise_norm for p in sorted(self.points, key=lambda p: p.squeezing)]
        return all(b <= a for a, b in zip(norms, norms[1:]))


def ideal_limit_check(
    build: Callable[[float], Pattern],
    target: np.ndarray,
    squeezings: Sequence[float],
    *,
    parameters: Mapping | None = None,
    expected_noise: Callable[[float], np.ndarray] | None = None,
) -> IdealLimitReport:
    """Compare ``gaussian_channel(build(r))`` with an ideal symplectic ``target`` for each r.

    Nothing is extrapolated: the report lists, per squeezing, the exact linear map, its
    deviation from ``target``, the added noise and (optionally) its deviation from an analytic
    ``expected_noise(r)``. Adaptive or non-Gaussian patterns raise as in ``gaussian_channel``.

    The channel is the outcome-averaged (unconditional) map. At finite ``r`` a single corrected
    trajectory generally still depends on its outcome (the measurement partially reads the
    input), so ``matrix_error == 0`` does not mean that each branch implements ``target``.
    """
    target = np.asarray(target, dtype=float)
    points = []
    for r in squeezings:
        channel = gaussian_channel(build(float(r)), parameters=parameters)
        if channel.matrix.shape != target.shape:
            raise ValueError(f"Channel matrix {channel.matrix.shape} vs target {target.shape}")
        noise_error = None
        if expected_noise is not None:
            noise_error = float(np.max(np.abs(channel.noise - expected_noise(float(r)))))
        points.append(
            IdealLimitPoint(
                float(r),
                channel.matrix,
                channel.noise,
                channel.displacement,
                float(np.max(np.abs(channel.matrix - target))),
                float(np.max(np.abs(channel.noise))),
                noise_error,
            )
        )
    if not points:
        raise ValueError("Provide at least one squeezing value")
    return IdealLimitReport(target, tuple(points))
