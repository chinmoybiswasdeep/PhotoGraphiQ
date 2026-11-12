"""Execution traces built from real pattern commands and a simulator ``Result``.

The trace is a display of one recorded trajectory: steps follow ``pattern.commands``
in order, measurement steps carry the outcomes the simulator actually recorded, and
every outcome-dependent argument (feed-forward displacement, adaptive angle, signal)
is evaluated from ``Result.records`` exactly as :func:`photographiq.simulate` does.
Postselected measurements are marked as conditional branches, never as deterministic.

With ``simulate(..., frame=True)`` displacements are evaluated identically but the
simulator defers them; the trace shows the command arguments, not the instant at
which the deferred frame was physically applied.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

from ..commands import (
    BeamSplitter,
    CubicPhase,
    Displace,
    Entangle,
    Kerr,
    Loss,
    Measure,
    Output,
    PhotonAdd,
    PhotonSubtract,
    Prepare,
    PrepareResource,
    QuadraticPhase,
    Rotate,
    Signal,
    Squeeze,
    command_dependencies,
    quantum_nodes,
)
from ..expressions import resolve
from ..measurements import Homodyne

_STRUCTURAL = {"node", "u", "v", "nodes", "state", "measurement", "key"}
_KINDS = {
    Prepare: "prepare",
    PrepareResource: "prepare",
    Entangle: "entangle",
    Measure: "measure",
    Signal: "signal",
    Output: "output",
    Loss: "channel",
}
_GATES = (
    Rotate,
    Squeeze,
    BeamSplitter,
    CubicPhase,
    Kerr,
    QuadraticPhase,
    PhotonAdd,
    PhotonSubtract,
)


def _plain(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_plain(v) for v in value]
    return value


@dataclass(frozen=True)
class TraceStep:
    """One executed command.

    Attributes:
        index: Position in ``pattern.commands`` (``len(commands)`` for implicit outputs).
        kind: prepare, entangle, gate, channel, measure, feed-forward, displacement,
            signal or output.
        nodes: Quantum modes acted on.
        label: Short description including evaluated arguments.
        values: Evaluated numeric arguments (e.g. ``{"q": .., "p": ..}``, ``{"angle": ..}``).
        depends_on: Classical keys the command reads.
        outcome_key / outcome: Measurement record key and recorded value.
        statistic: ``Result.measurement_statistics`` entry (kind, value, postselected).
    """

    index: int
    kind: str
    nodes: tuple
    label: str
    values: dict = field(default_factory=dict)
    depends_on: tuple = ()
    outcome_key: Hashable | None = None
    outcome: Any = None
    statistic: dict | None = None

    @property
    def postselected(self) -> bool:
        return bool(self.statistic and self.statistic.get("postselected"))


@dataclass(frozen=True)
class ExecutionTrace:
    """Ordered steps of one recorded trajectory plus run metadata."""

    steps: tuple
    backend: str
    seed: Any
    outcomes: dict
    physical_displacements: int

    @property
    def postselected(self) -> tuple:
        """Keys of postselected (conditional, non-deterministic) measurements."""
        return tuple(s.outcome_key for s in self.steps if s.postselected)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "seed": repr(self.seed),
            "physical_displacements": self.physical_displacements,
            "steps": [
                {
                    "index": s.index,
                    "kind": s.kind,
                    "nodes": list(s.nodes),
                    "label": s.label,
                    "values": {k: _plain(v) for k, v in s.values.items()},
                    "depends_on": list(s.depends_on),
                    "outcome_key": s.outcome_key,
                    "outcome": _plain(s.outcome),
                    "postselected": s.postselected,
                }
                for s in self.steps
            ],
        }


def _format(value: Any) -> str:
    value = _plain(value)
    if isinstance(value, list):
        return "(" + ", ".join(_format(v) for v in value) + ")"
    return f"{value + 0.0:.4g}" if isinstance(value, float) else str(value)  # no "-0"


def execution_trace(pattern, result, *, parameters: Mapping | None = None) -> ExecutionTrace:
    """Build the execution steps of ``pattern`` for the trajectory in ``result``.

    Args:
        pattern: The :class:`~photographiq.Pattern` that was simulated.
        result: Its :class:`~photographiq.Result`.
        parameters: The parameter bindings used for the simulation.

    Raises:
        ValueError: ``result`` does not record exactly this pattern's measurement keys.
    """
    pattern.validate()
    parameters = dict(parameters or {})
    keys = [c.result_key for c in pattern.commands if isinstance(c, Measure)]
    if set(keys) != set(result.outcomes) or len(keys) != len(result.outcomes):
        raise ValueError("The result does not record this pattern's measurements")
    records = result.records
    steps = []
    for index, command in enumerate(pattern.commands):
        values = {
            f.name: resolve(getattr(command, f.name), records, parameters)
            for f in fields(command)
            if f.name not in _STRUCTURAL
            and not (isinstance(command, Prepare) and command.state is not None)
        }
        depends = tuple(sorted(command_dependencies(command), key=str))
        kind = _KINDS.get(type(command), "gate" if isinstance(command, _GATES) else None)
        extra = {}
        if isinstance(command, Displace):
            kind = "feed-forward" if depends else "displacement"
        elif isinstance(command, Measure):
            measurement = command.measurement
            if isinstance(measurement, Homodyne):
                values["angle"] = resolve(measurement.angle, records, parameters)
            key = command.result_key
            extra = {
                "outcome_key": key,
                "outcome": records[key],
                "statistic": result.measurement_statistics.get(key),
            }
        elif isinstance(command, Signal):
            extra = {"outcome_key": command.key, "outcome": records[command.key]}
        if kind is None:
            raise NotImplementedError(f"No trace kind for {type(command).__name__}")
        name = type(command).__name__
        if isinstance(command, Measure):
            name = f"Measure[{type(command.measurement).__name__}]"
        if isinstance(command, Prepare) and command.state is not None:
            name = f"Prepare[{type(command.state).__name__}]"
        args = ", ".join(f"{k}={_format(v)}" for k, v in values.items())
        nodes = quantum_nodes(command)
        label = f"{name} {', '.join(map(str, nodes))}" + (f" ({args})" if args else "")
        if "outcome" in extra:
            label += f" -> {extra['outcome_key']} = {_format(extra['outcome'])}"
            if extra.get("statistic", {}) and extra["statistic"].get("postselected"):
                label += " [postselected: conditional branch]"
        steps.append(TraceStep(index, kind, nodes, label, values, depends, **extra))
    if not any(isinstance(c, Output) for c in pattern.commands):
        outputs = tuple(pattern.outputs)
        steps.append(
            TraceStep(len(pattern.commands), "output", outputs, f"Output {outputs} (implicit)")
        )
    return ExecutionTrace(
        tuple(steps),
        result.backend,
        result.seed,
        dict(result.outcomes),
        result.physical_displacements,
    )
