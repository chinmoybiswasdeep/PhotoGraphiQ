"""Hybrid execution planning: classify pattern commands into ordered segments (dry run).

Commands are never reordered: a segment is a maximal run of consecutive commands of the same
class, and every measurement / feed-forward boundary starts a new segment. Classes:

``gaussian-preparation``, ``fock-preparation``, ``non-gaussian-preparation`` (a
``CubicPhaseResource``: squeezed vacuum through the bridge, then the cubic Hamiltonian),
``gaussian-unitary``, ``non-gaussian-unitary``, ``open-system``,
``homodyne``, ``photon-counting``, ``feedforward`` (any unitary with outcome-valued parameters),
``output`` and ``unsupported``. The plan records per-segment engines, live modes, local
dimensions, classical dependencies and a memory estimate, without evolving any state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod

from .. import commands as c
from ..expressions import dependencies
from ..measurements import Homodyne, PhotonNumber
from ..resources import CubicPhaseResource
from ..states import FockInput, GaussianInput

GAUSSIAN_UNITARY = (c.Rotate, c.Squeeze, c.BeamSplitter, c.Entangle, c.QuadraticPhase, c.Displace)
NON_GAUSSIAN = (c.Kerr, c.CubicPhase)
PARAMETERS = ("q", "p", "angle", "r", "theta", "weight", "s", "kappa", "gamma")


def classify(cmd) -> str:
    if isinstance(cmd, c.Prepare):
        if cmd.state is None or isinstance(cmd.state, GaussianInput):
            return "gaussian-preparation"
        if isinstance(cmd.state, FockInput):
            return "fock-preparation"
        if isinstance(cmd.state, CubicPhaseResource):
            return "non-gaussian-preparation"
        return "unsupported"
    if isinstance(cmd, GAUSSIAN_UNITARY + NON_GAUSSIAN):
        if any(dependencies(getattr(cmd, f, 0.0)) for f in PARAMETERS):
            return "feedforward"  # outcome-valued unitary, evaluated on the actual records
        return "gaussian-unitary" if isinstance(cmd, GAUSSIAN_UNITARY) else "non-gaussian-unitary"
    if isinstance(cmd, c.Loss):
        return "open-system"
    if isinstance(cmd, c.Measure):
        if isinstance(cmd.measurement, Homodyne):
            return "homodyne"
        if isinstance(cmd.measurement, PhotonNumber):
            return "photon-counting"
        return "unsupported"
    if isinstance(cmd, c.Output):
        return "output"
    return "unsupported"


ENGINE = {
    "gaussian-preparation": "gaussian-to-fock bridge",
    "fock-preparation": "fock",
    "non-gaussian-preparation": "gaussian-to-fock bridge + hamiltonian",
    "gaussian-unitary": "hamiltonian",
    "non-gaussian-unitary": "hamiltonian",
    "open-system": "lindblad",
    "homodyne": "fock homodyne instrument",
    "photon-counting": "fock photon-number instrument",
    "feedforward": "hamiltonian (outcome-valued)",
    "output": "-",
    "unsupported": "-",
}


@dataclass
class Segment:
    kind: str
    commands: tuple  # command indices
    engine: str
    live_modes: tuple
    reads: tuple = ()
    writes: tuple = ()


@dataclass
class ExecutionPlan:
    segments: list
    dims: dict
    mixed: bool
    unsupported: list = field(default_factory=list)
    peak_dimension: int = 0
    peak_bytes: int = 0
    conversion_boundaries: list = field(default_factory=list)

    @property
    def executable(self):
        return not self.unsupported

    def to_dict(self):
        return {
            "segments": [s.__dict__ for s in self.segments],
            "dims": {repr(k): v for k, v in self.dims.items()},
            "mixed": self.mixed,
            "unsupported": self.unsupported,
            "peak_dimension": self.peak_dimension,
            "peak_bytes": self.peak_bytes,
            "conversion_boundaries": self.conversion_boundaries,
        }


def plan(pattern, dims, *, mixed=None) -> ExecutionPlan:
    """Dry-run plan of ``pattern`` with per-mode local ``dims`` (int or {mode: int})."""
    pattern.validate()
    live = list(pattern.inputs)
    lossy = any(isinstance(cmd, c.Loss) for cmd in pattern.commands)
    mixed = lossy if mixed is None else bool(mixed)
    local = {}

    def dim(m):
        return dims.get(m, dims.get("default")) if isinstance(dims, dict) else int(dims)

    for m in live:
        local[m] = dim(m)
    segments: list[Segment] = []
    unsupported: list = []
    boundaries: list = []
    peak = prod(local[m] for m in live) if live else 1
    for i, cmd in enumerate(pattern.commands):
        kind = classify(cmd)
        if kind == "unsupported":
            unsupported.append((i, type(cmd).__name__))
        if isinstance(cmd, c.Prepare):
            live.append(cmd.node)
            local[cmd.node] = dim(cmd.node)
        if isinstance(cmd, c.Measure):
            live.remove(cmd.node)
        size = prod(local[m] for m in live) if live else 1
        peak = max(peak, size)
        sources = [getattr(cmd, f, 0.0) for f in PARAMETERS] + [
            getattr(getattr(cmd, "measurement", None), "angle", 0.0)
        ]
        reads = tuple(sorted(map(repr, set().union(*map(dependencies, sources)))))
        writes = (repr(cmd.result_key),) if isinstance(cmd, c.Measure) else ()
        if (
            segments
            and segments[-1].kind == kind
            and kind not in ("homodyne", "photon-counting", "feedforward")
        ):
            segments[-1].commands += (i,)
            segments[-1].live_modes = tuple(live)
        else:
            if segments and kind in ("homodyne", "photon-counting"):
                boundaries.append(i)
            segments.append(Segment(kind, (i,), ENGINE[kind], tuple(live), reads, writes))
    bytes_per = 16 * (peak**2 if mixed else peak)
    return ExecutionPlan(segments, local, mixed, unsupported, peak, bytes_per, boundaries)
