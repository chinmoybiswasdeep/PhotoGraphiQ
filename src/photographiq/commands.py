"""Backend-independent commands in execution order."""

from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from .expressions import CallableExpression, Expr


@dataclass(frozen=True)
class Prepare:
    node: Any
    squeezing: Any = 1.0
    state: Any = None


@dataclass(frozen=True)
class Entangle:
    u: Any
    v: Any
    weight: Any = 1.0


@dataclass(frozen=True)
class Measure:
    node: Any
    measurement: Any
    key: Any = None

    @property
    def result_key(self):
        return self.node if self.key is None else self.key


@dataclass(frozen=True)
class Displace:
    node: Any
    q: Any = 0.0
    p: Any = 0.0


@dataclass(frozen=True)
class Rotate:
    node: Any
    angle: Any


@dataclass(frozen=True)
class Squeeze:
    node: Any
    r: Any


@dataclass(frozen=True)
class BeamSplitter:
    u: Any
    v: Any
    theta: Any


@dataclass(frozen=True)
class Loss:
    node: Any
    transmissivity: Any
    thermal_photons: Any = 0.0


@dataclass(frozen=True)
class CubicPhase:
    node: Any
    gamma: Any


@dataclass(frozen=True)
class Kerr:
    """exp(i kappa n^2)."""

    node: Any
    kappa: Any


@dataclass(frozen=True)
class QuadraticPhase:
    """exp(i s q^2/4), hence p -> p+s q."""

    node: Any
    s: Any


@dataclass(frozen=True)
class PhotonAdd:
    """Normalized ideal a-dagger operation; not a physical heralding channel."""

    node: Any


@dataclass(frozen=True)
class PhotonSubtract:
    """Normalized ideal a operation; not a physical heralding channel."""

    node: Any


@dataclass(frozen=True)
class PrepareResource:
    """Prepare a correlated sparse pure resource on explicitly ordered nodes."""

    nodes: tuple
    state: Any


@dataclass(frozen=True)
class Signal:
    key: Any
    value: Any


@dataclass(frozen=True)
class Output:
    nodes: tuple


COMMANDS = (
    Prepare,
    Entangle,
    Measure,
    Displace,
    Rotate,
    Squeeze,
    BeamSplitter,
    Loss,
    CubicPhase,
    Kerr,
    QuadraticPhase,
    PhotonAdd,
    PhotonSubtract,
    PrepareResource,
    Signal,
    Output,
)


def expressions(value):
    if isinstance(value, (Expr, CallableExpression)):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from expressions(getattr(value, field.name))


def command_dependencies(command):
    return frozenset().union(*(x.dependencies for x in expressions(command)))


def quantum_nodes(command):
    if isinstance(command, (Entangle, BeamSplitter)):
        return (command.u, command.v)
    if isinstance(command, (Output, PrepareResource)):
        return tuple(command.nodes)
    return (command.node,) if hasattr(command, "node") else ()
