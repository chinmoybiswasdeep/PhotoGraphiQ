"""Backend-independent commands in execution order."""

from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from .expressions import CallableExpression, Expr


@dataclass(frozen=True)
class Prepare:
    """Prepare a fresh mode; an explicit state overrides resource squeezing.

    Args:
        node (object): Hashable mode label.
        squeezing (float): Finite momentum resource squeezing; nonnegative.
        state (object): Supported state preparation or independent state snapshot.
    """

    node: Any
    squeezing: Any = 1.0
    state: Any = None


@dataclass(frozen=True)
class Entangle:
    """Controlled-Z resource interaction: p_u -> p_u + weight*q_v and conversely.

    Args:
        u (object): First mode label.
        v (object): Second mode label.
        weight (float): Real controlled-Z edge weight.
    """

    u: Any
    v: Any
    weight: Any = 1.0


@dataclass(frozen=True)
class Measure:
    """Destructively measure a mode and store one uniquely keyed classical result.

    Args:
        node (object): Hashable mode label.
        measurement (object): Homodyne, Heterodyne, Generaldyne or PhotonNumber description.
        key (object): Unique classical record key; None uses the measured node.
    """

    node: Any
    measurement: Any
    key: Any = None

    @property
    def result_key(self):
        """Explicit measurement key, or the measured node label when key is None."""
        return self.node if self.key is None else self.key


@dataclass(frozen=True)
class Displace:
    """Translate q and p quadratures by the specified real amounts.

    Args:
        node (object): Hashable mode label.
        q (float): Position translation; hbar=2 quadrature units.
        p (float): Momentum translation; hbar=2 quadrature units.
    """

    node: Any
    q: Any = 0.0
    p: Any = 0.0


@dataclass(frozen=True)
class Rotate:
    """Apply a phase-space rotation by angle radians.

    Args:
        node (object): Hashable mode label.
        angle (float): Quadrature or gate angle in radians; expressions allowed where documented.
    """

    node: Any
    angle: Any


@dataclass(frozen=True)
class Squeeze:
    """Apply single-mode squeezing, with positive r squeezing q.

    Args:
        node (object): Hashable mode label.
        r (float): Dimensionless squeezing parameter.
    """

    node: Any
    r: Any


@dataclass(frozen=True)
class BeamSplitter:
    """Mix two modes using a real beamsplitter angle in radians.

    Args:
        u (object): First mode label.
        v (object): Second mode label.
        theta (float): Beamsplitter mixing angle in radians.
    """

    u: Any
    v: Any
    theta: Any


@dataclass(frozen=True)
class Loss:
    """Attenuate a mode with intensity transmissivity and thermal environment occupation.

    Args:
        node (object): Hashable mode label.
        transmissivity (float): Intensity transmission in [0,1].
        thermal_photons (float): Nonnegative mean environment occupation.
    """

    node: Any
    transmissivity: Any
    thermal_photons: Any = 0.0


@dataclass(frozen=True)
class CubicPhase:
    """Apply exp(i gamma q^3/6), requiring a Fock-capable backend.

    Args:
        node (object): Hashable mode label.
        gamma (float): Cubic coefficient in exp(i gamma q³/6).
    """

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
    """Write an evaluated expression into a uniquely named classical register.

    Args:
        key (object): Unique classical record key; None uses the measured node.
        value (object): Finite real value or supported expression.
    """

    key: Any
    value: Any


@dataclass(frozen=True)
class Output:
    """Select the final ordered surviving output modes; must be the final command.

    Args:
        nodes (tuple): Ordered mode labels.
    """

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
