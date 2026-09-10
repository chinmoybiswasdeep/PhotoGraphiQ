"""Fundamental cluster protocols and a Braunstein--Kimble reference pattern."""

import numpy as np

from .commands import BeamSplitter, Displace, Output, Prepare
from .compiler import Circuit, _Builder
from .expressions import Outcome, Parameter
from .measurements import Homodyne
from .pattern import Pattern
from .states import GaussianInput


def wire(shears=(0.0,), *, squeezing=1.0):
    """Cluster wire; a single p measurement teleports a Fourier transform."""
    b = _Builder(1, squeezing)
    for k in shears:
        b.step(0, k)
    b.pattern.append(Output(tuple(b.frontier)))
    return b.pattern.validate()


def identity(*, squeezing=1.0):
    """Construct four Fourier wire steps whose ideal map is identity; finite noise remains.

    Args:
        squeezing (float): Finite momentum resource squeezing; nonnegative.
    """
    return wire([0.0] * 4, squeezing=squeezing)


def displacement(q=0.0, p=0.0, *, squeezing=1.0):
    """Compile quadrature displacement through identity wire transport.

    Args:
        q (float): Position translation; hbar=2 quadrature units.
        p (float): Momentum translation; hbar=2 quadrature units.
        squeezing (float): Finite momentum resource squeezing; nonnegative.
    """
    return Circuit(1).displace(0, q, p).compile(squeezing)


def rotation(angle, *, squeezing=1.0):
    """Return or compile the phase-space rotation in the package sign convention.

    Args:
        angle (float): Quadrature or gate angle in radians; expressions allowed where documented.
        squeezing (float): Finite momentum resource squeezing; nonnegative.
    """
    return Circuit(1).rotate(0, angle).compile(squeezing)


def squeeze(r, *, squeezing=1.0):
    """Append or apply q squeezing by parameter r.

    Args:
        r (float): Dimensionless squeezing parameter.
        squeezing (float): Finite momentum resource squeezing; nonnegative.
    """
    return Circuit(1).squeeze(0, r).compile(squeezing)


def gaussian(matrix, *, squeezing=1.0):
    """Compile a real single-mode symplectic matrix through teleportation steps.

    Args:
        matrix (array-like): Matrix in the documented quadrature or occupation basis.
        squeezing (float): Finite momentum resource squeezing; nonnegative.
    """
    return Circuit(1).gaussian(0, matrix).compile(squeezing)


def entangling(weight=1.0, *, squeezing=1.0):
    """Compile a logical two-mode CZ with transport and finite squeezing.

    Args:
        weight (float): Real controlled-Z edge weight.
        squeezing (float): Finite momentum resource squeezing; nonnegative.
    """
    return Circuit(2).cz(0, 1, weight).compile(squeezing)


def teleportation(*, squeezing=1.0):
    """Braunstein--Kimble teleportation: V_out=V_in+2 exp(-2r) I (ensemble).

    q1-q2 and p1+p2 are squeezed. This optical protocol is distinguished from
    the canonical CZ cluster wire.
    """
    pattern = Pattern(inputs=(0,))
    pattern.extend(
        [
            Prepare(1, state=GaussianInput.squeezed(-squeezing)),
            Prepare(2, state=GaussianInput.squeezed(squeezing)),
            BeamSplitter(1, 2, np.pi / 4),
            BeamSplitter(0, 1, np.pi / 4),
        ]
    )
    pattern.measure(0, Homodyne.q()).measure(1, Homodyne.p())
    pattern.append(Displace(2, np.sqrt(2) * Outcome(0), np.sqrt(2) * Outcome(1)))
    pattern.append(Output((2,)))
    return pattern.validate()


def adaptive(*, squeezing=1.0):
    """Construct a two-step wire whose second setting depends on the first outcome and k.

    Args:
        squeezing (float): Finite momentum resource squeezing; nonnegative.
    """
    return wire((0.0, Parameter("k") + 0.1 * Outcome(0)), squeezing=squeezing)
