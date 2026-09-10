"""Resource-assisted finite-energy patterns; no universality claim.

These builders use the same causal IR as Gaussian MBQC. Inputs and surviving
outputs are explicitly labelled; the Fock backend must be selected by the caller.
"""

import numpy as np

from .commands import (
    BeamSplitter,
    Displace,
    Entangle,
    Measure,
    Output,
    Prepare,
    QuadraticPhase,
    Rotate,
)
from .expressions import Outcome
from .measurements import Homodyne, PhotonNumber
from .pattern import Pattern
from .resources import CatResource, CubicPhaseResource
from .states import FockInput


def photon_subtraction(theta=0.1, *, input_node="in", ancilla="tap", key="count"):
    """Physical vacuum tap and PNR; select count=1 for heralded subtraction.

    No postselection is implicit. For input |n>, count k has probability
    binomial(n,k) sin(theta)^(2k) cos(theta)^(2(n-k)). The successful map includes
    attenuation and is not identical to normalized ideal a at finite theta.
    """
    if not np.isfinite(theta) or not 0 < theta < np.pi / 2:
        raise ValueError("Tap angle must lie strictly between 0 and pi/2")
    return (
        Pattern(inputs=(input_node,))
        .extend(
            [
                Prepare(ancilla, state=FockInput.number(0)),
                BeamSplitter(input_node, ancilla, theta),
                Measure(ancilla, PhotonNumber(), key),
                Output((input_node,)),
            ]
        )
        .validate()
    )


def resource_injection(resource, *, input_node="in", ancilla="resource", key="m"):
    """Inverse SUM and resource q measurement: psi(q) -> psi(q) phi(q+m).

    The output remains on input_node. This is a finite-resource conditional
    filter, not a deterministic gate for an arbitrary resource.
    """
    return (
        Pattern(inputs=(input_node,))
        .extend(
            [
                Prepare(ancilla, state=resource),
                Rotate(ancilla, np.pi / 2),
                Entangle(input_node, ancilla, -1.0),
                Rotate(ancilla, -np.pi / 2),
                Measure(ancilla, Homodyne.q(), key),
                Output((input_node,)),
            ]
        )
        .validate()
    )


def cat_injection(alpha=1.0, parity=1, **kwargs):
    """Inject a finite cat wavefunction as a conditional multiplicative filter."""
    return resource_injection(CatResource(alpha, parity), **kwargs)


def cubic_injection(gamma, squeezing=0.0, *, input_node="in", ancilla="resource", key="m"):
    r"""Finite-energy cubic injection with quadratic and nonlinear feedforward.

    CP(gamma)=exp(i gamma q^3/6). Given q_anc=m, correcting QP(-2 gamma m)
    and Z(-gamma m^2) leaves CP(gamma) times exp(-(q+m)^2/(4 exp(2r))).
    The envelope is physical filtering; increasing r also increases cutoff cost.
    """
    pattern = resource_injection(
        CubicPhaseResource(gamma, squeezing), input_node=input_node, ancilla=ancilla, key=key
    )
    pattern.commands.pop()
    return pattern.extend(
        [
            QuadraticPhase(input_node, -2 * gamma * Outcome(key)),
            Displace(input_node, p=-gamma * Outcome(key) ** 2),
            Output((input_node,)),
        ]
    ).validate()
