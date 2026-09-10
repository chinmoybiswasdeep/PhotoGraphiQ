"""Deterministic unconditional channels for fixed-angle, affine Gaussian patterns.

This propagates linear Wigner variables, not conditional simulator states. It is
independent of the trajectory backend and exposes the actual finite-resource noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .commands import (
    BeamSplitter,
    Displace,
    Entangle,
    Loss,
    Measure,
    Output,
    Prepare,
    Rotate,
    Signal,
    Squeeze,
)
from .expressions import CallableExpression, Expr, resolve
from .gaussian import beamsplitter, cz, rotation, squeezing
from .measurements import Homodyne
from .states import GaussianInput, GaussianState


@dataclass
class GaussianChannel:
    """Affine unconditional Gaussian channel: mean -> S mean + d, V -> S V S.T + N.

    Args:
        matrix (array-like): Matrix in the documented quadrature or occupation basis.
        noise (ndarray): Added output covariance matrix N.
        displacement (object): Output quadrature displacement vector d.
        inputs (tuple): Ordered input labels supplied externally.
        outputs (tuple): Ordered surviving output labels.

    Raises:
        ValueError: Input state ordering differs from channel.
    """

    matrix: np.ndarray
    noise: np.ndarray
    displacement: np.ndarray
    inputs: tuple
    outputs: tuple

    def apply(self, state):
        """Apply this affine channel to an input with matching ordered node labels.

        Args:
            state (object): Supported state preparation or independent state snapshot.

        Raises:
            ValueError: Input state ordering differs from channel.
        """
        if state.nodes != self.inputs:
            raise ValueError("Input state ordering differs from channel")
        return GaussianState(
            self.matrix @ state.mean + self.displacement,
            self.matrix @ state.covariance @ self.matrix.T + self.noise,
            self.outputs,
        )


def gaussian_channel(pattern, *, parameters=None):
    """Analyze a fixed Gaussian pattern exactly; reject adaptive/nonlinear rules."""
    pattern.validate()
    params = dict(parameters or {})
    ni = 2 * len(pattern.inputs)
    # Two source variables for every preparation/channel/detector is sufficient.
    width = ni + 2 * len(pattern.commands)
    covariance = np.zeros((width, width))
    source_mean = np.zeros(width)
    active: dict = {}
    records: dict = {}
    for i, node in enumerate(pattern.inputs):
        active[node] = np.eye(width)[2 * i : 2 * i + 2]
    cursor = ni

    def scalar(x):
        if getattr(x, "dependencies", ()):
            raise NotImplementedError("Outcome-dependent Gaussian gates require trajectories")
        return resolve(x, {}, params)

    def affine(x):
        # Constant coefficient is stored in a final extra entry, outside source space.
        if isinstance(x, CallableExpression):
            raise NotImplementedError("Callable channel analysis is unsupported")
        if not isinstance(x, Expr) or not x.dependencies:
            result = np.zeros(width + 1)
            result[-1] = scalar(x)
            return result
        if x.op == "outcome":
            if len(x.args) != 1:
                raise NotImplementedError("Vector outcome analysis is unsupported")
            return records[x.args[0]].copy()
        values = [affine(a) for a in x.args]
        if x.op == "add":
            return values[0] + values[1]
        if x.op == "sub":
            return values[0] - values[1]
        if x.op == "neg":
            return -values[0]
        if x.op == "mul":
            if not np.any(values[0][:-1]):
                return values[0][-1] * values[1]
            if not np.any(values[1][:-1]):
                return values[1][-1] * values[0]
        if x.op == "div" and not np.any(values[1][:-1]):
            return values[0] / values[1][-1]
        raise NotImplementedError(
            "The unconditional Gaussian analyzer requires affine feed-forward"
        )

    offsets = {n: np.zeros(2) for n in pattern.inputs}
    for c in pattern.commands:
        if isinstance(c, Prepare):
            state = c.state if c.state is not None else GaussianInput.squeezed(-scalar(c.squeezing))
            if not isinstance(state, GaussianInput):
                raise NotImplementedError("Non-Gaussian preparation")
            state.state(c.node)
            active[c.node] = np.eye(width)[cursor : cursor + 2]
            covariance[cursor : cursor + 2, cursor : cursor + 2] = state.covariance
            source_mean[cursor : cursor + 2] = state.mean
            offsets[c.node] = np.zeros(2)
            cursor += 2
        elif isinstance(c, (Entangle, BeamSplitter)):
            nodes = (c.u, c.v)
            s = cz(scalar(c.weight)) if isinstance(c, Entangle) else beamsplitter(scalar(c.theta))
            rows = s @ np.vstack([active[n] for n in nodes])
            shift = s @ np.concatenate([offsets[n] for n in nodes])
            for i, n in enumerate(nodes):
                active[n], offsets[n] = rows[2 * i : 2 * i + 2], shift[2 * i : 2 * i + 2]
        elif isinstance(c, (Rotate, Squeeze)):
            s = rotation(scalar(c.angle)) if isinstance(c, Rotate) else squeezing(scalar(c.r))
            active[c.node], offsets[c.node] = s @ active[c.node], s @ offsets[c.node]
        elif isinstance(c, Displace):
            shift = np.vstack([affine(c.q), affine(c.p)])
            active[c.node] += shift[:, :-1]
            offsets[c.node] += shift[:, -1]
        elif isinstance(c, Measure):
            if not isinstance(c.measurement, Homodyne):
                raise NotImplementedError("Analyzer supports homodyne only")
            theta = scalar(c.measurement.angle)
            h = np.array([np.cos(theta), np.sin(theta)])
            vector = h @ active.pop(c.node)
            vector[cursor] += 1
            covariance[cursor, cursor] = (
                1 - c.measurement.efficiency
            ) / c.measurement.efficiency + c.measurement.noise
            cursor += 1
            records[c.result_key] = np.r_[vector, h @ offsets.pop(c.node)]
        elif isinstance(c, Signal):
            records[c.key] = affine(c.value)
        elif isinstance(c, Loss):
            eta, thermal = scalar(c.transmissivity), scalar(c.thermal_photons)
            if not 0 <= eta <= 1 or thermal < 0:
                raise ValueError("Invalid channel parameters")
            active[c.node] *= np.sqrt(eta)
            offsets[c.node] *= np.sqrt(eta)
            active[c.node][:, cursor : cursor + 2] += np.eye(2)
            covariance[cursor : cursor + 2, cursor : cursor + 2] = (
                (1 - eta) * (2 * thermal + 1) * np.eye(2)
            )
            cursor += 2
        elif not isinstance(c, Output):
            raise NotImplementedError(type(c).__name__)
    outputs = pattern.outputs
    rows = np.vstack([active[n] for n in outputs]) if outputs else np.zeros((0, width))
    offsets_vector = np.concatenate([offsets[n] for n in outputs]) if outputs else np.zeros(0)
    return GaussianChannel(
        rows[:, :ni],
        rows @ covariance @ rows.T,
        rows @ source_mean + offsets_vector,
        pattern.inputs,
        outputs,
    )
