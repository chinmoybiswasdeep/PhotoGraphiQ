"""Sequential adaptive execution and independently seeded trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .backends import BaseBackend, GaussianBackend, PiquassoBackend
from .commands import (
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
)
from .expressions import resolve
from .gaussian import beamsplitter, cz, rotation, squeezing
from .measurements import Generaldyne, Heterodyne, Homodyne
from .states import GaussianState


@dataclass
class Result:
    outcomes: dict
    records: dict
    state: Any
    backend: str
    seed: object
    physical_displacements: int = 0
    measurement_statistics: dict = field(default_factory=dict)

    @property
    def log_likelihood(self):
        """Log product of recorded conditional probabilities/densities.

        If homodyne is present this is a joint density, not an event probability.
        Gaussian backends currently do not populate this diagnostic.
        """
        if len(self.measurement_statistics) != len(self.outcomes):
            raise NotImplementedError("Backend did not report every measurement likelihood")
        return float(sum(np.log(item["value"]) for item in self.measurement_statistics.values()))


@dataclass
class ShotResult:
    trajectories: list[Result]

    def values(self, key):
        return np.asarray([r.records[key] for r in self.trajectories])

    def ensemble_state(self):
        """Moment-matched state of the ensemble; mixture need not be Gaussian."""
        if not self.trajectories:
            raise ValueError("Empty ensemble")
        states = [r.state for r in self.trajectories]
        if not all(isinstance(s, GaussianState) for s in states):
            raise NotImplementedError("Gaussian moments require Gaussian trajectory states")
        means = np.array([s.mean for s in states])
        mean = means.mean(axis=0)
        cov = np.mean([s.covariance for s in states], axis=0)
        centered = means - mean
        cov += centered.T @ centered / len(states)
        return GaussianState(mean, cov, states[0].nodes)


def _backend(backend, cutoff):
    if isinstance(backend, BaseBackend):
        return backend
    if backend == "piquasso":
        return PiquassoBackend()
    if backend == "gaussian":
        return GaussianBackend()
    if backend == "piquasso-fock":
        from .backends.fock import PiquassoFockBackend

        return PiquassoFockBackend(cutoff=cutoff)
    raise ValueError(f"Unknown backend: {backend!r}")


def simulate(
    pattern,
    *,
    backend="piquasso",
    parameters=None,
    inputs=None,
    seed=None,
    frame=False,
    cutoff=None,
    initial_state=None,
    measurement_outcomes=None,
):
    """Execute one conditional trajectory. Missing inputs default to vacuum.

    ``frame=True`` defers Gaussian displacements, propagates them through gates,
    and materializes them before measurements or non-Gaussian operations.
    """
    from .states import GaussianInput

    pattern.validate()
    parameters, inputs = dict(parameters or {}), dict(inputs or {})
    missing = pattern.parameters - parameters.keys()
    if missing:
        raise ValueError(f"Unbound parameters: {sorted(missing)}")
    if not inputs.keys() <= set(pattern.inputs):
        raise ValueError("Input supplied for a non-input node")
    engine = _backend(backend, cutoff)
    from .capabilities import preflight
    from .states import FockSuperposition

    measurement_outcomes = dict(measurement_outcomes or {})
    preflight(engine, pattern, inputs, initial_state, measurement_outcomes)
    if frame and not isinstance(engine, GaussianBackend):
        raise NotImplementedError("Exact displacement-frame tracking requires a Gaussian backend")
    engine.reset(seed)
    if isinstance(initial_state, FockSuperposition):
        if inputs:
            raise ValueError("Do not combine correlated and individual inputs")
        engine.prepare_resource(pattern.inputs, initial_state)
    elif initial_state is not None:
        if (
            inputs
            or not isinstance(initial_state, GaussianState)
            or initial_state.nodes != pattern.inputs
        ):
            raise ValueError(
                "Provide a GaussianState ordered by pattern inputs, without individual inputs"
            )
        if not isinstance(engine, GaussianBackend):
            raise NotImplementedError("Correlated Gaussian injection requires a Gaussian backend")
        engine.set_state(initial_state)
    else:
        for node in pattern.inputs:
            engine.prepare(node, squeezing=0, state=inputs.get(node, GaussianInput()))
    outcomes: dict = {}
    records: dict = {}
    statistics: dict = {}
    pending: dict = {}
    applied = 0

    def value(x):
        return resolve(x, records, parameters)

    def flush(node):
        nonlocal applied
        q, p = pending.pop(node, np.zeros(2))
        if q != 0 or p != 0:
            engine.displace(node, q, p)
            applied += 1

    def propagate(nodes, matrix):
        if frame:
            vector = np.concatenate([pending.get(n, np.zeros(2)) for n in nodes])
            vector = matrix @ vector
            for i, n in enumerate(nodes):
                pending[n] = vector[2 * i : 2 * i + 2]

    for c in pattern.commands:
        if isinstance(c, Prepare):
            engine.prepare(c.node, value(c.squeezing), c.state)
        elif isinstance(c, PrepareResource):
            engine.prepare_resource(c.nodes, c.state)
        elif isinstance(c, Entangle):
            g = value(c.weight)
            engine.entangle(c.u, c.v, g)
            propagate((c.u, c.v), cz(g))
        elif isinstance(c, Displace):
            q, p = value(c.q), value(c.p)
            if frame:
                pending[c.node] = pending.get(c.node, np.zeros(2)) + [q, p]
            else:
                engine.displace(c.node, q, p)
                applied += 1
        elif isinstance(c, Rotate):
            angle = value(c.angle)
            engine.rotate(c.node, angle)
            propagate((c.node,), rotation(angle))
        elif isinstance(c, Squeeze):
            r = value(c.r)
            engine.squeeze(c.node, r)
            propagate((c.node,), squeezing(r))
        elif isinstance(c, BeamSplitter):
            theta = value(c.theta)
            engine.beamsplitter(c.u, c.v, theta)
            propagate((c.u, c.v), beamsplitter(theta))
        elif isinstance(c, Measure):
            angle = value(c.measurement.angle) if isinstance(c.measurement, Homodyne) else 0.0
            virtual = (
                frame
                and isinstance(engine, GaussianBackend)
                and isinstance(c.measurement, (Homodyne, Heterodyne, Generaldyne))
            )
            shift = pending.pop(c.node, np.zeros(2)) if virtual else np.zeros(2)
            if not virtual:
                flush(c.node)
            if c.result_key in measurement_outcomes:
                record = engine.measure(
                    c.node, c.measurement, angle, outcome=measurement_outcomes[c.result_key]
                )
            else:
                record = engine.measure(c.node, c.measurement, angle)
            if engine.supports("fock_input"):
                statistics[c.result_key] = dict(engine.last_measurement)
                statistics[c.result_key]["postselected"] = c.result_key in measurement_outcomes
            if virtual:
                if isinstance(c.measurement, Homodyne):
                    record += float(np.array([np.cos(angle), np.sin(angle)]) @ shift)
                else:
                    record = tuple(np.asarray(record) + shift)
            outcomes[c.result_key] = records[c.result_key] = record
        elif isinstance(c, Signal):
            records[c.key] = value(c.value)
        elif isinstance(c, Loss):
            eta = value(c.transmissivity)
            engine.loss(c.node, eta, value(c.thermal_photons))
            propagate((c.node,), np.eye(2) * np.sqrt(eta))
        elif isinstance(c, CubicPhase):
            flush(c.node)
            engine.cubic_phase(c.node, value(c.gamma))
        elif isinstance(c, Kerr):
            flush(c.node)
            engine.kerr(c.node, value(c.kappa))
        elif isinstance(c, QuadraticPhase):
            s = value(c.s)
            engine.quadratic_phase(c.node, s)
            propagate((c.node,), np.array([[1, 0], [s, 1]]))
        elif isinstance(c, (PhotonAdd, PhotonSubtract)):
            flush(c.node)
            engine.ladder(c.node, isinstance(c, PhotonAdd))
        elif not isinstance(c, Output):
            raise NotImplementedError(type(c).__name__)
    for node in tuple(pending):
        flush(node)
    return Result(
        outcomes,
        records,
        engine.get_state(pattern.outputs),
        type(engine).__name__,
        seed,
        applied,
        statistics,
    )


def run_shots(pattern, shots: int, *, seed=None, **kwargs):
    if not isinstance(shots, int) or isinstance(shots, bool) or shots < 1:
        raise ValueError("shots must be a positive integer")
    seeds = np.random.SeedSequence(seed).spawn(shots)
    return ShotResult([simulate(pattern, seed=s, **kwargs) for s in seeds])


sample = run_shots
