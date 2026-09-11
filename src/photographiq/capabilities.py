"""Preflight the whole pattern before allocating or mutating simulator state."""

import numpy as np

from . import commands as c
from .resources import CatResource, CubicPhaseResource
from .states import FockDensityMatrix, FockInput, FockSuperposition, GaussianInput, GaussianState


def preflight(engine, pattern, inputs, initial_state=None, measurement_outcomes=None):
    features = set()
    states = list(inputs.values())
    if initial_state is not None:
        states.append(initial_state)
    feature_commands = {
        c.CubicPhase: "cubic_phase",
        c.Kerr: "kerr",
        c.QuadraticPhase: "quadratic_phase",
        c.PhotonAdd: "photon_addition",
        c.PhotonSubtract: "photon_subtraction",
        c.Loss: "loss",
        c.PrepareResource: "multimode_fock",
    }
    keys = set()
    for command in pattern.commands:
        if type(command) in feature_commands:
            features.add(feature_commands[type(command)])
        if isinstance(command, (c.Prepare, c.PrepareResource)):
            states.append(command.state)
        if isinstance(command, c.Measure):
            keys.add(command.result_key)
            measurement = command.measurement
            features.update(measurement.required_capabilities)
    for state in states:
        from .gkp import GKPResource

        if isinstance(state, GKPResource):
            features.add("fock_input")
        if isinstance(state, FockDensityMatrix):
            features.add("mixed_fock")
        if isinstance(state, (FockInput, FockSuperposition)):
            features.add("fock_input")
        if isinstance(state, FockSuperposition):
            features.add("multimode_fock")
        if isinstance(state, CatResource):
            features.add("cat_state")
        if isinstance(state, CubicPhaseResource):
            features.add("cubic_phase")
        if (
            engine.supports("fock_input")
            and not engine.supports("mixed_fock")
            and isinstance(state, GaussianInput)
            and not np.isclose(np.linalg.det(state.covariance), 1, atol=1e-10, rtol=0)
        ):
            raise NotImplementedError("Mixed Gaussian inputs require a mixed Fock backend")
    if measurement_outcomes:
        if not measurement_outcomes.keys() <= keys:
            raise ValueError("Postselection key does not identify a measurement")
        features.add("postselection")
    engine.require(*features)
    for command in pattern.commands:
        if isinstance(command, c.Measure):
            command.measurement.validate_backend(engine)
    # Do this after capability checks but before reset or any preparation.
    if initial_state is not None:
        if inputs:
            raise ValueError("Do not combine correlated and individual inputs")
        if isinstance(initial_state, GaussianState):
            if engine.supports("fock_input"):
                raise NotImplementedError(
                    "Correlated Gaussian injection requires a Gaussian backend"
                )
        else:
            engine.validate_preparation(initial_state, len(pattern.inputs))
    for state in inputs.values():
        engine.validate_preparation(state)
    for command in pattern.commands:
        if isinstance(command, (c.Prepare, c.PrepareResource)):
            engine.validate_preparation(command.state, len(c.quantum_nodes(command)))
