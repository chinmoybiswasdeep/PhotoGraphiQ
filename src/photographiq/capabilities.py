"""Preflight the whole pattern before allocating or mutating simulator state."""

import numpy as np

from . import commands as c
from .measurements import Generaldyne, Heterodyne, Homodyne, PhotonNumber
from .resources import CatResource, CubicPhaseResource
from .states import FockInput, FockSuperposition, GaussianInput


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
            if isinstance(measurement, Homodyne):
                features.add("homodyne")
                if measurement.efficiency != 1 or measurement.noise != 0:
                    features.add("noisy_homodyne")
            else:
                features.add(
                    {
                        PhotonNumber: "photon_counting",
                        Heterodyne: "heterodyne",
                        Generaldyne: "generaldyne",
                    }[type(measurement)]
                )
    for state in states:
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
            and isinstance(state, GaussianInput)
            and not np.isclose(np.linalg.det(state.covariance), 1)
        ):
            raise NotImplementedError("Mixed Gaussian inputs require a mixed Fock backend")
    if measurement_outcomes:
        if not measurement_outcomes.keys() <= keys:
            raise ValueError("Postselection key does not identify a measurement")
        features.add("postselection")
    engine.require(*features)
