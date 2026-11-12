"""Allocation-free pattern analysis: required features and execution preflight.

``required_features`` names every backend capability a pattern needs without
constructing a backend. Besides class-level capability names it may contain two
*derived* requirements that are not backend capability strings:

``mixed_gaussian_input``
    A ``GaussianInput`` with det(V) != 1. Gaussian backends and mixed-Fock backends
    accept it; pure-Fock backends cannot represent it.
``correlated_gaussian_state``
    A correlated ``GaussianState`` passed as ``initial_state``; only
    ``GaussianBackend`` subclasses accept it.
"""

import numpy as np

from . import commands as c
from .resources import CatResource, CubicPhaseResource
from .states import FockDensityMatrix, FockInput, FockSuperposition, GaussianInput, GaussianState

MIXED_GAUSSIAN_INPUT = "mixed_gaussian_input"
CORRELATED_GAUSSIAN_STATE = "correlated_gaussian_state"
DERIVED_FEATURES = frozenset({MIXED_GAUSSIAN_INPUT, CORRELATED_GAUSSIAN_STATE})

_FEATURE_COMMANDS = {
    c.CubicPhase: "cubic_phase",
    c.Kerr: "kerr",
    c.QuadraticPhase: "quadratic_phase",
    c.PhotonAdd: "photon_addition",
    c.PhotonSubtract: "photon_subtraction",
    c.Loss: "loss",
    c.PrepareResource: "multimode_fock",
}


def _is_mixed_gaussian(state) -> bool:
    if not isinstance(state, GaussianInput):
        return False
    try:
        determinant = np.linalg.det(state.covariance)
    except (ValueError, np.linalg.LinAlgError):
        return False  # malformed; validate_preparation rejects it later
    return not np.isclose(determinant, 1, atol=1e-10, rtol=0)


def _scan(pattern, inputs, initial_state):
    """Return (capability features, measurement keys, input states) without allocation."""
    from .gkp import GKPResource

    features: set[str] = set()
    keys: set = set()
    states = list(inputs.values())
    if initial_state is not None:
        states.append(initial_state)
    for command in pattern.commands:
        if type(command) in _FEATURE_COMMANDS:
            features.add(_FEATURE_COMMANDS[type(command)])
        if isinstance(command, (c.Prepare, c.PrepareResource)):
            states.append(command.state)
        if isinstance(command, c.Measure):
            keys.add(command.result_key)
            features.update(command.measurement.required_capabilities)
    for state in states:
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
    return features, keys, states


def _check_postselection(features, keys, measurement_outcomes):
    if measurement_outcomes:
        if not measurement_outcomes.keys() <= keys:
            raise ValueError("Postselection key does not identify a measurement")
        features.add("postselection")


def required_features(
    pattern, inputs=None, initial_state=None, measurement_outcomes=None
) -> frozenset[str]:
    """Features a backend must provide to execute ``pattern``; nothing is allocated.

    Args:
        pattern (Pattern): Measurement pattern (not validated here).
        inputs (dict): Individual input descriptions keyed by input labels.
        initial_state (object): Correlated input state, as accepted by ``simulate``.
        measurement_outcomes (dict): Postselected outcomes keyed by result key.

    Returns:
        result (frozenset): Backend capability names plus the derived requirements
            documented in this module.

    Raises:
        ValueError: Postselection key does not identify a measurement.
    """
    features, keys, states = _scan(pattern, dict(inputs or {}), initial_state)
    _check_postselection(features, keys, measurement_outcomes)
    if any(_is_mixed_gaussian(state) for state in states):
        features.add(MIXED_GAUSSIAN_INPUT)
    if isinstance(initial_state, GaussianState):
        features.add(CORRELATED_GAUSSIAN_STATE)
    return frozenset(features)


def preflight(engine, pattern, inputs, initial_state=None, measurement_outcomes=None):
    """Check capabilities and resources against ``engine`` before any state allocation."""
    features, keys, states = _scan(pattern, inputs, initial_state)
    if (
        engine.supports("fock_input")
        and not engine.supports("mixed_fock")
        and any(_is_mixed_gaussian(state) for state in states)
    ):
        raise NotImplementedError("Mixed Gaussian inputs require a mixed Fock backend")
    _check_postselection(features, keys, measurement_outcomes)
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
