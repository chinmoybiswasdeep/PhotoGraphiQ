"""Specific, actionable error messages for invalid patterns, expressions and backends."""

import math

import numpy as np
import pytest

import photographiq as pg
from photographiq import registry
from photographiq.backends.native_gaussian import NativeGaussianBackend
from photographiq.expressions import Parameter, exp, expression
from photographiq.interactive.plots import photon_number_marginal

AVAILABLE = set(registry.available_backends())
FOCK = ("piquasso-fock", "piquasso-mixed-fock", "native-fock", "native-mixed-fock")


def test_self_cz_is_not_reported_as_a_cycle():
    pattern = pg.Pattern(inputs=(0,)).extend([pg.Prepare(1), pg.Entangle(1, 1)])
    with pytest.raises(ValueError, match=r"^Self CZ is invalid: command 1 entangles 1"):
        pattern.validate()


@pytest.mark.parametrize(
    "command", [pg.BeamSplitter(0, 0, 0.3), pg.Output((0, 0))], ids=["beamsplitter", "output"]
)
def test_repeated_mode_is_not_reported_as_a_cycle(command):
    pattern = pg.Pattern(inputs=(0,)).append(command)
    with pytest.raises(ValueError, match=r"^Command 0 \(\w+\) repeats a mode"):
        pattern.validate()


def test_genuine_cycle_still_reported():
    pattern = pg.Pattern().extend(
        [pg.Signal("a", pg.Outcome("b")), pg.Signal("b", pg.Outcome("a"))]
    )
    with pytest.raises(ValueError, match="Cyclic"):
        pattern.validate()


def _pair_pattern(measurement, value):
    pattern = pg.Pattern(inputs=(0,)).extend([pg.Prepare(1), pg.Entangle(0, 1)])
    return pattern.measure(0, measurement, key="m").displace(1, q=value)


@pytest.mark.parametrize(
    "measurement", [pg.Heterodyne(), pg.Generaldyne(np.diag([0.5, 2.0]))], ids=lambda m: str(m)
)
def test_two_component_outcome_without_component_rejected(measurement):
    name = type(measurement).__name__
    # Nested inside an expression, so the check must walk the whole tree.
    pattern = _pair_pattern(measurement, 0.5 * pg.sin(pg.Outcome("m")) + 1.0)
    with pytest.raises(ValueError, match=rf"reads the {name} outcome 'm'.*component="):
        pattern.validate()
    with pytest.raises(ValueError, match="component="):
        pg.simulate(pattern, seed=0)  # previously TypeError from float(tuple)
    for component in (0, 1):
        ok = _pair_pattern(measurement, pg.Outcome("m", component))
        result = pg.simulate(ok, seed=0)
        assert len(result.outcomes["m"]) == 2


def test_scalar_outcome_without_component_still_valid():
    pg.simulate(_pair_pattern(pg.Homodyne(), pg.Outcome("m")), seed=0)


@pytest.mark.parametrize("name", FOCK)
def test_missing_cutoff_names_the_backend(name):
    if name not in AVAILABLE:
        pytest.skip(f"{name} unavailable")
    with pytest.raises(ValueError, match=rf"^{name} requires an explicit total-photon cutoff"):
        registry.create_backend(name, None)


@pytest.mark.parametrize(
    "value",
    [
        expression(10.0) ** expression(1000.0),  # OverflowError
        exp(1000.0),  # OverflowError (math range error)
        expression(1.0) / Parameter("a"),  # ZeroDivisionError
        expression(0.0) ** expression(-1.0),  # ZeroDivisionError
    ],
)
def test_arithmetic_errors_become_documented_value_error(value):
    with pytest.raises(ValueError, match="finite real scalar") as info:
        value.evaluate({}, {"a": 0.0})
    assert isinstance(info.value.__cause__, (OverflowError, ZeroDivisionError))


def test_out_of_range_or_complex_values_become_documented_value_error():
    # An int beyond float range overflows in float(), outside the operator call.
    for value in (Parameter("a"), 2.0 * Parameter("a"), pg.Outcome("n")):
        with pytest.raises(ValueError, match="finite real scalar") as info:
            value.evaluate({"n": 10**400}, {"a": 10**400})
        assert isinstance(info.value.__cause__, OverflowError)
    # A negative base to a fractional power is complex in Python, not real.
    with pytest.raises(ValueError, match="finite real scalar"):
        (expression(-8.0) ** expression(1 / 3)).evaluate({}, {})


def test_finite_expression_unchanged():
    assert (expression(1.0) / Parameter("a")).evaluate({}, {"a": 4.0}) == 0.25
    assert exp(1.0).evaluate({}, {}) == math.e


def test_photon_counting_messages_mention_native_fock_backends():
    backend = NativeGaussianBackend()
    backend.prepare("a", 0.5)
    with pytest.raises(NotImplementedError, match="native-fock, native-mixed-fock"):
        backend.measure("a", pg.PhotonNumber())
    state = pg.GaussianState(np.zeros(2), np.eye(2), ("a",))
    with pytest.raises(NotImplementedError, match="native-fock, native-mixed-fock"):
        photon_number_marginal(state, 0)
