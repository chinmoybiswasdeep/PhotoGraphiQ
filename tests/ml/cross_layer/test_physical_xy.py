"""PhysicalMuTA with the physical logical-XY instrument family (measurement_family='XY')."""

import warnings

import numpy as np
import pytest

import photographiq as pg
from photographiqml import MuTA
from photographiqml.lowering import (
    GKPPhysicalConfig,
    UnsupportedPhysicalMeasurement,
    lower_muta_to_gkp,
    physical_capabilities,
)
from photographiqml.physical import PhysicalMuTA, compare_logical_physical

pytest.importorskip("photographiq._native")


def config(**changes):
    options = {
        "cutoff": 40,
        "peak_width": 0.6,
        "envelope": 0.6,
        "peaks": 6,
        "grid_points": 2049,
        "backend": "native-fock",
        "xy_protocol": "auto",
    }
    return GKPPhysicalConfig(**{**options, **changes})


def chain(**changes):
    return PhysicalMuTA(
        1, 1, one_column=True, measurement_family="XY", physical_config=config(**changes)
    )


def test_signed_x_default_is_unchanged():
    assert GKPPhysicalConfig().xy_protocol == "signed-x"
    model = PhysicalMuTA()
    assert model.measurement_family == "X"
    assert model.physical_config.xy_protocol == "signed-x"
    report = physical_capabilities(model, model.initialize(0))
    assert all(row["basis"] == "X" for row in report["angle_audit"])
    assert "logical XY instrument (pg.GKPLogicalXYMeasurement)" not in report["required_operations"]
    general = dict.fromkeys(model.parameters(), 0.3)
    assert len(physical_capabilities(model, general)["unsupported_nodes"]) == model.n_parameters
    with pytest.raises(UnsupportedPhysicalMeasurement, match="0 and pi"):
        lower_muta_to_gkp(model, general)


def test_family_and_protocol_must_agree():
    with pytest.raises(ValueError):
        PhysicalMuTA(measurement_family="Y")
    with pytest.raises(ValueError):
        PhysicalMuTA(measurement_family="XY", physical_config=GKPPhysicalConfig())
    with pytest.raises(ValueError):
        PhysicalMuTA(physical_config=config())
    with pytest.raises(ValueError):
        GKPPhysicalConfig(xy_protocol="rotated-homodyne")
    model = PhysicalMuTA(measurement_family="XY")
    assert model.physical_config.xy_protocol == "auto"
    assert all(high == pytest.approx(2 * np.pi) for _, high in model.angle_space().values())
    assert len(set(model.initialize(1).values())) == model.n_parameters  # continuous


def test_clifford_protocol_rejects_other_angles_before_allocation(monkeypatch):
    model = chain(xy_protocol="clifford")
    names = list(model.parameters())
    parameters = dict.fromkeys(names, 0.0) | {names[1]: 0.3}
    report = physical_capabilities(model, parameters)
    assert [row["parameter"] for row in report["unsupported_nodes"]] == [names[1]]

    def no_allocation(*args, **kwargs):
        raise AssertionError("Fock allocation attempted before rejection")

    monkeypatch.setattr(pg.GKPCode, "plus", no_allocation)
    with pytest.raises(UnsupportedPhysicalMeasurement, match="Clifford protocol"):
        lower_muta_to_gkp(model, parameters)


def test_lowering_emits_frame_adaptive_xy_instruments():
    model = chain()
    names = list(model.parameters())
    parameters = dict(zip(names, (0.3, 0.7, np.pi / 2, 1.1)))
    report = physical_capabilities(model, parameters)
    assert report["supported"] and report["xy_protocol"] == "auto"
    assert "finite GKP magic ancilla" in report["required_operations"]
    lowering = lower_muta_to_gkp(model, parameters)
    measures = [c for c in lowering.pattern.commands if isinstance(c, pg.Measure)]
    assert all(isinstance(c.measurement, pg.GKPLogicalXYMeasurement) for c in measures)
    first, second = measures[0].measurement, measures[1].measurement
    assert first.angle == pytest.approx(0.3)  # no byproducts reach the first node
    assert isinstance(second.angle, pg.CallableExpression)
    sources = second.angle.dependencies
    for x_bit in (0, 1):
        records = dict.fromkeys(sources, 0) | {("bit", model.measurement_order[0]): x_bit}
        assert second.angle.evaluate(records, {}) == pytest.approx((-1) ** x_bit * 0.7)


def test_xy_model_serialization_round_trip(tmp_path):
    model = chain()
    model.freeze([next(iter(model.parameters()))], 0.25)
    path = tmp_path / "xy.json"
    model.save(path)
    loaded = MuTA.load(path)
    assert isinstance(loaded, PhysicalMuTA)
    assert loaded.measurement_family == "XY"
    assert loaded.physical_config == model.physical_config


def run(model, angles, shots=3):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return model.run(
            [1, 0],
            dict(zip(model.parameters(), angles)),
            mode="physical-shots",
            shots=shots,
            seed=2,
        )


def test_xy_family_reproduces_signed_x_on_shared_angles():
    """Angles 0/pi through the XY instrument equal the original signed-X lowering exactly."""
    angles = (0.0, np.pi, 0.0, np.pi)
    xy = run(chain(cutoff=40), angles)
    signed = run(
        PhysicalMuTA(
            1, 1, one_column=True, physical_config=config(cutoff=40, xy_protocol="signed-x")
        ),
        angles,
    )
    # Same seed and uniform variates; the instrument inverts a tabulated CDF (linear
    # interpolation) where the generic sampler root-finds, so analog outcomes agree to ~1e-7.
    assert np.allclose(
        list(xy.decoded_joint_probabilities.values()),
        list(signed.decoded_joint_probabilities.values()),
        atol=1e-6,
        rtol=0,
    )


def test_injected_xy_chain_executes_with_heralded_records():
    result = run(chain(cutoff=56, max_dimension=200_000), (0.0, 0.0, np.pi / 4, 0.0))
    probabilities = np.array(list(result.decoded_joint_probabilities.values()))
    assert probabilities.min() >= -1e-9 and probabilities.sum() == pytest.approx(1, abs=1e-9)
    records = [r for shot in result.measurement_records for r in shot.values()]
    assert all(isinstance(r, pg.LogicalXYMeasurementResult) for r in records)
    assert all(r.heralded_success for r in records)
    assert any(r.rounds for r in records)  # the non-Clifford node used a magic ancilla
    assert 0 <= compare_logical_physical(result)["total_variation_distance"] <= 1
