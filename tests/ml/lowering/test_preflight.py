from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

import photographiq as pg
from photographiqml import GKPBridge, GKPPhysicalConfig, MuTA, PhysicalMuTA, lower_muta_to_gkp
from photographiqml.lowering import UnsupportedPhysicalMeasurement, product_input


@pytest.mark.parametrize("angle", [0.37, 1e-5, 1e-12, np.pi / 2, np.pi / 4, np.nan])
def test_reject_every_angle_before_any_fock_allocation(monkeypatch, angle):
    model = MuTA(2, 2)
    name = model.parameter_name(model.measurement_order[-1])
    poison = Mock(side_effect=AssertionError("Fock allocation was reached"))
    monkeypatch.setattr(pg.GKPCode, "plus", poison)
    monkeypatch.setattr(pg.GKPResource, "project", poison)
    monkeypatch.setattr(pg, "simulate", poison)
    monkeypatch.setattr(pg, "run_shots", poison)
    with pytest.raises((UnsupportedPhysicalMeasurement, ValueError)):
        GKPBridge().run(model, [1, 0, 0, 0], {name: angle})
    poison.assert_not_called()


@pytest.mark.parametrize(
    "angle,flip", [(0, 0), (np.pi, 1), (-np.pi, 1), (2 * np.pi, 0), (1e-15, 0)]
)
def test_authoritative_upstream_tolerance(angle, flip):
    model = MuTA(1)
    audit = model.physical_capabilities({"alpha.w0.c0": angle})
    assert audit["supported"]
    assert audit["angle_audit"][0]["flip"] == flip


@pytest.mark.parametrize(
    "config",
    [
        GKPPhysicalConfig(max_dimension=1),
        GKPPhysicalConfig(max_matrix_bytes=1),
        GKPPhysicalConfig(decoder="soft"),
        GKPPhysicalConfig(grid_points=100_000_001),
    ],
)
def test_resource_and_decoder_guards_preallocation(monkeypatch, config):
    poison = Mock(side_effect=AssertionError("projection"))
    monkeypatch.setattr(pg.GKPResource, "project", poison)
    with pytest.raises(ValueError):
        lower_muta_to_gkp(MuTA(2), config=config)
    poison.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"mode": "bad"},
        {"shots": 0},
        {"output_basis": "Y"},
        {"mode": "physical-conditional"},
        {"analog_outcomes": {}},
        {"mode": "physical-conditional", "analog_outcomes": {(0, c): np.inf for c in range(4)}},
    ],
)
def test_execution_request_guards_preallocation(monkeypatch, options):
    poison = Mock(side_effect=AssertionError("projection"))
    monkeypatch.setattr(pg.GKPResource, "project", poison)
    with pytest.raises((ValueError, NotImplementedError)):
        PhysicalMuTA().run([1, 0], **options)
    poison.assert_not_called()


def test_entangled_input_is_not_product_encoded(monkeypatch):
    poison = Mock(side_effect=AssertionError("projection"))
    monkeypatch.setattr(pg.GKPResource, "project", poison)
    with pytest.raises(NotImplementedError, match="Entangled"):
        PhysicalMuTA(2).run(np.array([1, 0, 0, 1]) / np.sqrt(2))
    poison.assert_not_called()
    original = np.kron([1, 0], np.array([1, 1j]) / np.sqrt(2))
    joint, factors = product_input(original, 2)
    assert np.allclose(joint, np.kron(*factors))
    assert np.allclose(product_input([[1, 0], [0, 1]], 2)[0], [0, 1, 0, 0])


def test_lowering_semantic_maps_and_commands():
    model = PhysicalMuTA(2, one_column=True)
    lowering = lower_muta_to_gkp(model, {"alpha.w1.c1": np.pi})
    assert lowering.pattern.inputs == model.input_nodes
    assert lowering.pattern.outputs == model.output_nodes
    edges = [frozenset((c.u, c.v)) for c in lowering.pattern.commands if isinstance(c, pg.Entangle)]
    assert set(edges) == {frozenset(e) for e in model.graph.edges}
    assert len(edges) == len(model.graph.edges)
    assert all(u == v for u, v in lowering.node_map.items())
    assert len(lowering.measurement_keys) == 8
    assert lowering.audit["peak_live_modes"] == 3
    assert not any(isinstance(c, pg.Displace) for c in lowering.pattern.commands)
    assert len([c for c in lowering.pattern.commands if isinstance(c, pg.Signal)]) == 8
    assert (
        lower_muta_to_gkp(
            PhysicalMuTA(physical_config=GKPPhysicalConfig(cutoff=24))
        ).resource_config.cutoff
        == 24
    )


def test_config_validation_and_roundtrip():
    config = GKPPhysicalConfig()
    assert GKPPhysicalConfig(**config.to_dict()) == config
    for options in (
        {"cutoff": 1},
        {"backend": "gaussian"},
        {"decoder": "invented"},
        {"max_dimension": True},
        {"grid_points": 32},
    ):
        with pytest.raises(ValueError):
            replace(config, **options)
