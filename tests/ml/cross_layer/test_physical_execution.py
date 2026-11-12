"""Finite Fock integration checks; broad resources keep these contract tests small."""

from dataclasses import replace

import numpy as np
import pytest

import photographiq as pg
from photographiqml import GKPBridge, GKPPhysicalConfig, MuTA, PhysicalMuTA
from photographiqml.physical import compare_logical_physical, validate_physical_model


@pytest.fixture
def config():
    return GKPPhysicalConfig(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)


def test_conditional_against_independent_public_pattern(config):
    model = PhysicalMuTA(1, physical_config=config)
    angles = [0, np.pi, 0, np.pi]
    analog = dict.fromkeys(model.measurement_order, 0.0)
    result = model.run([1, 0], angles, mode="physical-conditional", analog_outcomes=analog)
    code = config.code()
    # Independent linear-chain construction, with no downstream lowering helper.
    pattern = pg.Pattern(inputs=(0,))
    for i, angle in enumerate(angles):
        pattern.append(pg.Prepare(i + 1, state=code.plus()))
        pattern.append(code.logical_cz(i, i + 1))
        pattern.append(pg.Measure(i, code.logical_measurement("XY", alpha=angle), i))
    pattern.append(pg.Output((4,)))
    direct = pg.simulate(
        pattern,
        inputs={0: code.zero()},
        cutoff=config.cutoff,
        backend=config.backend,
        measurement_outcomes=dict.fromkeys(range(4), 0.0),
    )
    assert np.allclose(result.physical_result.state.density_matrix, direct.state.density_matrix)
    assert sum(result.decoded_joint_probabilities.values()) == pytest.approx(1)
    assert result.logical_target.representation == "logical"
    assert not hasattr(result, "state")
    assert result.standard_errors is None and result.empirical_probabilities is None
    assert result.diagnostics["decoder_confidence"] is None
    assert result.diagnostics["joint_code_subspace_leakage"] is None
    assert result.diagnostics["backend_diagnostics"][0]
    assert compare_logical_physical(result)["finite_state_fidelity"] is None
    for record in result.measurement_records[0].values():
        assert record.confidence is None and record.probabilities is None


def test_physical_shots_repeatability_and_error_estimates(config):
    model = PhysicalMuTA(physical_config=config)
    a = model.run([1, 0], shots=3, seed=14, output_basis="X")
    b = GKPBridge().run(model, [1, 0], shots=3, seed=14, output_basis="X")
    assert a.sampled_output_bits == b.sampled_output_bits
    assert a.decoded_joint_probabilities == b.decoded_joint_probabilities
    assert len(a.physical_result.trajectories) == len(a.measurement_records) == 3
    assert sum(a.empirical_probabilities.values()) == pytest.approx(1)
    assert all(v >= 0 for v in a.standard_errors.values())
    assert all(v >= 0 for v in a.empirical_standard_errors.values())
    assert 0 <= compare_logical_physical(a)["total_variation_distance"] <= 1
    single = model.run([1, 0], shots=1, seed=14)
    assert single.standard_errors is single.empirical_standard_errors is None


def test_two_wire_joint_readout_retains_correlations(config):
    model = PhysicalMuTA(2, physical_config=config)
    result = model.run(
        [1, 0, 0, 0],
        mode="physical-conditional",
        analog_outcomes=dict.fromkeys(model.measurement_order, 0.0),
    )
    direct = pg.multimode_readout(
        result.physical_result.state,
        dict.fromkeys(model.output_nodes, "Z"),
        frames=result.frames[0],
    )
    assert result.decoded_joint_probabilities == direct["joint_probabilities"]
    p = result.decoded_joint_probabilities
    marginals = list(result.decoded_marginals.values())
    assert abs(p[(0, 0)] - marginals[0][0] * marginals[1][0]) > 1e-5
    assert result.diagnostics["pair_correlations"][(0, 1)] == pytest.approx(
        p[(0, 0)] + p[(1, 1)] - p[(0, 1)] - p[(1, 0)]
    )


def test_hard_soft_resource_decoder_and_resource_study(config):
    bridge = GKPBridge(cutoff=24, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)
    hard = bridge.resource_readout().decoder.decode(0.2)
    soft = bridge.resource_readout(decoder="soft").decoder.decode(0.2)
    assert hard.confidence is hard.probabilities is None
    assert sum(soft.probabilities) == pytest.approx(1)
    assert soft.confidence == max(soft.probabilities)
    with pytest.raises(ValueError):
        bridge.resource_readout("Y")
    study = bridge.measurement_convergence([20, 24])
    assert len(study.rows) == 2 and study.axis == "cutoff"


def test_independent_whole_model_study(config):
    model = PhysicalMuTA(physical_config=config)
    study = validate_physical_model(
        model,
        [1, 0],
        [513, 1025],
        axis="grid_points",
        mode="physical-conditional",
        analog_outcomes=dict.fromkeys(model.measurement_order, 0.0),
    )
    assert not study["certified"] and study["axis_type"] == "numerical refinement"
    assert study["rows"][0]["max_probability_delta"] is None
    assert study["rows"][1]["max_probability_delta"] < 1e-6
    for axis, values in [("bad", [1, 2]), ("cutoff", [24]), ("cutoff", [24, 20])]:
        with pytest.raises(ValueError):
            model.physical_convergence([1, 0], values, axis=axis)


def test_physical_interface_boundaries(config, monkeypatch):
    model = PhysicalMuTA(physical_config=config)
    assert set(model.initialize(3).values()) <= {0, np.pi}
    assert all(v == (0, np.pi) for v in model.angle_space().values())
    for operation in [
        lambda: model.initialize(scale=0.1),
        lambda: PhysicalMuTA(measurement_family="Y"),
        lambda: PhysicalMuTA(representation="gkp"),
        lambda: PhysicalMuTA(physical_config="bad"),
    ]:
        with pytest.raises((ValueError, TypeError)):
            operation()
    for operation in [lambda: model.unitary(), lambda: model.run_batch([[1, 0]])]:
        with pytest.raises(NotImplementedError):
            operation()
    monkeypatch.setattr(pg.GKPCode, "plus", lambda *a: pytest.fail("allocation"))
    for seed in [-1, True, "bad"]:
        with pytest.raises(ValueError, match="seed"):
            model.run([1, 0], seed=seed)
    # Every point is audited before a convergence sweep begins.
    with pytest.raises(ValueError, match="budget"):
        model.physical_convergence([1, 0], [24, 1000])
    with pytest.raises(NotImplementedError, match="0 and pi"):
        GKPBridge().run(MuTA(1), [1, 0], [0, 0, 0.1, 0], config=config)


def test_mixed_backend_fails_before_projection(config, monkeypatch):
    monkeypatch.setattr(pg.GKPResource, "project", lambda *a: pytest.fail("allocation"))
    with pytest.raises(ValueError, match="mixed Fock lowering remains unsupported"):
        replace(config, backend="piquasso-mixed-fock")


def test_draw_capability_failure_without_projection(config, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(pg.GKPCode, "plus", lambda *a: pytest.fail("allocation"))
    model = PhysicalMuTA(physical_config=config)
    ax = model.draw_physical({"alpha.w0.c3": 0.37})
    assert any("unsupported XY" in text.get_text() for text in ax.texts)
    plt.close(ax.figure)
