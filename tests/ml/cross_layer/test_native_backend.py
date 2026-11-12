"""Physical MuTA on backend='native-fock' against the default backend='piquasso-fock'.

The engines differ only in how a CZ is truncated: native-fock applies the exact projected
P U P, piquasso-fock applies Euler factors and truncates after each one. Both converge to
the same state. The tolerances are the scale of each engine's own truncation error at
cutoff 32, measured as the decoded-probability distance to native-fock at cutoff 44
(2026-11-11): 1 wire, seeds 0-2: Piquasso 4.2e-6/1.7e-5/6.5e-5, native 3.4e-6/1.6e-5/
6.4e-5; 2 wires: Piquasso 5.5e-5/7.8e-5/6.4e-5, native 4.9e-5/8.4e-5/1.0e-4. Observed
engine differences at cutoff 32: max |dP| 7.6e-6 (1 wire, Z), 7.9e-5 (1 wire, X),
9.0e-5 (2 wires); final-state infidelity <= 1.5e-7; decoded bits and sampled outputs
identical in every case.
"""

from dataclasses import replace

import numpy as np
import pytest

from photographiq import native

try:
    native.require("fock")
except ImportError as exc:  # pragma: no cover - exercised only without the extension
    pytest.skip(f"native fock component unavailable: {exc}", allow_module_level=True)

pytest.importorskip("piquasso")

from photographiq.fock_analysis import fidelity  # noqa: E402
from photographiqml import GKPPhysicalConfig, PhysicalMuTA  # noqa: E402

BASE = GKPPhysicalConfig(cutoff=32, peak_width=0.9, envelope=0.9, peaks=4, grid_points=1025)


def _run(wires, backend, seed, mode, basis="Z"):
    model = PhysicalMuTA(wires, physical_config=replace(BASE, backend=backend))
    rng = np.random.default_rng(seed)
    params = dict(zip(model.trainable_parameters(), rng.choice([0.0, np.pi], model.n_parameters)))
    x = rng.normal(size=(wires, 2))
    state = x / np.linalg.norm(x, axis=1, keepdims=True)
    if mode == "physical-conditional":
        outcomes = rng.normal(0, 0.4, len(model.measurement_order))
        analog = dict(zip(model.measurement_order, map(float, outcomes)))
        return model.run(state, params, mode=mode, analog_outcomes=analog, output_basis=basis)
    return model.run(state, params, mode=mode, shots=3, seed=seed, output_basis=basis)


def _trajectories(result):
    if result.mode == "physical-shots":
        return result.physical_result.trajectories
    return [result.physical_result]


def _states(result):
    return [t.state for t in _trajectories(result)]


def _assert_engine(result, config_name, engine, state_module):
    # resource_config only echoes the request; Result.backend is type(engine).__name__ of the
    # backend that executed, and the state object comes from that engine.
    assert result.resource_config.backend == config_name
    for trajectory in _trajectories(result):
        assert trajectory.backend == engine
        assert type(trajectory.state.native).__module__.startswith(state_module)


def _compare(reference, result, probability_tol, infidelity_tol):
    _assert_engine(reference, "piquasso-fock", "PiquassoFockBackend", "piquasso")
    _assert_engine(result, "native-fock", "NativeFockBackend", "photographiq.backends.native_fock")
    p = np.array(list(reference.decoded_joint_probabilities.values()))
    n = np.array(list(result.decoded_joint_probabilities.values()))
    assert np.abs(p - n).max() < probability_tol
    for a, b in zip(_states(reference), _states(result), strict=True):
        assert 1 - fidelity(a, b) < infidelity_tol
    bits = [{k: v.bit for k, v in rec.items()} for rec in reference.measurement_records]
    assert bits == [{k: v.bit for k, v in rec.items()} for rec in result.measurement_records]
    assert reference.sampled_output_bits == result.sampled_output_bits
    assert sum(n) == pytest.approx(1, abs=1e-12)


def test_config_accepts_native_fock_and_keeps_piquasso_default():
    assert GKPPhysicalConfig().backend == "piquasso-fock"
    assert replace(BASE, backend="native-fock").backend == "native-fock"
    with pytest.raises(ValueError, match="mixed Fock lowering remains unsupported"):
        replace(BASE, backend="native-mixed-fock")


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_one_wire_conditional_matches_piquasso(seed, basis):
    reference = _run(1, "piquasso-fock", seed, "physical-conditional", basis)
    result = _run(1, "native-fock", seed, "physical-conditional", basis)
    _compare(reference, result, probability_tol=1e-4, infidelity_tol=2e-6)
    diag = result.diagnostics["backend_diagnostics"][0]
    assert diag and all("retained_norm" in d for d in diag if "boundary_population" in d)


def test_one_wire_shots_sample_identically():
    reference = _run(1, "piquasso-fock", 0, "physical-shots")
    result = _run(1, "native-fock", 0, "physical-shots")
    _compare(reference, result, probability_tol=1e-4, infidelity_tol=2e-6)
    assert reference.empirical_probabilities == result.empirical_probabilities


def test_two_wire_conditional_matches_piquasso():
    reference = _run(2, "piquasso-fock", 2, "physical-conditional")
    result = _run(2, "native-fock", 2, "physical-conditional")
    _compare(reference, result, probability_tol=2e-4, infidelity_tol=2e-6)


def test_validate_physical_model_builds_default_config_only_when_needed(monkeypatch):
    """The GKPPhysicalConfig() fallback is lazy: never built when the model carries a config."""
    from types import SimpleNamespace

    from photographiqml import physical

    built = []

    def default():
        built.append(1)
        return BASE

    monkeypatch.setattr(physical, "GKPPhysicalConfig", default)
    # An unknown axis raises right after config resolution, before any execution.
    with pytest.raises(ValueError, match="Unknown convergence axis"):
        physical.validate_physical_model(
            SimpleNamespace(physical_config=BASE), None, (1, 2), axis="x"
        )
    with pytest.raises(ValueError, match="Unknown convergence axis"):
        physical.validate_physical_model(SimpleNamespace(), None, (1, 2), axis="x", config=BASE)
    assert built == []
    with pytest.raises(ValueError, match="Unknown convergence axis"):
        physical.validate_physical_model(SimpleNamespace(), None, (1, 2), axis="x")
    assert built == [1]
